import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import math
import os

# --- Configuration ---
CELL_CAPACITY_AH = 3.5 
PARALLEL_STRINGS = 12
SERIES_CELLS = 34
PACK_CAPACITY_AH = CELL_CAPACITY_AH * PARALLEL_STRINGS # 42.0 Ah
SECONDS_PER_HOUR = 3600.0

# --- Parameters ---
# R0 approx 25mOhm/cell * 34 / 12 = ~0.07 Ohm
R0_PACK_OHMS = 0.07 
R1_PACK_OHMS = 0.035 
C1_PACK_FARAD = 285.0 
OCV_CSV_PATH = 'curve-compute/processed_red_curve_data.csv'


# --- Improved OCV Mapping for Molicel INR-18650-M35A ---
# The previous map was too optimistic at low voltages (e.g. 3.7V is not 80%).
# This curve is more representative of a standard high-capacity 18650.
# --- Improved OCV Mapping (Based on M35A 0.2C / Red Line) ---
# This curve assumes low-current "Cruising" voltage (approx 0.7A per cell).
# It is "stiffer" (higher voltage for same SOC) than the previous map.
try:
    
    print(f"Loading OCV curve from: {OCV_CSV_PATH}")
    ocv_df = pd.read_csv(OCV_CSV_PATH)
    
    # Extract SOC (0-1 -> 0-100%) and Voltage
    # CSV Columns: 'Capacity (mAh)', 'Voltage (V)', 'State of Charge'
    soc_points = ocv_df['State of Charge'].values * 100.0
    voltage_points = ocv_df['Voltage (V)'].values
    
    # Create the OCV_DATA array [SOC%, Voltage]
    OCV_DATA = np.column_stack((soc_points, voltage_points))
    
    # Ensure data is sorted by SOC (ascending) for interpolation
    # The CSV usually goes High Voltage -> Low Voltage (Discharge), so SOC goes 100 -> 0.
    # We need to sort it 0 -> 100 for interp1d.
    sort_indices = np.argsort(OCV_DATA[:, 0])
    OCV_DATA = OCV_DATA[sort_indices]

except Exception as e:
    print(f"Error loading OCV CSV: {e}")
    print("Falling back to hardcoded OCV map (Less Accurate)")
    OCV_DATA = np.array([
        [0.0,   2.50],
        [2.5,   2.80],
        [5.0,   2.98],
        [14.2,  3.05],
        [28.57, 3.33],
        [42.9,  3.50],
        [50.0,  3.55],
        [57.1,  3.60],
        [71.4,  3.75],
        [85.7,  3.87],
        [90.0,  4.04],
        [95.0,  4.08],
        [100.0, 4.20]
    ])


# Prepare Interpolators
OCV_SOC_MAP_PERCENT = OCV_DATA[:, 0]
CELL_VOLTAGE_MAP = OCV_DATA[:, 1]
OCV_PACK_MAP_VOLTS = CELL_VOLTAGE_MAP * SERIES_CELLS

# Sort for interpolation stability
sort_idx = np.argsort(OCV_SOC_MAP_PERCENT)
OCV_SOC_MAP_PERCENT = OCV_SOC_MAP_PERCENT[sort_idx]
OCV_PACK_MAP_VOLTS = OCV_PACK_MAP_VOLTS[sort_idx]

ocv_to_soc_interp = interp1d(OCV_PACK_MAP_VOLTS, OCV_SOC_MAP_PERCENT, kind='linear', fill_value="extrapolate")
soc_to_ocv_interp = interp1d(OCV_SOC_MAP_PERCENT, OCV_PACK_MAP_VOLTS, kind='linear', fill_value="extrapolate")

class EKF_SOCEstimator:
    def __init__(self, initial_soc_percent, initial_vrc1_volt, dt_sec=0.5):
        self.dt = dt_sec
        self.Qn_Ah = PACK_CAPACITY_AH
        self.soc_to_ocv = soc_to_ocv_interp
        
        initial_soc = initial_soc_percent / 100.0 
        self.x = np.array([[initial_soc], [initial_vrc1_volt]])  
        
        # Initialization Confidence
        self.P = np.diag([1e-3, 1e-4])
        
        # Process Noise (Trust Coulomb Counting generally, but allow drift)
        self.Q = np.diag([1e-8, 1e-6]) 
        
        # Measurement Noise (Voltage Sensor Noise)
        self.R = np.array([[0.05**2]]) 

        self.A_rc = math.exp(-self.dt / (R1_PACK_OHMS * C1_PACK_FARAD))
        self.B_rc = R1_PACK_OHMS * (1 - self.A_rc)
        
        self.last_current = 0.0

    def get_ocv_prime(self):
        delta = 0.001
        soc_est = self.x[0, 0] * 100.0
        ocv_plus = self.soc_to_ocv(min(100.0, soc_est + delta))
        ocv_minus = self.soc_to_ocv(max(0.0, soc_est - delta))
        return (ocv_plus - ocv_minus) / (2 * delta)

    def predict(self, current_amp):
        # Current Positive = Discharge (as per equations below)
        I = current_amp
        
        # SOC Prediction (Coulomb Counting)
        # x_minus = x - (I * dt / Q)  => Positive I reduces SOC
        x_minus = np.zeros((2, 1))
        x_minus[0, 0] = self.x[0, 0] - (I * self.dt) / (self.Qn_Ah * SECONDS_PER_HOUR)
        x_minus[0, 0] = np.clip(x_minus[0, 0], 0.0, 1.0)
        
        # V_RC1 Prediction
        x_minus[1, 0] = self.x[1, 0] * self.A_rc + I * self.B_rc
        
        A_jac = np.array([
            [1.0, 0.0],
            [0.0, self.A_rc]
        ])
        
        P_minus = A_jac @ self.P @ A_jac.T + self.Q
        self.x = x_minus
        self.P = P_minus
        self.last_current = I

    def correct(self, measured_voltage):
        ocv_pred = self.soc_to_ocv(self.x[0, 0] * 100.0)
        
        # Voltage Prediction: V = OCV - V_rc - I*R0
        # Positive I (Discharge) -> Voltage Sags (Lower than OCV)
        V_t_pred = ocv_pred - self.x[1, 0] - self.last_current * R0_PACK_OHMS
        
        y = measured_voltage - V_t_pred
        
        ocv_prime = self.get_ocv_prime()
        C_jac = np.array([[ocv_prime * 100, -1.0]])
        
        S = C_jac @ self.P @ C_jac.T + self.R
        K = self.P @ C_jac.T @ np.linalg.inv(S)
        
        self.x = self.x + K * y
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        
        I_matrix = np.eye(self.x.shape[0])
        self.P = (I_matrix - K @ C_jac) @ self.P
        
        return self.x[0, 0] * 100.0

def run_ekf_simulation(file_path):
    print("Loading and processing data...")
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        'riedon.riedon_current_mA': 'Current_mA',
        'bms.pack_voltage_V': 'Voltage_V',
        'time': 'Timestamp'
    })
    
    # Flexible date parsing
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    except:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Drop missing values
    df = df.dropna(subset=['Current_mA', 'Voltage_V'])
    
    # --- FIX 1: Filter Outliers ---
    # The data contains garbage values (e.g. 1 billion mA)
    # Filter to keep realistic currents (+/- 200A)
    valid_mask = (df['Current_mA'] < 200000) & (df['Current_mA'] > -200000)
    df = df[valid_mask].copy()

    # --- FIX 2: Correct Sign Convention ---
    # Analysis shows 'Negative' current in CSV corresponds to voltage drop (Discharge).
    # The EKF equations expect 'Positive' current for Discharge.
    # We invert the sign and convert mA to Amps.
    df['Current_A'] = -1.0 * df['Current_mA'] / 1000.0
    
    # Calculate time step
    df['dt'] = df['Timestamp'].diff().dt.total_seconds().fillna(0.5)
    
    # Initialize State
    first_current = df['Current_A'].iloc[0]
    first_voltage = df['Voltage_V'].iloc[0]
    
    # Estimate Initial OCV: V + I*R (compensate for initial sag)
    initial_ocv = first_voltage + first_current * R0_PACK_OHMS
    initial_soc_percent = ocv_to_soc_interp(initial_ocv)
    initial_soc_percent = np.clip(initial_soc_percent, 0.0, 100.0)
    
    ekf = EKF_SOCEstimator(initial_soc_percent, 0.0, dt_sec=0.5)
    
    results = []
    cumulative_ah_discharged = 0.0
    
    print(f"Initial SOC Estimated: {initial_soc_percent:.2f}%")
    
    for _, row in df.iterrows():
        current = row['Current_A']
        voltage = row['Voltage_V']
        dt = row['dt']
        
        # 1. EKF Update
        ekf.dt = dt
        ekf.predict(current)
        soc_estimate = ekf.correct(voltage)

        # 2. Coulomb Counter Update (Simple)
        # Current is Positive for Discharge.
        # Track total Ah removed.
        ah_step = current * (dt / SECONDS_PER_HOUR)
        cumulative_ah_discharged += ah_step
        
        # --- FIX 3: Removed the 'Reset if > 0.1' logic ---
        # SOC = Start - (Discharged / Capacity)
        soc_cc = initial_soc_percent - (cumulative_ah_discharged / ekf.Qn_Ah) * 100.0
        soc_cc = np.clip(soc_cc, 0.0, 100.0)

        # 3. Pure ECM OCV-based SoC Update
        # Calculates SoC by reversing the voltage drop equation: OCV = V_meas + V_rc + I*R0
        # This gives the "instantaneous" SoC based on the voltage curve and impedance model, 
        # ignoring the EKF's historical filtering/covariance smoothing for the SoC state itself.
        v_rc_estimated = ekf.x[1, 0]
        ecm_ocv = voltage + (current * R0_PACK_OHMS) + v_rc_estimated
        soc_ecm_ocv = ocv_to_soc_interp(ecm_ocv)
        soc_ecm_ocv = np.clip(soc_ecm_ocv, 0.0, 100.0)
        
        results.append({
            'Timestamp': row['Timestamp'],
            'Current_A': current,
            'Voltage_V': voltage,
            'SOC_EKF_%': soc_estimate,
            'SOC_CC_%': soc_cc,
            'SOC_ECM_OCV_%': soc_ecm_ocv
        })

    results_df = pd.DataFrame(results)
    print("EKF simulation complete.")
    return results_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EKF simulation on battery data.")
    parser.add_argument("-i", "--input", default="FSGP_day1.csv", help="Input CSV file path")
    parser.add_argument("-o", "--output", default="ekf_soc_results.csv", help="Output CSV file path")
    
    args = parser.parse_args()

    try:
        # Note: Ensure 'run_ekf_simulation' is defined or imported before running this block
        results = run_ekf_simulation(args.input)
        print("\n--- Final Results ---")
        print(results.tail())
        
        results.to_csv(args.output, index=False)
        print(f"\nSaved to '{args.output}'")
    except FileNotFoundError:
        print(f"File not found: {args.input}")
    except NameError:
        print("Error: 'run_ekf_simulation' is not defined. Please import or define the simulation function.")