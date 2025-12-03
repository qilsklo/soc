import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import math

# --- Configuration & Constants ---
# Matches kalman3.py parameters
CELL_CAPACITY_AH = 3.5 
PARALLEL_STRINGS = 12
SERIES_CELLS = 34
PACK_CAPACITY_AH = CELL_CAPACITY_AH * PARALLEL_STRINGS # 42.0 Ah
SECONDS_PER_HOUR = 3600.0

# Circuit Parameters (Estimated)
R0_PACK_OHMS = 0.07 
R1_PACK_OHMS = 0.035 
C1_PACK_FARAD = 285.0 

# --- OCV Mapping for Molicel INR-18650-M35A ---
# (Copied from kalman3.py)
OCV_DATA = np.array([
    # SOC (%) | Voltage (V)
    [100, 4.20],
    [95,  4.14],
    [90,  4.07],
    [80,  4.00],
    [70,  3.93],
    [60,  3.84],
    [50,  3.74],
    [40,  3.66],
    [30,  3.56],
    [20,  3.42],
    [10,  3.25],
    [5,   3.10],
    [0,   2.50]
])

class BatteryEKF:
    def __init__(self, capacity_ah, dt_initial=1.0):
        self.Qn_Ah = capacity_ah
        self.Qn_Coulombs = capacity_ah * 3600.0
        
        # OCV Interpolator (SOC 0-1 -> Voltage)
        self.ocv_func = interp1d(
            OCV_DATA[:, 0] / 100.0, 
            OCV_DATA[:, 1], 
            kind='cubic', 
            fill_value="extrapolate"
        )
        
        # State Vector [SOC, V1]
        self.x = np.array([[0.8], [0.0]]) 
        
        # Covariance Matrix P
        self.P = np.array([
            [0.01, 0],
            [0, 0.01]
        ])
        
        # Process Noise Q
        self.Q = np.array([
            [1e-5, 0],
            [0, 1e-4]
        ])
        
        # Measurement Noise R
        self.R = 0.1
        
        self.dt = dt_initial

    def get_ocv(self, soc):
        # Scale Pack Voltage based on Series Cells
        return self.ocv_func(soc) * SERIES_CELLS

    def get_ocv_derivative(self, soc):
        delta = 0.001
        v_plus = self.get_ocv(soc + delta)
        v_minus = self.get_ocv(soc - delta)
        return (v_plus - v_minus) / (2 * delta)

    def predict(self, current_amp):
        """
        Prediction Step: Propagate state forward based on Current.
        current_amp: Positive = Discharge, Negative = Charge
        """
        soc_prev = self.x[0, 0]
        v1_prev = self.x[1, 0]
        
        # --- State Transition ---
        # 1. SOC Integration: SOC_new = SOC_old - (I * dt) / Qn
        new_soc = soc_prev - (current_amp * self.dt) / self.Qn_Coulombs
        
        # Clip SOC to physical limits for stability
        new_soc = np.clip(new_soc, 0.0, 1.0)
        
        # 2. RC Voltage Dynamics
        alpha = np.exp(-self.dt / (R1_PACK_OHMS * C1_PACK_FARAD))
        new_v1 = v1_prev * alpha + R1_PACK_OHMS * current_amp * (1 - alpha)
        
        self.x = np.array([[new_soc], [new_v1]])
        
        # --- Jacobian A (State Transition Matrix) ---
        A = np.array([
            [1.0, 0.0],
            [0.0, alpha]
        ])
        
        # Predict Covariance: P = A*P*A.T + Q
        self.P = A @ self.P @ A.T + self.Q

    def correct(self, measured_voltage):
        """
        Correction Step: Update state based on measured Voltage.
        """
        soc_pred = self.x[0, 0]
        v1_pred = self.x[1, 0]
        
        ocv_pred = self.get_ocv(soc_pred)
        
        # Measurement Model h(x) = OCV(SOC) - V1
        voltage_pred = ocv_pred - v1_pred 
        
        # The Innovation (y)
        y = measured_voltage - voltage_pred
        
        # Jacobian H (Measurement Matrix)
        d_ocv = self.get_ocv_derivative(soc_pred)
        H = np.array([[d_ocv, -1.0]])
        
        # Innovation Covariance S
        S = H @ self.P @ H.T + self.R
        
        # Kalman Gain K
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update State x
        self.x = self.x + K * y
        
        # Update Covariance P
        I_mat = np.eye(2)
        self.P = (I_mat - K @ H) @ self.P
        
        return self.x[0, 0] * 100.0 # Return SOC %

def preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Identify column mapping based on available columns
    if 'Current_A' in df.columns and 'Voltage_V' in df.columns:
        # Parsed format
        pass
    elif 'riedon.riedon_current_mA' in df.columns:
        # Raw format
        df = df.rename(columns={
            'time': 'Timestamp',
            'riedon.riedon_current_mA': 'Current_mA',
            'bms.pack_voltage_V': 'Voltage_V'
        })
        df['Current_A'] = df['Current_mA'] / 1000.0
    
    # Ensure Timestamp
    if 'Timestamp' not in df.columns and 'time' in df.columns:
         df['Timestamp'] = pd.to_datetime(df['time'])
    else:
         df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # Calculate dt
    df['dt'] = df['Timestamp'].diff().dt.total_seconds()
    
    # Drop first row (NaN dt)
    df = df.dropna(subset=['dt'])
    
    # Filter duplicates (dt == 0)
    df = df[df['dt'] > 0]
    
    # Interpolate Missing Data
    cols_to_interp = ['Current_A', 'Voltage_V']
    
    # Check if columns exist
    cols_present = [c for c in cols_to_interp if c in df.columns]
    
    if cols_present:
        df[cols_present] = df[cols_present].interpolate(method='linear', limit_direction='both')
        df = df.dropna(subset=cols_present)
    
    return df

def run_evaluation(csv_path, output_path):
    df = preprocess_data(csv_path)
    
    # Initialize EKF
    ekf = BatteryEKF(PACK_CAPACITY_AH)
    
    # Initialize SOC based on first voltage reading (OCV inversion)
    v_start = df['Voltage_V'].iloc[0]
    
    # Simple search for initial SOC
    possible_socs = np.linspace(0, 100, 1000)
    possible_vs = ekf.ocv_func(possible_socs/100.0) * SERIES_CELLS
    idx = (np.abs(possible_vs - v_start)).argmin()
    initial_soc_percent = possible_socs[idx]
    
    # Set EKF initial state
    ekf.x[0, 0] = initial_soc_percent / 100.0
    ekf.x[1, 0] = 0.0 # Assume relaxed battery initially
    
    print(f"Initial Voltage: {v_start:.2f}V")
    print(f"Initial Estimated SOC: {initial_soc_percent:.2f}%")
    
    # Coulomb Counter variables
    cumulative_ah_discharged = 0.0
    
    results = []
    
    print("Running EKF Simulation...")
    for i, row in df.iterrows():
        dt = row['dt']
        current = row['Current_A']
        voltage = row['Voltage_V']
        
        # 1. Predict
        ekf.dt = dt
        
        # Correct Voltage = Measured + I*R0
        corrected_voltage_measure = voltage + (current * R0_PACK_OHMS)
        
        ekf.predict(current)
        soc_ekf = ekf.correct(corrected_voltage_measure)
        
        # 2. Coulomb Counting (Benchmark)
        ah_step = current * (dt / SECONDS_PER_HOUR)
        cumulative_ah_discharged += ah_step
        
        soc_cc = initial_soc_percent - (cumulative_ah_discharged / PACK_CAPACITY_AH) * 100.0
        
        results.append({
            'Timestamp': row['Timestamp'],
            'Current_A': current,
            'Voltage_V': voltage,
            'SOC_EKF_%': soc_ekf,
            'SOC_CC_%': soc_cc
        })
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Completed. Results saved to {output_path}")
    
    # Quick Stats
    final_ekf = results_df['SOC_EKF_%'].iloc[-1]
    final_cc = results_df['SOC_CC_%'].iloc[-1]
    print(f"Final SOC (EKF): {final_ekf:.2f}%")
    print(f"Final SOC (CC):  {final_cc:.2f}%")

if __name__ == "__main__":
    run_evaluation('parsed_save_data.csv', 'parsed_save_data_soc_results.csv')