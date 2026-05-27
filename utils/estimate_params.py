import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

# --- Configuration ---
INPUT_FILE = 'FSGP_day1.csv'
OCV_CSV_PATH = 'curve-compute/processed_red_curve_data.csv'
SERIES_CELLS = 34

# --- OCV Logic (Copied from kalman3.py) ---
try:
    print(f"Loading OCV curve from: {OCV_CSV_PATH}")
    ocv_df = pd.read_csv(OCV_CSV_PATH)
    soc_points = ocv_df['State of Charge'].values * 100.0
    voltage_points = ocv_df['Voltage (V)'].values
    OCV_DATA = np.column_stack((soc_points, voltage_points))
    sort_indices = np.argsort(OCV_DATA[:, 0])
    OCV_DATA = OCV_DATA[sort_indices]
except Exception as e:
    print(f"Error loading OCV CSV: {e}")
    print("Falling back to hardcoded OCV map")
    OCV_DATA = np.array([
        [0.0,   2.50], [2.5,   2.80], [5.0,   2.98], [14.2,  3.05],
        [28.57, 3.33], [42.9,  3.50], [50.0,  3.55], [57.1,  3.60],
        [71.4,  3.75], [85.7,  3.87], [90.0,  4.04], [95.0,  4.08], [100.0, 4.20]
    ])

OCV_SOC_MAP_PERCENT = OCV_DATA[:, 0]
OCV_PACK_MAP_VOLTS = OCV_DATA[:, 1] * SERIES_CELLS

# Sort for interpolation
sort_idx = np.argsort(OCV_SOC_MAP_PERCENT)
OCV_SOC_MAP_PERCENT = OCV_SOC_MAP_PERCENT[sort_idx]
OCV_PACK_MAP_VOLTS = OCV_PACK_MAP_VOLTS[sort_idx]

ocv_to_soc_interp = interp1d(OCV_PACK_MAP_VOLTS, OCV_SOC_MAP_PERCENT, kind='linear', fill_value="extrapolate")

def exponential_decay(t, V_final, V_polarization, tau):
    return V_final - V_polarization * np.exp(-t / tau)

def analyze_relaxation_events(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    df = df.rename(columns={
        'riedon.riedon_current_mA': 'Current_mA',
        'bms.pack_voltage_V': 'Voltage_V',
        'bms.pack_highest_temp_C': 'Temp_C',
        'time': 'Timestamp'
    })
    
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    except:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
    df['Current_A'] = -1.0 * df['Current_mA'] / 1000.0
    df['Time_Sec'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
    
    events = []
    i = 100
    while i < len(df) - 100:
        current_load = df['Current_A'].iloc[i-10:i].mean()
        current_now = df['Current_A'].iloc[i:i+10].mean()
        
        # Relaxed criteria: Load > 2A, Rest < 2A
        if current_load > 2.0 and abs(current_now) < 2.0:
            rest_duration = 0
            j = i
            while j < len(df) and abs(df['Current_A'].iloc[j]) < 2.0:
                rest_duration += 1
                j += 1
            
            if rest_duration > 10:
                start_idx = i
                end_idx = i + rest_duration
                
                t_data = df['Time_Sec'].iloc[start_idx:end_idx].values
                v_data = df['Voltage_V'].iloc[start_idx:end_idx].values
                temp_c = df['Temp_C'].iloc[start_idx] # Assume temp is constant during event
                
                t_data = t_data - t_data[0]
                
                v_final_guess = v_data[-1]
                v_pol_guess = v_data[-1] - v_data[0]
                tau_guess = 10.0
                
                try:
                    popt, _ = curve_fit(exponential_decay, t_data, v_data, 
                                      p0=[v_final_guess, v_pol_guess, tau_guess],
                                      bounds=([v_data[0], 0, 0.1], [v_data[-1]+5, 50, 1000]))
                    
                    v_final_est, v_pol_est, tau_est = popt
                    
                    r1_est = v_pol_est / current_load
                    c1_est = tau_est / r1_est
                    
                    # Estimate SoC from V_final (approx OCV)
                    soc_est = ocv_to_soc_interp(v_final_est)
                    
                    events.append({
                        'Start_Time': df['Timestamp'].iloc[start_idx],
                        'Temp_C': temp_c,
                        'SoC_Percent': soc_est,
                        'R1_Ohms': r1_est,
                        'C1_Farads': c1_est
                    })
                    
                except Exception:
                    pass
                
                i = end_idx
            else:
                i += 1
        else:
            i += 1
            
    if not events:
        print("No suitable relaxation events found.")
        return

    results_df = pd.DataFrame(events)
    
    # Filter crazy values
    results_df = results_df[
        (results_df['R1_Ohms'] > 0.001) & (results_df['R1_Ohms'] < 1.0) &
        (results_df['C1_Farads'] > 10) & (results_df['C1_Farads'] < 5000)
    ]
    
    print("\n--- Estimated Parameters ---")
    print(results_df[['Start_Time', 'Temp_C', 'SoC_Percent', 'R1_Ohms', 'C1_Farads']])
    
    # --- Plotting ---
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: R1 vs Temp
    plt.figure(figsize=(10, 5))
    plt.scatter(results_df['Temp_C'], results_df['R1_Ohms'], c='red')
    plt.title('Estimated R1 vs Temperature')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('R1 (Ohms)')
    plt.grid(True)
    plt.savefig('plots/R1_vs_Temp.png')
    print("Saved plots/R1_vs_Temp.png")
    
    # Plot 2: R1 vs SoC
    plt.figure(figsize=(10, 5))
    plt.scatter(results_df['SoC_Percent'], results_df['R1_Ohms'], c='blue')
    plt.title('Estimated R1 vs SoC')
    plt.xlabel('SoC (%)')
    plt.ylabel('R1 (Ohms)')
    plt.grid(True)
    plt.savefig('plots/R1_vs_SoC.png')
    print("Saved plots/R1_vs_SoC.png")

if __name__ == "__main__":
    analyze_relaxation_events(INPUT_FILE)
