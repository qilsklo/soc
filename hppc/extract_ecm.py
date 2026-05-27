import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json

file_path = '/home/fred/Documents/calsol/soc/hppc/Cell1_incremental_capacity_Channel_8_Wb_1.CSV'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Create a group id for contiguous step indices
df['Step Group'] = (df['Step Index'] != df['Step Index'].shift()).cumsum()

# Function to fit relaxation curve
def exp_func(t, a, b, c):
    # a: OCV, b: -V_c0, c: tau
    return a + b * np.exp(-t / c)

results = []

# Nominal capacity is max discharge capacity
max_cap = df['Discharge Capacity (Ah)'].max()
pulse_current = 1.7 # A (magnitude)

# We pair each Step 3 (discharge) with the subsequent Step 4 (rest)
groups = df['Step Group'].unique()
for i in range(len(groups) - 1):
    g1 = groups[i]
    g2 = groups[i+1]
    
    df1 = df[df['Step Group'] == g1]
    df2 = df[df['Step Group'] == g2]
    
    if df1['Step Index'].iloc[0] == 3 and df2['Step Index'].iloc[0] == 4:
        # Extract R0
        v_end_pulse = df1['Voltage (V)'].iloc[-1]
        v_start_rest = df2['Voltage (V)'].iloc[0]
        i_end_pulse = df1['Current (A)'].iloc[-1]
        i_start_rest = df2['Current (A)'].iloc[0]
        
        delta_i = i_start_rest - i_end_pulse # e.g. 0 - (-1.7) = 1.7
        if delta_i == 0:
            continue
            
        r0 = (v_start_rest - v_end_pulse) / delta_i
        
        # Fit R1, C1
        t = df2['Test Time (s)'].values
        t = t - t[0] # normalize to 0
        v = df2['Voltage (V)'].values
        
        # initial guesses: a = final voltage, b = v[0] - a, c = 100
        p0 = [v[-1], v[0] - v[-1], 100]
        try:
            popt, _ = curve_fit(exp_func, t, v, p0=p0, maxfev=10000)
            ocv = popt[0]
            v_c0 = -popt[1]
            tau = popt[2]
            
            # Pulse duration
            t_p = df1['Test Time (s)'].iloc[-1] - df1['Test Time (s)'].iloc[0]
            
            # Calculate R1, C1
            # v_c0 = I * R1 * (1 - exp(-t_p / tau))
            r1 = v_c0 / (pulse_current * (1 - np.exp(-t_p / tau)))
            c1 = tau / r1 if r1 != 0 else 0
            
            # SOC
            dis_cap = df1['Discharge Capacity (Ah)'].iloc[-1]
            soc = 1 - (dis_cap / max_cap)
            
            results.append({
                'SOC': soc,
                'OCV': ocv,
                'R0': r0,
                'R1': r1,
                'C1': c1,
                'tau': tau
            })
        except Exception as e:
            print(f"Failed to fit group {g2}: {e}")

res_df = pd.DataFrame(results)
print(res_df)

# Save to csv for artifact
res_df.to_csv('hppc_params.csv', index=False)

# Plot
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(res_df['SOC'], res_df['R0'] * 1000, marker='o')
plt.title('R0 vs SOC')
plt.xlabel('SOC')
plt.ylabel('R0 (mOhm)')
plt.gca().invert_xaxis()

plt.subplot(2, 2, 2)
plt.plot(res_df['SOC'], res_df['R1'] * 1000, marker='o', color='orange')
plt.title('R1 vs SOC')
plt.xlabel('SOC')
plt.ylabel('R1 (mOhm)')
plt.gca().invert_xaxis()

plt.subplot(2, 2, 3)
plt.plot(res_df['SOC'], res_df['C1'], marker='o', color='green')
plt.title('C1 vs SOC')
plt.xlabel('SOC')
plt.ylabel('C1 (F)')
plt.gca().invert_xaxis()

plt.subplot(2, 2, 4)
plt.plot(res_df['SOC'], res_df['tau'], marker='o', color='red')
plt.title('Time Constant (tau) vs SOC')
plt.xlabel('SOC')
plt.ylabel('tau (s)')
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig('hppc_params_vs_soc.png')
