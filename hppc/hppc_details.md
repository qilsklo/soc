# HPPC Parameter Extraction for 1st-Order Thevenin ECM

Based on the provided HPPC data (`Cell1_incremental_capacity_Channel_8_Wb_1.CSV`), I have extracted the parameters for a 1st-order Thevenin Equivalent Circuit Model (ECM). 

## Parameter Values vs SOC

The extraction identifies consecutive pairs of "Discharge" and "Rest" steps to fit the parameters at various Depths of Discharge (DOD). The table below lists the parameters at the corresponding State of Charge (SOC) at the end of each discharge pulse.

| SOC (%) | OCV (V) | R0 (mΩ) | R1 (mΩ) | C1 (F) | Tau (s) |
|---|---|---|---|---|---|
| 94.90 | 4.097 | 32.3 | 14.8 | 2702 | 40.1 |
| 89.79 | 4.074 | 32.1 | 21.2 | 1974 | 41.9 |
| 84.69 | 4.046 | 31.8 | 25.5 | 2939 | 74.8 |
| 79.59 | 3.991 | 31.5 | 27.1 | 6065 | 164.1 |
| 74.49 | 3.924 | 31.5 | 20.7 | 8038 | 166.2 |
| 69.39 | 3.874 | 31.2 | 17.5 | 4543 | 79.5 |
| 64.28 | 3.834 | 31.2 | 18.4 | 3463 | 63.8 |
| 59.18 | 3.791 | 31.2 | 20.7 | 2866 | 59.2 |
| 54.07 | 3.745 | 31.0 | 22.6 | 3027 | 68.3 |
| 48.97 | 3.696 | 31.0 | 24.2 | 3635 | 88.0 |
| 43.87 | 3.648 | 31.0 | 25.2 | 4050 | 102.0 |
| 38.77 | 3.598 | 30.7 | 23.8 | 4139 | 98.7 |
| 33.66 | 3.531 | 30.5 | 15.7 | 4533 | 71.1 |
| 28.56 | 3.487 | 30.9 | 22.8 | 2319 | 52.9 |
| 23.46 | 3.402 | 31.2 | 27.2 | 3467 | 94.2 |
| 18.36 | 3.307 | 31.9 | 29.7 | 3189 | 94.8 |
| 13.25 | 3.229 | 33.4 | 34.0 | 2856 | 97.1 |
| 8.15 | 3.147 | 36.9 | 43.8 | 2288 | 100.2 |
| 3.05 | 3.016 | 48.7 | 56.7 | 1709 | 96.9 |

*Note: The combined internal resistance ($R_0 + R_1$) generally lies in the 45–65 mΩ range for most of the SOC curve, going up to ~105 mΩ at 3% SOC, which accurately maps to the expected DCIR characteristics of the LG INR18650MJ1 cell.*

![Parameter Plots vs SOC](/home/fred/.gemini/antigravity/brain/a61c46f0-4a7f-4be5-9e67-7a78285f0cd0/hppc_params_vs_soc.png)

## Assumptions & Equations

### 1. Identify Pulses
A sequence grouping method was used to identify all pairs of discharging steps (`Step Index` 3) followed immediately by resting steps (`Step Index` 4). Discharging current magnitude was confirmed to be ~1.7 A (which correlates to roughly C/2 for the 3.45Ah nominal capacity).

### 2. Instantaneous Voltage Drop ($R_0$)
The instantaneous series resistance ($R_0$) represents the purely ohmic behavior of the cell. It is computed at the transition from discharge to rest:
$$ R_0 = \frac{V_{rest, start} - V_{pulse, end}}{\Delta I} $$
where $\Delta I$ is the magnitude of the current step (1.7 A), $V_{pulse, end}$ is the voltage just prior to current shut-off, and $V_{rest, start}$ is the first recorded voltage point during relaxation.

### 3. Transient Relaxation Response ($R_1$, $C_1$)
During the rest phase, the voltage relaxes to the Open Circuit Voltage ($OCV$). The 1st-order Thevenin ECM models this relaxation as:
$$ V(t) = OCV - V_{c0} \cdot e^{-t / \tau} $$
where $\tau = R_1 C_1$ is the time constant, and $V_{c0}$ is the initial polarization voltage across the RC pair at the start of the rest.
This equation is fitted against the rest time-series data using nonlinear least squares to determine $OCV$, $V_{c0}$, and $\tau$.

Once $V_{c0}$ and $\tau$ are known, we can trace back how much voltage accumulated on the capacitor during the preceding discharge pulse (of duration $t_{pulse} \approx 369.6\text{s}$). Assuming the capacitor was fully relaxed before the pulse, the voltage accumulation follows:
$$ V_{c0} = I \cdot R_1 \left(1 - e^{-t_{pulse} / \tau}\right) $$
We can re-arrange this to solve for $R_1$:
$$ R_1 = \frac{V_{c0}}{I \left(1 - e^{-t_{pulse} / \tau}\right)} $$
And $C_1$ is subsequently found from the time constant:
$$ C_1 = \frac{\tau}{R_1} $$

**Key Assumptions**:
- The model treats temperature as constant across the test, ignoring the non-linear coupling of self-heating on impedance.
- It assumes $V_c$ has decayed fully to 0 V by the end of the previous 600-second rest. Some minor residual polarization may persist.
- Measurements reflect the cell at the particular ending SOC level, though the 1.7 A pulse over ~370 seconds shifts the cell's SOC by approximately 5% throughout the pulse.

## Python Extraction Code

```python
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def exp_func(t, a, b, c):
    return a + b * np.exp(-t / c)

def extract_ecm_parameters(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Identify contiguous step groups
    df['Step Group'] = (df['Step Index'] != df['Step Index'].shift()).cumsum()
    groups = df['Step Group'].unique()

    max_cap = df['Discharge Capacity (Ah)'].max()
    pulse_current = 1.7 # Magnitue of discharge current

    results = []
    
    for i in range(len(groups) - 1):
        g1 = groups[i]
        g2 = groups[i+1]
        
        df1 = df[df['Step Group'] == g1]
        df2 = df[df['Step Group'] == g2]
        
        # Look for Pulse (Step 3) followed by Rest (Step 4)
        if df1['Step Index'].iloc[0] == 3 and df2['Step Index'].iloc[0] == 4:
            v_end_pulse = df1['Voltage (V)'].iloc[-1]
            v_start_rest = df2['Voltage (V)'].iloc[0]
            delta_i = df2['Current (A)'].iloc[0] - df1['Current (A)'].iloc[-1]
            
            if delta_i == 0:
                continue
                
            r0 = (v_start_rest - v_end_pulse) / delta_i
            
            # Rest relaxation fitting
            t = df2['Test Time (s)'].values
            t = t - t[0]
            v = df2['Voltage (V)'].values
            
            p0 = [v[-1], v[0] - v[-1], 100]
            try:
                popt, _ = curve_fit(exp_func, t, v, p0=p0, maxfev=10000)
                ocv = popt[0]
                v_c0 = -popt[1]
                tau = popt[2]
                
                t_p = df1['Test Time (s)'].iloc[-1] - df1['Test Time (s)'].iloc[0]
                
                # R1 and C1 logic
                r1 = v_c0 / (pulse_current * (1 - np.exp(-t_p / tau)))
                c1 = tau / r1 if r1 != 0 else 0
                soc = 1 - (df1['Discharge Capacity (Ah)'].iloc[-1] / max_cap)
                
                results.append({'SOC': soc, 'OCV': ocv, 'R0': r0, 'R1': r1, 'C1': c1, 'tau': tau})
            except Exception as e:
                print(f"Skipped fitting group {g2}: {e}")

    return pd.DataFrame(results)

# Example usage
# res_df = extract_ecm_parameters('Cell1_incremental_capacity_Channel_8_Wb_1.CSV')
```
