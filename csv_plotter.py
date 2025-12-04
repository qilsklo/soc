import pandas as pd
import matplotlib.pyplot as plt

csv_file = "FSGP_day1.csv"

df = pd.read_csv(csv_file, na_values=["NA"])

df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['riedon.riedon_current_mA'] = pd.to_numeric(df['riedon.riedon_current_mA'], errors='coerce')
df['bms.pack_voltage_V'] = pd.to_numeric(df['bms.pack_voltage_V'], errors='coerce')

df = df.dropna(subset=['time'])

df = df.dropna(subset=['riedon.riedon_current_mA', 'bms.pack_voltage_V'], how='all')

df = df[(df['riedon.riedon_current_mA'].isna()) |
        ((df['riedon.riedon_current_mA'] > -2000) &
         (df['riedon.riedon_current_mA'] < 2000))]

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("Time")
ax1.set_ylabel("Current (mA)")
ax1.plot(df['time'], df['riedon.riedon_current_mA'], label="Current", linewidth=1)
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel("Voltage (V)")
ax2.plot(df['time'], df['bms.pack_voltage_V'], label="Voltage", color='orange', linewidth=1)
ax2.tick_params(axis='y')

plt.title("Current and Voltage vs Time")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("current_voltage_plot.png")
print("Saved plot to current_voltage_plot.png")

