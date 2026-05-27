import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load the result CSV
file_path = 'results/FSGP_day1result.csv'
df = pd.read_csv(file_path)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Set style for a beautiful premium design
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, dpi=300)

# Colors from a harmonious modern palette
color_ekf = '#1f77b4'       # Vibrant Blue
color_cc = '#ff7f0e'        # Vibrant Coral/Orange
color_ecm = '#2ca02c'       # Soft Emerald Green
color_current = '#7f7f7f'   # Muted Gray
color_voltage = '#d62728'   # Crimson Red

# Panel 1: Current Profile
ax1.plot(df['Timestamp'], df['Current_A'], color=color_current, alpha=0.8, linewidth=1.2)
ax1.fill_between(df['Timestamp'], df['Current_A'], 0, color=color_current, alpha=0.1)
ax1.set_ylabel('Pack Current (A)', fontsize=12, fontweight='bold', color='#333333')
ax1.set_title('Pack Current Profile (FSGP Day 1)', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.tick_params(colors='#333333', labelsize=10)

# Panel 2: Voltage Profile
ax2.plot(df['Timestamp'], df['Voltage_V'], color=color_voltage, alpha=0.9, linewidth=1.2)
ax2.set_ylabel('Pack Voltage (V)', fontsize=12, fontweight='bold', color='#333333')
ax2.set_title('Pack Voltage Profile', fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.tick_params(colors='#333333', labelsize=10)

# Panel 3: SOC Estimator Comparison
ax3.plot(df['Timestamp'], df['SOC_EKF_%'], label='EKF (Dynamic 1st-Order ECM)', color=color_ekf, linewidth=2)
ax3.plot(df['Timestamp'], df['SOC_CC_%'], label='Coulomb Counting (Pure Integration)', color=color_cc, linewidth=1.5, linestyle='--')
ax3.plot(df['Timestamp'], df['SOC_ECM_OCV_%'], label='Instantaneous ECM-OCV (No Filter)', color=color_ecm, alpha=0.6, linewidth=1)

ax3.set_ylabel('State of Charge (%)', fontsize=12, fontweight='bold', color='#333333')
ax3.set_title('State of Charge (SOC) Estimator Comparison', fontsize=14, fontweight='bold', pad=15)
ax3.set_ylim(-2, 102)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='lower left', fontsize=11, frameon=True, facecolor='white', edgecolor='#e0e0e0', framealpha=0.9)
ax3.tick_params(colors='#333333', labelsize=10)

# Format the X-axis (Timestamps)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=15)
ax3.set_xlabel('Time (HH:MM:SS)', fontsize=12, fontweight='bold', labelpad=10)

# Adjust layouts and spacing
plt.tight_layout()

# Save plot
output_path = 'plots/fsgp_day1_soc_comparison.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Plot successfully saved to: {output_path}")
