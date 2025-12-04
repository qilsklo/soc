import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg"
import matplotlib.pyplot as plt
import pandas as pd

# Load your CSV
df = pd.read_csv("ekf_soc_results_fixed.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

plt.plot(df["Timestamp"], df["SOC_EKF_%"], label="SOC EKF")
plt.plot(df["Timestamp"], df["SOC_CC_%"], label="SOC CC")
plt.xlabel("Time")
plt.ylabel("SOC (%)")
plt.title("SOC Over Time")
plt.legend()
plt.tight_layout()

# Save the figure to a file
plt.savefig("ekf_vs_cc_soc.png", dpi=300)

# Show the figure interactively
plt.show()
