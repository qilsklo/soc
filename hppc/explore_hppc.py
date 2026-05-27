import pandas as pd
import numpy as np

file_path = '/home/fred/Documents/calsol/soc/hppc/Cell1_incremental_capacity_Channel_8_Wb_1.CSV'
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()
df['Step Group'] = (df['Step Index'] != df['Step Index'].shift()).cumsum()

summary = df.groupby(['Step Group', 'Step Index']).agg(
    start_time=('Test Time (s)', 'min'),
    end_time=('Test Time (s)', 'max'),
    duration=('Test Time (s)', lambda x: x.max() - x.min()),
    avg_current=('Current (A)', 'mean'),
    min_current=('Current (A)', 'min'),
    max_current=('Current (A)', 'max'),
    start_voltage=('Voltage (V)', 'first'),
    end_voltage=('Voltage (V)', 'last'),
    count=('Data Point', 'count')
)

pd.set_option('display.max_rows', None)
print(summary)
