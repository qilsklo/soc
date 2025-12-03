import re
import pandas as pd
from datetime import datetime, timedelta

def parse_save_file(input_file, output_file):
    print(f"Parsing {input_file}...")

    # Data storage
    data_points = []
    
    # State variables for parsing blocks
    capture_start_time = None
    first_mcu_time_s = None
    
    current_mcu_time_s = None
    current_lem_a = None
    current_pack_v = None

    # Regex Patterns
    # "Serial capture started at 2025-11-22 13:17:06"
    re_start_time = re.compile(r"Serial capture started at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    
    # "Time: 028 m 18 s 501 ms"
    re_time = re.compile(r"Time:\s*(\d+)\s*m\s*(\d+)\s*s\s*(\d+)\s*ms")
    
    # "LEM Current (A): 000.072484"
    re_lem = re.compile(r"LEM Current \(A\):\s*([\d\.-]+)")
    
    # "Pack Voltage (V): 134.8065"
    re_voltage = re.compile(r"Pack Voltage \(V\):\s*([\d\.-]+)")

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # 1. Check for Start Time (Header)
            if capture_start_time is None:
                match_start = re_start_time.search(line)
                if match_start:
                    capture_start_time = datetime.strptime(match_start.group(1), "%Y-%m-%d %H:%M:%S")
                    print(f"Found Capture Start Time: {capture_start_time}")
                    continue

            # 2. Check for MCU Time (Start of a data block)
            match_time = re_time.search(line)
            if match_time:
                # Calculate total seconds for this timestamp
                mins = int(match_time.group(1))
                secs = int(match_time.group(2))
                ms = int(match_time.group(3))
                
                new_mcu_time_s = mins * 60 + secs + ms / 1000.0
                
                # Update state
                current_mcu_time_s = new_mcu_time_s
                
                # Reset sensor values for this new time block
                current_lem_a = None
                current_pack_v = None
                continue

            # 3. Check for LEM Current
            match_lem = re_lem.search(line)
            if match_lem:
                try:
                    current_lem_a = float(match_lem.group(1))
                except ValueError:
                    continue

            # 4. Check for Pack Voltage (End of critical data in block)
            match_voltage = re_voltage.search(line)
            if match_voltage:
                try:
                    current_pack_v = float(match_voltage.group(1))
                    
                    # We have all three needed components?
                    if current_mcu_time_s is not None and current_lem_a is not None:
                        # Logic to establish absolute timestamp
                        if first_mcu_time_s is None:
                            first_mcu_time_s = current_mcu_time_s
                        
                        # Relative time from start of log
                        rel_seconds = current_mcu_time_s - first_mcu_time_s
                        
                        # Handle potential counter wraparound or reset (simple check)
                        if rel_seconds < 0:
                            # If time went backward significantly, maybe ignore or reset base?
                            # For now, assume monotonic increase.
                            rel_seconds = 0
                            
                        # Absolute timestamp
                        if capture_start_time:
                            final_timestamp = capture_start_time + timedelta(seconds=rel_seconds)
                        else:
                            # Fallback if no header found: use relative seconds as placeholder
                            # or start from now.
                            final_timestamp = datetime.now() + timedelta(seconds=rel_seconds)

                        data_points.append({
                            'Timestamp': final_timestamp,
                            'Current_A': current_lem_a,
                            'Voltage_V': current_pack_v
                        })
                        
                except ValueError:
                    continue

    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    if df.empty:
        print("Warning: No valid data points parsed.")
        return

    # Save to CSV
    print(f"Extracted {len(df)} data points.")
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")
    
    # Verify Content
    print("\n--- Data Verification (First 5 Rows) ---")
    print(df.head())
    print("\n--- Stats ---")
    print(df.describe()[['Current_A', 'Voltage_V']])

if __name__ == "__main__":
    # If capture start time isn't in file, script will default to 'now' + relative time.
    # Ideally, ensure 'Serial capture started at...' line exists in save.txt
    parse_save_file('save.txt', 'parsed_save_data.csv')