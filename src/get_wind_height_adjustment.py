import pandas as pd
import csv

# Configuration
INPUT_FILE = 'data/meteoblue/meteoblue_forecasted.csv'
OUTPUT_FILE = "data/meteoblue/meteoblue_forecasted_100m.csv"
SHEAR_EXPONENT = 0.143
OLD_HEIGHT = 80.0
NEW_HEIGHT = 100.0

def adjust_weather_file():
    print(f"Processing {INPUT_FILE}...")
    
    # 1. Read the Metadata Lines (First 9 lines, indices 0-8)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        metadata_lines = [f.readline() for _ in range(9)]
        
    # 2. Update Metadata 'level' row (Row 6)
    # The format is: level, 2 m..., sfc, sfc, sfc, MSL, 80 m, 80 m
    # We want to change the '80 m' corresponding to Wind Speed to '100 m'
    level_row = metadata_lines[6].strip().split(',')
    
    # Identify the index for Wind Speed. 
    # Based on file inspection, Wind Speed is usually the 7th column (index 6)
    # but let's be safe and check the 'variable' row (Row 4)
    var_row = metadata_lines[4].strip().split(',')
    
    try:
        ws_index = var_row.index('Wind Speed')
        # Update the level for this index
        if level_row[ws_index].strip() == '80 m':
            level_row[ws_index] = '100 m'
            print(f"Updated metadata level at index {ws_index} to 100 m")
            
        # Reconstruct the line
        metadata_lines[6] = ",".join(level_row) + "\n"
        
    except ValueError:
        print("Warning: 'Wind Speed' variable not found in metadata. Skipping metadata update.")

    # 3. Read the Data (Header is on line 9, so header=9)
    df = pd.read_csv(INPUT_FILE, header=9)
    
    # 4. Identify the Wind Speed Column
    # Look for the column name containing "Wind Speed [80 m]"
    ws_col = [c for c in df.columns if "Wind Speed [80 m]" in c]
    
    if not ws_col:
        print("Error: Could not find a column with 'Wind Speed [80 m]'.")
        return
    
    ws_col_name = ws_col[0]
    print(f"Found column: {ws_col_name}")
    
    # 5. Apply Height Correction to Data
    # v2 = v1 * (h2/h1)^alpha
    correction_factor = (NEW_HEIGHT / OLD_HEIGHT) ** SHEAR_EXPONENT
    df[ws_col_name] = df[ws_col_name] * correction_factor
    
    # 6. Rename the Column
    new_col_name = ws_col_name.replace("80 m", "100 m")
    df.rename(columns={ws_col_name: new_col_name}, inplace=True)
    
    # 7. Write the Output File
    # First write the modified metadata
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        f.writelines(metadata_lines)
        
    # Then append the DataFrame (csv mode='a')
    df.to_csv(OUTPUT_FILE, mode='a', index=False, float_format='%.2f')
    
    print(f"Successfully created {OUTPUT_FILE} with corrected wind speed data.")

if __name__ == "__main__":
    adjust_weather_file()