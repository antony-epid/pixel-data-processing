import os
import json
import argparse
from datetime import datetime
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def to_datetime_index(timestamps):
    return pd.to_datetime(timestamps, unit='s')

def process_file(file_path, output_path):
    with h5py.File(file_path, 'r') as f:
        # Extract ID from group
        device_id = list(f.keys())[0]
        base_path = f"{device_id}"

        acc_path = f"{base_path}/Accelerometer/Ch_0/Data"
        t_acc = f[acc_path + "/t"][:]
        x = f[acc_path + "/x"][:]
        y = f[acc_path + "/y"][:]
        z = f[acc_path + "/z"][:]

        hr_path = f"{base_path}/Heart_rate/Ch_0/Data"
        t_hr = f[hr_path + "/t"][:]
        hr = f[hr_path + "/heart_rate"][:]

        step_path = f"{base_path}/Step_count/Ch_0/Data"
        t_step = f[step_path + "/t"][:]
        steps = f[step_path + "/steps"][:]

        # Accelerometer
        df_acc = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z
        }, index=to_datetime_index(t_acc))
        acc_minute = df_acc.resample('1T').mean()

        # Heart rate
        df_hr = pd.DataFrame({'heart_rate': hr}, index=to_datetime_index(t_hr))
        hr_minute = df_hr.resample('1T').mean()

        # Step count (cumulative to per-minute steps)
        df_steps = pd.DataFrame({'steps': steps}, index=to_datetime_index(t_step))
        steps_diff = df_steps.diff().clip(lower=0)  # avoid negative steps
        steps_minute = steps_diff.resample('1T').sum()

        # Combine
        combined = pd.concat([acc_minute, hr_minute, steps_minute], axis=1)

        # Format timestamp with :00 seconds
        combined.reset_index(inplace=True)
        #combined['timestamp'] = combined['index'].dt.strftime('%Y-%m-%d %H:%M:00Z')
        #combined['timestamp'] = combined['index']
        #create timestamp column as the first column in the dataframe
        combined.insert(0, 'timestamp', combined['index'].dt.strftime('%Y-%m-%d %H:%M:00Z'))
        combined.drop(columns=['index'], inplace=True)

        #combined = combined.where(pd.notna(combined),None)
        combined['steps'] = combined['steps'].round().astype('Int64')  # using Int64 preserves Null
        #combined['steps'] = combined['steps'].where(pd.notna(combined['steps']), None) 
        #combined['steps'] = combined['steps'].fillna(value=pd.NA)
        #combined['steps'] = combined['steps'].replace(pd.NA, None)
        combined['heart_rate'] = combined['heart_rate'].round().astype('Int64')

        #Format output
        data = []
        # Specify group columns
        group_keys = ['x', 'y', 'z']        
        for _, row in combined.iterrows():
            # data.append({
            #     #"timestamp": row['timestamp'].strftime('%Y-%m-%dT%H:%M:00Z'),
            #     "timestamp": row['timestamp'],                
            #     #"heart_rate": int(round(row['heart_rate'])),
            #     "heart_rate": row['heart_rate'],
            #     "acceleration": {
            #         "x": round(row['x'], 2),
            #         "y": round(row['y'], 2),
            #         "z": round(row['z'], 2)
            #     },
            #     "step_count": row['steps']
            # })

            # JSON-friendly dict : convert to dict and replace pd.NA with None
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            
            # Extract group values
            group_dict = {k: round(row_dict.pop(k),2) for k in group_keys}
            
            # Add group nesting
            row_dict['acceleration'] = group_dict
            
            data.append(row_dict)

            

        result = {
            "deviceid": str(device_id),
            "device": "Pixel Watch",
            "metadata": {
                "version": "1.0",
                "units": {
                    "heartRate": "bpm",
                    "stepCount": "count",
                    "acceleration": "mg"
                }
            },
            "data": data
            #'data': combined.to_dict(orient='records')  # Convert to JSON-safe structure
        }

        # Write to JSON
        #filename = os.path.basename(file_path).replace('.hdf5', '.json')
        filename = str(Path(file_path).with_suffix('.json').name)
        out_file = os.path.join(output_path, filename)
        with open(out_file, 'w') as f_out:
            json.dump(result, f_out, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filename', required=True, help='Input file: an .hdf5 file')
    parser.add_argument('--output-dir', required=True, help='Output directory to save .json files')
    args = parser.parse_args()

    file_path = args.input_filename
    output_dir = args.output_dir

    #will not work in GCS
    #os.makedirs(output_dir, exist_ok=True)

    if file_path.lower().endswith(('.h5', '.hdf5')):
        process_file(file_path, output_dir)
        print(f'Processed: {file_path}')

if __name__ == '__main__':
    main()

