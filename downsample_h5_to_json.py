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
        #x = f[acc_path + "/x"][:]
        #y = f[acc_path + "/y"][:]
        #z = f[acc_path + "/z"][:]

        hr_path = f"{base_path}/Heart_rate/Ch_0/Data"
        t_hr = f[hr_path + "/t"][:]
        hr = f[hr_path + "/heart_rate"][:]

        step_path = f"{base_path}/Step_count/Ch_0/Data"
        t_step = f[step_path + "/t"][:]
        #steps = f[step_path + "/steps"][:]

        # Accelerometer
        #df_acc = pd.DataFrame({
        #    'x': x,
        #    'y': y,
        #    'z': z
        #}, index=to_datetime_index(t_acc))
        #acc_minute = df_acc.resample('1T').mean()

        #dura_index = pd.date_range(start=acc_minute[0].floor('T'), end=acc_minute[-1].floor('T'), freq='min')

        # Heart rate
        df_hr = pd.DataFrame({'heart_rate': hr}, index=to_datetime_index(t_hr))
        hr_minute = df_hr.resample('1T').mean()

        # Step detection, extracted from cummulative stepCount data which increases on detecting a new step
        df_steps = pd.DataFrame({'step_count':1}, index=to_datetime_index(t_step))  
        steps_minute = df_steps.resample('1T').sum()

        # Combine
        #combined = pd.concat([acc_minute, hr_minute, steps_minute], axis=1)
        combined = pd.concat([hr_minute, steps_minute], axis=1)        
        # Replace NaN in column 'step' with 0
        combined['step_count'] = combined['step_count'].fillna(0)

        # Format timestamp with :00 seconds
        combined.reset_index(inplace=True)
        combined.insert(0, 'timestamp', combined['index'].dt.strftime('%Y-%m-%d %H:%M:00Z'))
        combined.drop(columns=['index'], inplace=True)

        combined['step_count'] = combined['step_count'].round().astype('Int64')  # using Int64 preserves Null        
        combined['heart_rate'] = combined['heart_rate'].round().astype('Int64')

        #The following is not needed when using 'data':combined.to_dict. It's needed if we build it from scratch (list of data from each timestamp) 
        #combined = combined.applymap(lambda x: None if pd.isna(x) else x)

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
            'data': combined.to_dict(orient='records')  # Convert to JSON-safe structure
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

