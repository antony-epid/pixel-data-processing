import os
import json
import argparse
from datetime import datetime
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def transform_path(path: str) -> str:
    parts = path.strip().split('/')
    if len(parts) < 5:
        #raise ValueError("Path must contain at least 5 components")
        filebase = os.path.splitext(parts[-1])[0]
        proctime = datetime.now().strftime('%Y%m%d_%H%M%S')
        return 'InvalidPINfolder', f"{proctime}_{filebase}.json"

    short_dir, uuid, dataset, compound_part, filename = parts[-5:]
    
    compound_parts = compound_part.split(':')
    #compound_parts = compound_part.split('-')
    compound_joined = '-'.join(compound_parts)
    id_part = compound_parts[-1]
    
    file_base = os.path.splitext(filename)[0]
    
    #return id_part, f"{id_part}_{short_dir}-{uuid}-{dataset}-{compound_joined}_{file_base}.json"
    #return id_part, f"{id_part}_{short_dir}-{uuid}-{compound_joined}_{file_base}.json"
    return id_part, f"{short_dir}-{uuid}-{compound_joined}_{file_base}.json"


def to_datetime_index(timestamps):
    return pd.to_datetime(timestamps, unit='s')

def process_file(file_path, output_path):
    with h5py.File(file_path, 'r') as f:
        # Extract ID from group
        rootgrp = list(f.keys())[0]
        base_path = f"{rootgrp}"

        pin = f.attrs.get('pin', 'InvalidPIN')
        device_id = str(pin) if isinstance(pin, (str, int, float, bool, bytes)) else 'InvalidPIN'
        
        acc_path = f"{base_path}/Accelerometer/Ch_0/Data"
        t_acc = f[acc_path + "/t"][:]
        # Create a DatetimeIndex from accelerometer timestamps, resampled to minute-level
        acc_time_index = to_datetime_index(t_acc)
        acc_minute_index = pd.Series(index=acc_time_index).resample('1T').mean().index  

        hr_path = f"{base_path}/Heart_rate/Ch_0/Data"
        if 'Heart_rate' not in f[base_path]:
            # Create empty DataFrame with null heart_rate for the same index as acc timestamps
            #hr_minute = pd.DataFrame({'heart_rate': pd.Series(dtype='Int64')}, index=to_datetime_index(t_acc))
            #hr_minute = hr_minute.resample('1T').asfreq()
            hr_minute = pd.DataFrame({'heart_rate': pd.Series(dtype='Int64')}, index=acc_minute_index)
        else:
            t_hr = f[hr_path + "/t"][:]
            hr = f[hr_path + "/heart_rate"][:]
            # Heart rate
            df_hr = pd.DataFrame({'heart_rate': hr}, index=to_datetime_index(t_hr))
            hr_minute = df_hr.resample('1T').mean()
            hr_minute = hr_minute.reindex(acc_minute_index)  # Reindex to match acc_minute_index

        # Set the 'step_count' column as null in steps_minute dataframe if Step_count group is not found in the HDF5 f object
        if 'Step_count' not in f[base_path]:
            #steps_minute = pd.DataFrame({'step_count': pd.Series(dtype='Int64')}, index=to_datetime_index(t_acc))
            steps_minute = pd.DataFrame({'step_count': pd.Series(dtype='Int64')}, index=acc_minute_index)            
        else:
            step_path = f"{base_path}/Step_count/Ch_0/Data"
            t_step = f[step_path + "/t"][:]
            #steps = f[step_path + "/steps"][:]
            
            # Step detection, extracted from cummulative stepCount data which increases on detecting a new step
            df_steps = pd.DataFrame({'step_count': 1}, index=to_datetime_index(t_step))  
            steps_minute = df_steps.resample('1T').sum()
            # Reindex to match hr_minute's index, fill missing with NaN (which will become null in JSON)
            #steps_minute = steps_minute.reindex(hr_minute.index)
            steps_minute = steps_minute.reindex(acc_minute_index)  # Reindex to match acc_minute_index
            # Replace NaN in column 'step' with 0, I think this is no longer necessary ?
            steps_minute['step_count'] = steps_minute['step_count'].fillna(0)

        # Combine
        #combined = pd.concat([acc_minute, hr_minute, steps_minute], axis=1)
        combined = pd.concat([hr_minute, steps_minute], axis=1)        
        ## Replace NaN in column 'step' with 0, I think this is no longer necessary ?
        #combined['step_count'] = combined['step_count'].fillna(0)

        # Format timestamp with :00 seconds
        combined.reset_index(inplace=True)
        combined.insert(0, 'timestamp', combined['index'].dt.strftime('%Y-%m-%d %H:%M:00Z'))
        combined.drop(columns=['index'], inplace=True)

        combined['step_count'] = combined['step_count'].round().astype('Int64')  # using Int64 preserves Null        
        combined['heart_rate'] = combined['heart_rate'].round().astype('Int64')

        #folder_pin, filename = transform_path(file_path)
        #filename = device_id + '_' + filename 

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
        with open(output_path, 'w') as f_out:
            json.dump(result, f_out, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-filename', required=True, help='Input file: an .hdf5 file')
    parser.add_argument('--output-filename', required=True, help='Output file to save .json files')

    args = parser.parse_args()

    file_path = args.input_filename
    output_path = args.output_filename

    #will not work in GCS
    #os.makedirs(output_dir, exist_ok=True)

    if file_path.lower().endswith(('.h5', '.hdf5')):
        process_file(file_path, output_path)
        print(f'Processed: {file_path}')

if __name__ == '__main__':
    main()
