import os
import json
import argparse
from datetime import datetime
import h5py
import numpy as np
import pandas as pd

def process_file(file_path, output_path):
    with h5py.File(file_path, 'r') as f:
        # Extract metadata from the attributes
        metadata = dict(f.attrs)
        device_id = metadata.get('deviceid', 'unknown')
        
        # Read time-series datasets
        timestamps = f['timestamps'][:]
        heart_rate = f['HR'][:]
        steps = f['Steps'][:]
        x = f['X'][:]
        y = f['Y'][:]
        z = f['Z'][:]

        # Convert timestamps to datetime
        times = pd.to_datetime(timestamps, unit='s')

        # Build DataFrame
        df = pd.DataFrame({
            'timestamp': times,
            'heart_rate': heart_rate,
            'step_count': steps,
            'x': x,
            'y': y,
            'z': z
        })

        # Round timestamps to nearest minute (floor)
        df['timestamp'] = df['timestamp'].dt.floor('T')

        # Downsample: mean for HR and acceleration, sum for steps
        grouped = df.groupby('timestamp').agg({
            'heart_rate': 'mean',
            'x': 'mean',
            'y': 'mean',
            'z': 'mean',
            'step_count': 'sum'
        }).reset_index()

        # Format output
        data = []
        for _, row in grouped.iterrows():
            data.append({
                "timestamp": row['timestamp'].strftime('%Y-%m-%dT%H:%M:00Z'),
                "heart_rate": int(round(row['heart_rate'])),
                "acceleration": {
                    "x": round(row['x'], 2),
                    "y": round(row['y'], 2),
                    "z": round(row['z'], 2)
                },
                "step_count": int(row['step_count'])
            })

        result = {
            "deviceid": str(device_id),
            "device": "Pixel Watch",
            "metadata": {
                "version": "1.0",
                "units": {
                    "heartRate": "bpm",
                    "stepCount": "count",
                    "acceleration": "m/s2"
                }
            },
            "data": data
        }

        # Write to JSON
        filename = os.path.basename(file_path).replace('.hdf5', '.json')
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

    if file_path.endswith('.hdf5'):
        process_file(file_path, output_dir)
        print(f'Processed: {file_path}')

if __name__ == '__main__':
    main()

