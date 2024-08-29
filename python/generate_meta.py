import pandas as pd
import seaborn as sns
import numpy as np
import os
import glob

def aggregate_key_values(base_directory, target_key):
    aggregated_data = {}

    # Find all folders that start with "obj_"
    folders = [f.path for f in os.scandir(base_directory) if f.is_dir() and f.name.startswith('obj_')]

    for folder in folders:
        folder_name = os.path.basename(folder)
        aggregated_data[folder_name] = []

        meta_files = glob.glob(os.path.join(folder, '*_meta.txt'))

        for meta_file in meta_files:
            with open(meta_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Aggregate values for the target key
                        if key == target_key:
                            aggregated_data[folder_name].append(value)

    return aggregated_data


def save_data_to_file(data, filename):
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    df.to_csv(filename, index=False)


base_directory = './task2/dfs_meta/'  # Replace with the path to the directory containing obj_ folders with the meta data
categories = [
    'distance to target',
    'number of views',
    'traversed distance',
    'compute time',
    'structural similarity (SSIM)',
    'peak signal to noise ratio (PSNR)'
]

for category in categories:
    data = aggregate_key_values(base_directory, category)
    save_data_to_file(data, f'./python/pgf/{category}.csv')
