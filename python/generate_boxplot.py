import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

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

def make_boxplot(base_directory, key_name):
    data = aggregate_key_values(base_directory, key_name)
    print(data)

    # Convert the values to floats for plotting
    for key in data:
        data[key] = list(map(float, data[key]))

    # Prepare data for plotting
    x_data = list(data.keys())
    y_data = list(data.values())

    # Create the boxplot
    plt.figure(figsize=(12, 6))  # Increase figure size for better layout
    sns.boxplot(data=y_data)

    plt.xlabel('Objects')
    plt.ylabel(key_name)
    plt.title(f'{key_name} by Object')
    plt.xticks(range(len(x_data)), x_data, rotation=45)  # Rotate x labels for better readability
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(f"./python/output/{key_name}.pdf")

def print_category_statistics(base_directory, key_name):
    data = aggregate_key_values(base_directory, key_name)

    # Flatten the list of values for each category
    all_values = []
    for key in data:
        all_values.extend(map(float, data[key]))

    # Calculate mean, standard deviation, median, lower quartile, upper quartile
    mean_value = np.mean(all_values)
    std_value = np.std(all_values)
    median_value = np.median(all_values)
    lower_quartile = np.percentile(all_values, 25)
    upper_quartile = np.percentile(all_values, 75)
    
    # Calculate IQR (Interquartile Range)
    iqr = upper_quartile - lower_quartile
    
    # Calculate lower whisker and upper whisker
    # Whiskers typically extend to 1.5 * IQR from the quartiles, not beyond the data range
    lower_whisker = max(min(all_values), lower_quartile - 1.5 * iqr)
    upper_whisker = min(max(all_values), upper_quartile + 1.5 * iqr)

    print(f"Category: {key_name}")
    print(f"Mean: {mean_value:.2f}")
    print(f"Standard Deviation: {std_value:.2f}")
    print(f"Median: {median_value:.2f}")
    print(f"Lower Quartile (25th percentile): {lower_quartile:.2f}")
    print(f"Upper Quartile (75th percentile): {upper_quartile:.2f}")
    print(f"Lower Whisker: {lower_whisker:.2f}")
    print(f"Upper Whisker: {upper_whisker:.2f}")
    print()

# Usage
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
    make_boxplot(base_directory, category)
    print_category_statistics(base_directory, category)
