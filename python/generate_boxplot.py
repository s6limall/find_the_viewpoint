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

        # Find all files that end with "_meta"
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

def make_violinplot
# Usage
base_directory = './task2/selected_views/'  # Replace with the path to the directory containing obj_ folders
key_name = 'distance_to_target'

data = aggregate_key_values(base_directory, key_name)
print(data)

# Convert the values to floats for plotting
for key in data:
    data[key] = list(map(float, data[key]))

# Prepare data for plotting
x_data = []
y_data = []

for obj_name, values in data.items():
    x_data.extend([obj_name] * len(values))
    y_data.extend(values)

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=x_data, y=y_data)

plt.xlabel('Objects')
plt.ylabel('Values')
plt.title('Violin Plot of Values by Object')
plt.xticks(rotation=45)  # Rotate x labels for better readability
plt.savefig("./python/output/test.svg")