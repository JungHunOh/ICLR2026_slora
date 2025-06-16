import os
import re
import csv
from collections import defaultdict
import ast

data = 'metamath'

# Directory containing the experiment text files
experiment_dir = './experiment'

# Pattern to match filenames and extract parameters
filename_pattern = re.compile(
    rf"llama3_{data}(?P<dl>\d+)bs(?P<bs>\d+)epoch(?P<epoch>\d+)_(?P<method>[\w\d]+)\_r(?P<r>\d+)_lr0.0002_seed(?P<seed>\d+)_(?P<dataset>[\w\d]+)\.txt"
)

# filename_pattern = re.compile(
#     rf"llama3_{data}(?P<dl>\d+)bs(?P<bs>\d+)epoch(?P<epoch>\d+)_(?P<method>[\w\d]+)\_r(?P<r>\d+)_lr(?P<lr>[\d.]+)_seed(?P<seed>\d+)_(?P<dataset>[\w\d]+)\.txt"
# )

# Output CSV file
output_csv = 'summary.csv'

results = defaultdict(list)

# Process each file in the directory
for filename in os.listdir(experiment_dir):
    match = filename_pattern.match(filename)
    if match:
        args = match.groupdict()
        key = (
            int(args['dl']),
            int(args['bs']),
            int(args['epoch']),
            str(args['method']),
            int(args['r']),
        )

        filepath = os.path.join(experiment_dir, filename)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2 and lines[1].startswith("Accuracy:"):
                    accuracy = float(lines[1].split("Accuracy:")[1].strip())
                    results[key].append(accuracy)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue

# Write averaged results to CSV
output_csv = 'summary_math.csv'
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['dl', 'bs', 'epoch', 'r', 'lora', 'slora', 'improvement'])

    tmp = {}
    for key, accuracies in sorted(results.items()):
        tmp[str(list(key[:3]) + list(key[4:]))] = [0,0]
    
    for key, accuracies in sorted(results.items()):
        real_key = str(list(key[:3]) + list(key[4:]))
        avg_accuracy = round(sum(accuracies) / len(accuracies)*100,2)
        if 'slora' in key:
            tmp[real_key][1] = avg_accuracy
        else:
            tmp[real_key][0] = avg_accuracy
    
    for key, accuracies in sorted(results.items()):
        real_key = str(list(key[:3]) + list(key[4:]))
        try:
            accuracies = tmp[real_key]
            writer.writerow(ast.literal_eval(real_key) + accuracies + [round(accuracies[1]-accuracies[0],2)])
            del tmp[real_key]
        except:
            pass

import csv
from tabulate import tabulate

csv_file = 'summary_math.csv'
output_file = 'summary_math.txt'

# Read CSV
with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    rows = list(reader)

# Format table
table_str = tabulate(rows[1:], headers=rows[0], tablefmt='github')

# Save to file
with open(output_file, 'w') as f:
    f.write(table_str)

print(f"Table saved to {output_file}")