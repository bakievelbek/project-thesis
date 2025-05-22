import os
import csv

corrupted_root = 'D:\\PyCharm projects\\project-thesis\\LibriSpeech\\train-corrupted-100'
clean_root = 'D:\\PyCharm projects\\project-thesis\\LibriSpeech\\train-chunks-100'
output_csv = 'dataset_index.csv'

pairs = []

for root, _, files in os.walk(corrupted_root):
    for file in files:
        if file.endswith('.flac'):
            # Compute relative path from corrupted root
            rel_dir = os.path.relpath(root, corrupted_root)
            corrupted_path = os.path.join(root, file)

            # Corresponding clean file path
            clean_path = os.path.join(clean_root, rel_dir, file)

            # Check if clean file exists
            if os.path.isfile(clean_path):
                pairs.append((corrupted_path, clean_path))
            else:
                print(f"Warning: Clean file not found for {corrupted_path}")

# Write to CSV
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['corrupted_path', 'clean_path'])
    writer.writerows(pairs)

print(f"Index file created with {len(pairs)} pairs at '{output_csv}'")
