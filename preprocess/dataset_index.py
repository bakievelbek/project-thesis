# import os
# import csv
#
# corrupt_dir = '../LibriSpeech/train-chunks-noisy-100/'
# clean_dir = '../LibriSpeech/train-chunks-100'
# output_csv = 'dataset_index.csv'
#
# pairs = []
#
# for root, _, files in os.walk(corrupted_root):
#     for file in files:
#         if file.endswith('.flac'):
#             # Compute relative path from corrupted root
#             rel_dir = os.path.relpath(root, corrupted_root)
#             corrupted_path = os.path.join(root, file)
#
#             # Corresponding clean file path
#             clean_path = os.path.join(clean_root, rel_dir, file)
#
#             # Check if clean file exists
#             if os.path.isfile(clean_path):
#                 pairs.append((corrupted_path, clean_path))
#             else:
#                 print(f"Warning: Clean file not found for {corrupted_path}")
#
# # Write to CSV
# with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['corrupted_path', 'clean_path'])
#     writer.writerows(pairs)
#
# print(f"Index file created with {len(pairs)} pairs at '{output_csv}'")



import os
import csv
import random

corrupt_dir = '../LibriSpeech/train-chunks-noisy-100/'
clean_dir = '../LibriSpeech/train-chunks-100'
metadata_dir = 'metadata'
os.makedirs(metadata_dir, exist_ok=True)

pairs = []

for root, _, files in os.walk(corrupt_dir):
    for file in files:
        if file.endswith('.wav'):
            rel_dir = os.path.join(root)
            corrupted_path = os.path.join(root, file)

            clean_path = corrupted_path.replace('train-chunks-noisy-100', 'train-chunks-100')

            if os.path.exists(clean_path):
                pairs.append((corrupted_path, clean_path))

print(f"Найдено пар: {len(pairs)}")

# Перемешиваем и делим
random.seed(42)
random.shuffle(pairs)

total = len(pairs)
train_size = int(0.9 * total)
val_size = int(0.1 * total)

train = pairs[:train_size]
val = pairs[train_size:train_size+val_size]

def write_csv(pairs, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["noisy", "clean"])
        writer.writerows(pairs)

write_csv(train, os.path.join(metadata_dir, "train.csv"))
write_csv(val, os.path.join(metadata_dir, "val.csv"))

print(f"✅ CSV-файлы сохранены в {metadata_dir}")
#
#
