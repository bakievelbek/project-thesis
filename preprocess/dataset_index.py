# import os
# import csv
#
# corrupted_root = '../LibriSpeech/train-corrupted-100'
# clean_root = '../LibriSpeech/train-chunks-100'
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



# import os
# import csv
# import random
#
# corrupt_dir = '/Users/elbekbakiev/project-thesis/LibriSpeech/train-chunks-corrupt-100/'
# clean_dir = '/Users/elbekbakiev/project-thesis/LibriSpeech/train-chunks-clean-100'
# metadata_dir = 'metadata'
# os.makedirs(metadata_dir, exist_ok=True)
#
# pairs = []
#
# for fname in os.listdir(corrupt_dir):
#     corrupt_path = os.path.join(corrupt_dir, fname)
#     clean_path = os.path.join(clean_dir, fname)
#     if os.path.exists(clean_path):
#         pairs.append((corrupt_path, clean_path))
#
# print(f"Найдено пар: {len(pairs)}")
#
# # Перемешиваем и делим
# random.seed(42)
# random.shuffle(pairs)
#
# total = len(pairs)
# train_size = int(0.8 * total)
# val_size = int(0.1 * total)
#
# train = pairs[:train_size]
# val = pairs[train_size:train_size+val_size]
# test = pairs[train_size+val_size:]
#
# def write_csv(pairs, path):
#     with open(path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["input_path", "target_path"])
#         writer.writerows(pairs)
#
# write_csv(train, os.path.join(metadata_dir, "train.csv"))
# write_csv(val, os.path.join(metadata_dir, "val.csv"))
# write_csv(test, os.path.join(metadata_dir, "test.csv"))
#
# print(f"✅ CSV-файлы сохранены в {metadata_dir}")



import pandas as pd

INPUT_CSV = "metadata/train.csv"
OUTPUT_CSV = "metadata/train-100.csv"
N = 100   # количество строк (можешь выбрать 20, 50, 100...)

# Читаем исходный train.csv
df = pd.read_csv(INPUT_CSV)

# Берем только первые N строк (можно заменить на random.sample, если хочешь случайный поднабор)
df_small = df.sample(n=N, random_state=42)


# Сохраняем новый csv
df_small.to_csv(OUTPUT_CSV, index=False)

print(f"Сохранено {N} строк в {OUTPUT_CSV}")
