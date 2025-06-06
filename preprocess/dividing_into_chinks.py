# import os
# import librosa
# import soundfile as sf
#
# # Configuration
#
#
# input_dir = '../LibriSpeech/train-wav-100'
# output_dir = '../LibriSpeech/train-chunks-100'
#
# chunk_duration = 3.0  # seconds
# sr = 16000  # target sampling rate
#
#
# def split_and_save(file_path, out_base_dir, chunk_duration=3.0, sr=16000):
#     audio, _ = librosa.load(file_path, sr=sr, mono=True)
#     chunk_samples = int(chunk_duration * sr)
#     total_samples = len(audio)
#
#     # Calculate how many chunks we can make
#     num_chunks = total_samples // chunk_samples
#
#     # Relative path to maintain folder structure
#     rel_path = os.path.relpath(file_path, input_dir)
#     rel_dir = os.path.dirname(rel_path)
#
#     for i in range(num_chunks):
#         start = i * chunk_samples
#         end = start + chunk_samples
#         chunk_audio = audio[start:end]
#
#         # Create output directory if not exists
#         out_dir = os.path.join(out_base_dir, rel_dir)
#         os.makedirs(out_dir, exist_ok=True)
#
#         # Filename pattern: originalname_chunkXX.wav
#         base_name = os.path.splitext(os.path.basename(file_path))[0]
#         out_file = os.path.join(out_dir, f"{base_name}_chunk{i + 1:02d}.wav")
#
#         # Save chunk
#         sf.write(out_file, chunk_audio, sr)
#         print(f"Saved chunk: {out_file}")
#
#
# def process_all(input_dir, output_dir, chunk_duration=3.0, sr=16000):
#     for root, _, files in os.walk(input_dir):
#         for file in files:
#             if file.startswith('._'):
#                 continue
#             if file.lower().endswith('.wav'):
#                 full_path = os.path.join(root, file)
#                 split_and_save(full_path, output_dir, chunk_duration, sr)
#
#
# if __name__ == "__main__":
#     process_all(input_dir, output_dir, chunk_duration, sr)

import os
import soundfile as sf
import numpy as np

chunk_duration = 3  # seconds
sr = 16000
chunk_samples = chunk_duration * sr

input_dir_clean = '../LibriSpeech/train-wav-100'    # full length clean audios
input_dir_corrupt = '../LibriSpeech/train-dropouts-full-100/'  # full length dropouted audios
output_clean_chunks = 'data_chunks/clean'  #
output_corrupt_chunks = 'data_chunks/corrupt'

os.makedirs(output_clean_chunks, exist_ok=True)
os.makedirs(output_corrupt_chunks, exist_ok=True)

for fname in os.listdir(input_dir_clean):
    if not fname.endswith(".wav"):
        continue

    clean_path = os.path.join(input_dir_clean, fname)
    corrupt_path = os.path.join(input_dir_corrupt, fname)

    # Load both
    clean_audio, sr1 = sf.read(clean_path)
    corrupt_audio, sr2 = sf.read(corrupt_path)

    if sr1 != sr or sr2 != sr:
        print(f"Sampling rate mismatch in {fname}")
        continue

    min_len = min(len(clean_audio), len(corrupt_audio))
    clean_audio = clean_audio[:min_len]
    corrupt_audio = corrupt_audio[:min_len]

    total_chunks = min_len // chunk_samples

    for i in range(total_chunks):
        start = i * chunk_samples
        end = start + chunk_samples

        clean_chunk = clean_audio[start:end]
        corrupt_chunk = corrupt_audio[start:end]

        base_name = fname.replace('.wav', f'_chunk{i:03d}.wav')

        sf.write(os.path.join(output_clean_chunks, base_name), clean_chunk, sr)
        sf.write(os.path.join(output_corrupt_chunks, base_name), corrupt_chunk, sr)

    print(f"Processed {fname} → {total_chunks} chunks")

