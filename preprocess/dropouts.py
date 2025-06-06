import os
import librosa
import soundfile as sf
import numpy as np

original_dataset_path = '../LibriSpeech/train-wav-100/'
corrupted_dataset_path = '../LibriSpeech/train-dropouts-full-100/'

sr = 16000
dropout_ms = 50


def apply_multiple_dropouts(segment, sr=16000, min_dropout_ms=150, max_dropout_ms=200, num_dropouts=3):
    segment = segment.copy()
    audio_length = len(segment)

    for _ in range(num_dropouts):
        dropout_ms = np.random.randint(min_dropout_ms, max_dropout_ms + 1)
        dropout_samples = int(sr * dropout_ms / 1000)
        start = np.random.randint(0, audio_length - dropout_samples)
        segment[start:start + dropout_samples] = 0

    return segment


for root, dirs, files in os.walk(original_dataset_path):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            if file.startswith('._'):
                continue
            audio, _ = librosa.load(file_path, sr=sr, mono=True)
            corrupted_audio = apply_multiple_dropouts(audio, sr=sr)

            relative_path = os.path.relpath(root, original_dataset_path)
            output_dir = os.path.join(corrupted_dataset_path, relative_path)

            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, file)
            print(output_file_path)
            sf.write(output_file_path, corrupted_audio, sr)
            print(f"Processed and saved: {output_file_path}")
