import os
import librosa
import numpy as np

# Change this to your extracted dataset path
LIBRISPEECH_PATH = 'LibriSpeech/train-clean-100/'


def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio


def segment_audio(audio, segment_length=3, sr=16000):
    segment_samples = segment_length * sr
    segments = []
    for start in range(0, len(audio), segment_samples):
        end = start + segment_samples
        segment = audio[start:end]
        if len(segment) == segment_samples:
            segments.append(segment)
    return segments


def apply_multiple_dropouts(segment, sr=16000, dropout_ms=50, num_dropouts=3):
    segment = segment.copy()
    dropout_samples = int(sr * dropout_ms / 1000)
    audio_length = len(segment)

    for _ in range(num_dropouts):
        start = np.random.randint(0, audio_length - dropout_samples)
        segment[start:start + dropout_samples] = 0

    return segment


def preprocess_librispeech(libri_path, max_files=1000):
    processed_segments = []
    count = 0
    for root, _, files in os.walk(libri_path):
        for file in files:
            if file.endswith('.flac'):
                file_path = os.path.join(root, file)
                audio = load_audio(file_path)
                segments = segment_audio(audio)
                for seg in segments:
                    corrupted = apply_multiple_dropouts(seg)
                    processed_segments.append((corrupted, seg))
                count += 1
    return processed_segments


if __name__ == "__main__":
    dataset = preprocess_librispeech(LIBRISPEECH_PATH, max_files=5)
    print(f"Processed {len(dataset)} segments.")
    # dataset contains tuples of (corrupted_segment, original_segment)
