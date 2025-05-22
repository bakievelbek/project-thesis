import os
import librosa
import soundfile as sf

# Configuration


input_dir = 'D:\\PyCharm projects\\project-thesis\\LibriSpeech\\train-clean-100'
output_dir = 'D:\\PyCharm projects\\project-thesis\\LibriSpeech\\train-chunks-100'

chunk_duration = 3.0  # seconds
sr = 16000  # target sampling rate


def split_and_save(file_path, out_base_dir, chunk_duration=3.0, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    chunk_samples = int(chunk_duration * sr)
    total_samples = len(audio)

    # Calculate how many chunks we can make
    num_chunks = total_samples // chunk_samples

    # Relative path to maintain folder structure
    rel_path = os.path.relpath(file_path, input_dir)
    rel_dir = os.path.dirname(rel_path)

    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk_audio = audio[start:end]

        # Create output directory if not exists
        out_dir = os.path.join(out_base_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Filename pattern: originalname_chunkXX.flac
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_file = os.path.join(out_dir, f"{base_name}_chunk{i + 1:02d}.flac")

        # Save chunk
        sf.write(out_file, chunk_audio, sr)
        print(f"Saved chunk: {out_file}")


def process_all(input_dir, output_dir, chunk_duration=3.0, sr=16000):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith('._'):
                continue
            if file.lower().endswith('.flac'):
                full_path = os.path.join(root, file)
                split_and_save(full_path, output_dir, chunk_duration, sr)


if __name__ == "__main__":
    process_all(input_dir, output_dir, chunk_duration, sr)
