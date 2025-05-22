import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pystoi import stoi
# from pesq import pesq

matplotlib.use('TkAgg')

def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio


def plot_waveforms(original, corrupted, sr=16000):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original Waveform')
    plt.plot(np.linspace(0, len(original) / sr, len(original)), original)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(1, 2, 2)
    plt.title('Corrupted Waveform')
    plt.plot(np.linspace(0, len(corrupted) / sr, len(corrupted)), corrupted)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def plot_mel_spectrograms(original, corrupted, sr=16000, n_mels=128):
    plt.figure(figsize=(12, 6))

    # Original Mel Spectrogram
    plt.subplot(2, 1, 1)
    S_orig = librosa.feature.melspectrogram(y=original, sr=sr, n_mels=n_mels)
    S_db_orig = librosa.power_to_db(S_orig, ref=np.max)
    librosa.display.specshow(S_db_orig, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Original Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    # Corrupted Mel Spectrogram
    plt.subplot(2, 1, 2)
    S_corr = librosa.feature.melspectrogram(y=corrupted, sr=sr, n_mels=n_mels)
    S_db_corr = librosa.power_to_db(S_corr, ref=np.max)
    librosa.display.specshow(S_db_corr, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Corrupted Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()


def calculate_snr(original, corrupted):
    noise = original - corrupted
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def main(original_path, corrupted_path):
    sr = 16000
    # Load audio
    original = load_audio(original_path, sr)
    corrupted = load_audio(corrupted_path, sr)

    # Make sure lengths are equal
    min_len = min(len(original), len(corrupted))
    original = original[:min_len]
    corrupted = corrupted[:min_len]

    # Plot waveforms
    plot_waveforms(original, corrupted, sr)

    # Plot mel spectrograms
    import librosa.display  # imported here to avoid issues if matplotlib not installed
    plot_mel_spectrograms(original, corrupted, sr)

    # Calculate metrics
    snr_value = calculate_snr(original, corrupted)
    # pesq_value = pesq(sr, original, corrupted, 'wb')  # 'wb' for wideband (16kHz)
    stoi_value = stoi(original, corrupted, sr, extended=False)

    print(f"SNR: {snr_value:.2f} dB")
    # print(f"PESQ: {pesq_value:.3f}")
    print(f"STOI: {stoi_value:.3f}")


if __name__ == "__main__":
    original_audio_path = "D:\\PyCharm projects\\project-thesis\\LibriSpeech\\train-chunks-100\\19\\198\\19-198-0033_chunk01.flac"
    corrupted_audio_path = "D:\\PyCharm projects\\project-thesis\\LibriSpeech\\train-corrupted-100\\19\\198\\19-198-0033_chunk01.flac"
    main(original_audio_path, corrupted_audio_path)
