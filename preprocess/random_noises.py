import numpy as np
import soundfile as sf
import os

# 🔧 Set the path to your input WAV file
INPUT_PATH = "/Users/elbekbakiev/PycharmProjects/project_thesis/LibriSpeech/train-wav-100/19/227/19-227-0026.wav"

# Set the SNR levels (in dB)
SNR_VALUES = [0, 5, 10, 15, 20]


def add_white_noise(signal, snr_db):
    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"[!] File not found: {INPUT_PATH}")
        return

    signal, sample_rate = sf.read(INPUT_PATH)
    base = os.path.splitext(os.path.basename(INPUT_PATH))[0]

    for snr in SNR_VALUES:
        noisy_signal = add_white_noise(signal, snr)
        noisy_signal = np.clip(noisy_signal, -1.0, 1.0)  # avoid clipping
        output_path = f"{base}_white_noise_snr{snr}.wav"
        sf.write(output_path, noisy_signal, sample_rate)
        print(f"[+] Saved: {output_path}")


if __name__ == "__main__":
    main()
