import os
import librosa
import soundfile as sf
import numpy as np

original_dataset_path = '../LibriSpeech/train-wav-100/19/198'
noisy_dataset_path = '../LibriSpeech/train-noisy-full-100/19/198/'
sr = 16000


def add_white_noise(audio, snr_db):
    rms = np.sqrt(np.mean(audio**2))
    noise_std = rms / (10**(snr_db / 20))
    noise = np.random.normal(0, noise_std, audio.shape)
    return audio + noise

def add_pink_noise(audio, snr_db):
    # Генерируем белый шум, потом фильтруем его
    rng = np.random.default_rng()
    white = rng.normal(size=audio.shape)
    # Взвешиваем спектр шума, чтобы сделать его "розовым"
    fft = np.fft.rfft(white)
    S = np.arange(1, fft.size + 1)
    fft = fft / np.sqrt(S)
    pink = np.fft.irfft(fft)
    pink = pink[:audio.shape[0]]
    # Нормируем под нужный SNR
    rms = np.sqrt(np.mean(audio**2))
    noise_std = rms / (10**(snr_db / 20))
    pink *= noise_std / np.std(pink)
    return audio + pink

def add_impulse_noise(audio, snr_db, num_impulses=10):
    noisy = audio.copy()
    for _ in range(num_impulses):
        idx = np.random.randint(0, len(noisy))
        amplitude = np.max(np.abs(audio)) / (10**(snr_db / 20)) * np.random.uniform(0.5, 1.5)
        noisy[idx] += amplitude * np.random.choice([-1, 1])
    return noisy

snr_levels = [0, 5, 10]

for root, dirs, files in os.walk(original_dataset_path):
    for file in files:
        if file.endswith('.flac'):
            file_path = os.path.join(root, file)
            if file.startswith('._'):
                continue
            audio, _ = librosa.load(file_path, sr=sr, mono=True)

            for snr in snr_levels:
                for noise_type in ['white']:
                    if noise_type == 'white':
                        noisy = add_white_noise(audio, snr)
                    # elif noise_type == 'pink':
                    #     noisy = add_pink_noise(audio, snr)
                    # elif noise_type == 'impulse':
                    #     noisy = add_impulse_noise(audio, snr)

                    relative_path = os.path.relpath(root, original_dataset_path)
                    output_dir = os.path.join(noisy_dataset_path, f"{noise_type}_snr{snr}", relative_path)
                    os.makedirs(output_dir, exist_ok=True)
                    output_file_path = os.path.join(output_dir, file).replace('.flac', '.wav')

                    sf.write(output_file_path, noisy, sr)
                    print(f"Saved: {output_file_path}")
