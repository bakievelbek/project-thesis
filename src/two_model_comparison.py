import torch
import librosa
import numpy as np
import soundfile as sf
from model import Conv2dAE
from pystoi import stoi  # pip install pystoi
from mir_eval.separation import bss_eval_sources  # pip install mir_eval

def compute_snr(clean, enhanced):
    eps = 1e-9
    clean, enhanced = np.asarray(clean), np.asarray(enhanced)
    min_len = min(len(clean), len(enhanced))
    clean, enhanced = clean[:min_len], enhanced[:min_len]
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum((clean - enhanced) ** 2) + eps
    return 10 * np.log10(signal_power / noise_power + eps)

def compute_sdr(clean, enhanced):
    min_len = min(len(clean), len(enhanced))
    clean, enhanced = clean[:min_len], enhanced[:min_len]
    sdr, _, _, _ = bss_eval_sources(clean[None, :], enhanced[None, :])
    return sdr[0]

def process_with_model(audio, model, sr=16000, n_mels=128, n_mfcc=13, segment_duration=3.0):
    segment_samples = int(segment_duration * sr)
    model.eval()
    chunks = []
    starts = np.arange(0, len(audio), segment_samples)
    for start in starts:
        end = min(start + segment_samples, len(audio))
        chunk = audio[start:end]
        if len(chunk) < segment_samples:
            chunk = np.pad(chunk, (0, segment_samples - len(chunk)))
        # features
        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < mel_db.shape[1]:
            mfcc = np.pad(mfcc, ((0, 0), (0, mel_db.shape[1] - mfcc.shape[1])))
        elif mfcc.shape[1] > mel_db.shape[1]:
            mfcc = mfcc[:, :mel_db.shape[1]]
        features = np.concatenate([mel_db, mfcc], axis=0)
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True)
        features_norm = (features - mean) / (std + 1e-9)
        input_tensor = torch.tensor(features_norm, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(next(model.parameters()).device)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        restored = output_tensor.squeeze().cpu().numpy()
        min_freq = min(restored.shape[0], features.shape[0])
        min_time = min(restored.shape[1], features.shape[1])
        restored = restored[:min_freq, :min_time]
        mean = mean[:min_freq]
        std = std[:min_freq]
        restored = restored * std + mean
        restored_mel_db = restored[:n_mels, :]
        restored_mel_power = librosa.db_to_power(restored_mel_db)
        restored_audio = librosa.feature.inverse.mel_to_audio(restored_mel_power, sr=sr, n_iter=32)
        if start + segment_samples > len(audio):
            restored_audio = restored_audio[:end - start]
        chunks.append(restored_audio)
    return np.concatenate(chunks)

# === Настройки ===
sr = 16000
corrupted_audio_path = '../LibriSpeech/train-noisy-100/white_snr0/19/198/19-198-0001_chunk02.wav'
clean_audio_path = '../LibriSpeech/train-clean-wav-100/19/198/19-198-0001_chunk02.wav'
model1_path = '../outputs/conv2dae_model_1.pth'
model2_path = '../outputs/conv2dae_model_4.pth'

# === Загрузка моделей ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = Conv2dAE(input_channels=1).to(device)
model1.load_state_dict(torch.load(model1_path, map_location=device))
model2 = Conv2dAE(input_channels=1).to(device)
model2.load_state_dict(torch.load(model2_path, map_location=device))

# === Аудио ===
corrupted_audio, _ = librosa.load(corrupted_audio_path, sr=sr, mono=True)
clean_audio, _ = librosa.load(clean_audio_path, sr=sr, mono=True)

# === Прогон ===
restored1 = process_with_model(corrupted_audio, model1, sr=sr)
restored2 = process_with_model(corrupted_audio, model2, sr=sr)

# === Метрики ===
snr1 = compute_snr(clean_audio, restored1)
sdr1 = compute_sdr(clean_audio, restored1)
snr2 = compute_snr(clean_audio, restored2)
sdr2 = compute_sdr(clean_audio, restored2)

print(f"Model 1: SNR = {snr1:.2f} dB | SDR = {sdr1:.2f} dB")
print(f"Model 2: SNR = {snr2:.2f} dB | SDR = {sdr2:.2f} dB")
