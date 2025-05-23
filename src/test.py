import torch
import librosa
import numpy as np
import soundfile as sf
from model import TDNN

sr = 16000
n_mels = 128
segment_duration = 3.0  # в секундах
segment_samples = int(segment_duration * sr)

corrupted_audio_path = '../LibriSpeech/train-corrupted-2/19/198/19-198-0003.flac'
restored_audio_path = '../restored_long_20.flac'
model_path = '../outputs/model_epoch_20.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TDNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

audio, _ = librosa.load(corrupted_audio_path, sr=sr, mono=True)

chunks = []
starts = np.arange(0, len(audio), segment_samples)
for start in starts:
    end = min(start + segment_samples, len(audio))
    chunk = audio[start:end]
    if len(chunk) < segment_samples:
        chunk = np.pad(chunk, (0, segment_samples - len(chunk)))
    chunks.append(chunk)

print(f'Total chunks: {len(chunks)}')

restored_chunks = []
for i, chunk in enumerate(chunks):
    # 1. Получить mel-спектрограмму
    mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mean, std = mel_db.mean(), mel_db.std()
    mel_db_norm = (mel_db - mean) / (std + 1e-9)
    # 2. Преобразовать в тензор и прогнать через модель
    input_tensor = torch.tensor(mel_db_norm, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 128, T]
    with torch.no_grad():
        output_tensor = model(input_tensor)
    restored_mel = output_tensor.squeeze().cpu().numpy()  # [128, time]
    # 3. Обратно денормализовать
    restored_mel_db = restored_mel * std + mean
    # 4. Инвертировать в аудио
    restored_mel_power = librosa.db_to_power(restored_mel_db)
    restored_audio = librosa.feature.inverse.mel_to_audio(restored_mel_power, sr=sr, n_iter=32)
    # 5. Обрезать до оригинального размера (если последний кусок)
    if i == len(chunks) - 1:
        restored_audio = restored_audio[:end - start]
    restored_chunks.append(restored_audio)

# === Склеиваем все куски обратно ===
final_audio = np.concatenate(restored_chunks)
sf.write(restored_audio_path, final_audio, sr)
print(f'Saved restored audio to {restored_audio_path}')
