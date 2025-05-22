import torch
import librosa
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SpeechReconstructionDataset(Dataset):
    def __init__(self, csv_path, sr=16000, n_mels=128, segment_duration=None, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV index file with corrupted and clean audio paths.
            sr (int): Sampling rate for loading audio.
            n_mels (int): Number of Mel filter banks.
            segment_duration (float or None): Duration in seconds to crop audio (optional).
            transform (callable or None): Optional transform to apply to features.
        """
        self.data = pd.read_csv(csv_path)
        self.sr = sr
        self.n_mels = n_mels
        self.segment_duration = segment_duration
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def load_audio(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        if self.segment_duration:
            max_len = int(self.segment_duration * self.sr)
            if len(audio) > max_len:
                audio = audio[:max_len]
            else:
                # Pad shorter audio with zeros
                audio = np.pad(audio, (0, max_len - len(audio)))
        return audio

    def extract_mel_spectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_db

    def __getitem__(self, idx):
        corrupted_path = self.data.iloc[idx]['corrupted_path']
        clean_path = self.data.iloc[idx]['clean_path']

        corrupted_audio = self.load_audio(corrupted_path)
        clean_audio = self.load_audio(clean_path)

        corrupted_feat = self.extract_mel_spectrogram(corrupted_audio)
        clean_feat = self.extract_mel_spectrogram(clean_audio)

        # Optional transform (e.g., normalization)
        if self.transform:
            corrupted_feat = self.transform(corrupted_feat)
            clean_feat = self.transform(clean_feat)

        # Convert to torch tensors, add channel dimension (C x F x T)
        corrupted_tensor = torch.tensor(corrupted_feat, dtype=torch.float).unsqueeze(0)
        clean_tensor = torch.tensor(clean_feat, dtype=torch.float).unsqueeze(0)
        return corrupted_tensor, clean_tensor


dataset = SpeechReconstructionDataset(
    csv_path='D:\\PyCharm projects\\project-thesis\\preprocess\\dataset_index.csv',
    segment_duration=3.0)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for corrupted_batch, clean_batch in dataloader:
    print(corrupted_batch.shape)  # e.g., (16, 1, 128, time_frames)
    print(clean_batch.shape)
    break
