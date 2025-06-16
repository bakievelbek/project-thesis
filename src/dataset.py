# dataset.py
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

class SpeechPairsDataset(Dataset):
    def __init__(self, csv_path, sample_rate=16000, segment_length=3.0):
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.segment_length = int(sample_rate * segment_length)

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        audio, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        audio = audio.mean(dim=0)  # mono
        # Cut or pad to self.segment_length
        if audio.shape[0] > self.segment_length:
            audio = audio[:self.segment_length]
        elif audio.shape[0] < self.segment_length:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.shape[0]))
        return audio

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clean = self.load_audio(row['clean'])
        noisy = self.load_audio(row['noisy'])
        return {
            'clean': clean,
            'noisy': noisy,
        }
