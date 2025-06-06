import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import random

class SpeechEnhancementDataset(Dataset):
    def __init__(self, csv_file, segment_length=4.0, sample_rate=16000, augment=True):
        self.df = pd.read_csv(csv_file)
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform[0]
        return waveform

    def random_chunk(self, waveform):
        num_samples = int(self.segment_length * self.sample_rate)
        if len(waveform) > num_samples:
            start = random.randint(0, len(waveform) - num_samples)
            chunk = waveform[start:start+num_samples]
        else:
            chunk = torch.zeros(num_samples)
            chunk[:len(waveform)] = waveform
        return chunk

    def add_white_noise(self, audio, snr_db=10):
        rms = audio.pow(2).mean().sqrt()
        noise = torch.randn_like(audio)
        noise = noise / noise.pow(2).mean().sqrt() * rms / (10 ** (snr_db / 20))
        return audio + noise

    def add_random_gaps(self, audio, gap_prob=0.1, gap_length=0.1):
        audio = audio.clone()
        gap_samples = int(gap_length * self.sample_rate)
        i = 0
        while i < len(audio):
            if random.random() < gap_prob:
                end = min(i + gap_samples, len(audio))
                audio[i:end] = 0
                i = end
            else:
                i += 1
        return audio

    def __getitem__(self, idx):
        clean_path = self.df.iloc[idx]['clean_path']
        waveform = self.load_audio(clean_path)
        waveform = self.random_chunk(waveform)

        if self.augment:
            corrupted = waveform.clone()
            if random.random() < 0.5:
                corrupted = self.add_white_noise(corrupted, snr_db=random.uniform(5, 20))
            if random.random() < 0.5:
                corrupted = self.add_random_gaps(corrupted, gap_prob=0.05, gap_length=0.05)
        else:
            corrupted = waveform.clone()

        # Convert to magnitude spectrograms
        spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128, power=2)(waveform)
        corrupted_spec = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128, power=2)(corrupted)

        # Shape: (freq_bins, time_frames)
        # For models: (time_frames, freq_bins)
        spectrogram = spectrogram.T
        corrupted_spec = corrupted_spec.T

        return corrupted_spec, spectrogram  # (corrupted, clean)
