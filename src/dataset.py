import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class AudioPairsDataset(Dataset):
    def __init__(self, csv_path, audio_len=3, sample_rate=16000, n_fft=512, hop_length=128):
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.audio_len = audio_len * sample_rate  # seconds to samples
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        noisy_path = self.df.iloc[idx]['noisy']
        clean_path = self.df.iloc[idx]['clean']
        noisy, sr1 = torchaudio.load(noisy_path)
        clean, sr2 = torchaudio.load(clean_path)
        # Resample if needed
        if sr1 != self.sample_rate:
            noisy = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=self.sample_rate)(noisy)
        if sr2 != self.sample_rate:
            clean = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=self.sample_rate)(clean)
        # Trim or pad to audio_len samples
        noisy = noisy[:, :self.audio_len]
        clean = clean[:, :self.audio_len]
        if noisy.shape[1] < self.audio_len:
            pad = self.audio_len - noisy.shape[1]
            noisy = torch.nn.functional.pad(noisy, (0, pad))
            clean = torch.nn.functional.pad(clean, (0, pad))

        # STFT (for generator and discriminator)
        stft_noisy = torch.stft(
            noisy.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True
        )  # [freq, frames]
        stft_clean = torch.stft(
            clean.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True
        )
        # Magnitude and phase (only magnitude for generator)
        mag_noisy = stft_noisy.abs().T  # [frames, freq]
        mag_clean = stft_clean.abs().T  # [frames, freq]

        # For discriminator, need real+imag channels: [2, freq, frames]
        spec_noisy = torch.stack([stft_noisy.real, stft_noisy.imag], dim=0)
        spec_clean = torch.stack([stft_clean.real, stft_clean.imag], dim=0)

        return mag_noisy, mag_clean, spec_noisy, spec_clean
