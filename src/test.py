import torch
import torchaudio
from model import EnhancementGenerator

def enhance_file(
    noisy_wav_path,
    output_path,
    model_ckpt,
    device="cuda" if torch.cuda.is_available() else "cpu",
    n_fft=512,
    hop_length=128
):
    # 1. Load model and checkpoint
    generator = EnhancementGenerator().to(device)
    generator.load_state_dict(torch.load(model_ckpt, map_location=device))
    generator.eval()

    # 2. Load audio
    noisy, sr = torchaudio.load(noisy_wav_path)
    assert sr == 16000, "Expected 16kHz audio."
    if noisy.shape[0] > 1:
        noisy = noisy.mean(dim=0, keepdim=True)
    noisy = noisy.to(device)

    # 3. Compute STFT
    stft = torch.stft(noisy.squeeze(0), n_fft=n_fft, hop_length=hop_length, return_complex=True)
    mag = stft.abs().transpose(0, 1)      # [frames, freq]
    phase = torch.angle(stft).transpose(0, 1) # [frames, freq]

    # 4. Prepare input for generator (batch, time, features)
    mag_input = mag.unsqueeze(0)          # [1, frames, freq]
    lengths = torch.tensor([1.0]).to(device)  # full length

    # 5. Forward pass to get mask
    with torch.no_grad():
        mask = generator(mag_input, lengths)   # [1, frames, freq]
    enhanced_mag = mask.squeeze(0) * mag      # [frames, freq]

    # 6. Reconstruct complex spectrogram
    real = enhanced_mag * torch.cos(phase)
    imag = enhanced_mag * torch.sin(phase)
    enhanced_stft = torch.complex(real, imag).transpose(0, 1) # [freq, frames]

    # 7. Inverse STFT to get waveform
    enhanced_wav = torch.istft(
        enhanced_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        length=noisy.shape[1]
    )

    # 8. Save result
    torchaudio.save(output_path, enhanced_wav.unsqueeze(0).cpu(), sr)
    print(f"Enhanced audio saved as {output_path}")

if __name__ == "__main__":
    enhance_file(
        noisy_wav_path="/Users/elbekbakiev/project-thesis/project/backend/testing_sample/noised.wav",
        output_path="enhanced_output.wav",
        model_ckpt="/Users/elbekbakiev/project-thesis/src/checkpoints/metric_gan_model_4.pt"
    )
