import torch
import torchaudio
from model import build_hybrid_model

def enhance_audio(noisy_path, output_path, model_ckpt, sample_rate=16000):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_hybrid_model(device)
    model.tdnn.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    noisy, sr = torchaudio.load(noisy_path)
    if sr != sample_rate:
        noisy = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(noisy)
    if noisy.shape[0] > 1:
        noisy = noisy.mean(dim=0, keepdim=True)
    noisy = noisy.unsqueeze(0)  # [1, 1, T]
    noisy = noisy.to(device)

    with torch.no_grad():
        enhanced = model(noisy, sample_rate)
    enhanced = enhanced.squeeze().cpu()
    torchaudio.save(output_path, enhanced.unsqueeze(0), sample_rate)
    print(f"Enhanced audio saved to {output_path}")

if __name__ == "__main__":
    noisy_path = "/Users/elbekbakiev/project-thesis/project/backend/testing_sample/noised.wav"
    output_path = "enhanced_output.wav"
    model_ckpt = "checkpoints/hybrid_3.pt"
    enhance_audio(noisy_path, output_path, model_ckpt)
