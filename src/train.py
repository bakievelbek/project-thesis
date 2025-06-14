import torch
from torch.utils.data import DataLoader
from dataset import AudioPairsDataset
from model import EnhancementGenerator, MetricDiscriminator
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train_metricgan(
    train_csv, val_csv,
    epochs=20, batch_size=8, lr=1e-4,
    checkpoint_dir="checkpoints", device=None,
    n_fft=512, hop_length=128
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Data ---
    train_ds = AudioPairsDataset(train_csv, n_fft=n_fft, hop_length=hop_length)
    val_ds = AudioPairsDataset(val_csv, n_fft=n_fft, hop_length=hop_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- Model ---
    generator = EnhancementGenerator().to(device)
    discriminator = MetricDiscriminator().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        total_g_loss, total_d_loss = 0, 0
        print(f"\n[Epoch {epoch}/{epochs}]")

        # --- Training loop ---
        for mag_noisy, mag_clean, spec_noisy, spec_clean in tqdm(train_loader, desc="Train"):
            mag_noisy = mag_noisy.to(device)   # [B, frames, freq]
            mag_clean = mag_clean.to(device)
            spec_noisy = spec_noisy.to(device) # [B, 2, freq, frames]
            spec_clean = spec_clean.to(device)

            lengths = torch.ones(mag_noisy.shape[0]).to(device)  # [B]

            # --- Generator forward ---
            mask = generator(mag_noisy, lengths)     # [B, frames, freq]
            enhanced_mag = mask * mag_noisy          # [B, frames, freq]

            # Prepare enhanced spec for discriminator
            # We'll use the phase of noisy for enhanced, as in inference
            # So enhanced_spec = [real, imag] (for disc)
            phase_noisy = torch.atan2(spec_noisy[:,1], spec_noisy[:,0])  # [B, freq, frames]
            real = enhanced_mag.transpose(1,2) * torch.cos(phase_noisy)
            imag = enhanced_mag.transpose(1,2) * torch.sin(phase_noisy)
            enhanced_spec = torch.stack([real, imag], dim=1)  # [B, 2, freq, frames]

            # --- Discriminator step ---
            real_score = discriminator(spec_clean)
            fake_score = discriminator(enhanced_spec.detach())
            d_loss = -torch.mean(real_score) + torch.mean(fake_score)
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # --- Generator adversarial + MSE loss ---
            fake_score = discriminator(enhanced_spec)
            g_adv = -torch.mean(fake_score)
            g_mse = mse_loss(enhanced_mag, mag_clean)
            g_loss = g_adv + g_mse
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

        print(f"    G_loss={total_g_loss/len(train_loader):.4f}, D_loss={total_d_loss/len(train_loader):.4f}")

        # --- Validation (just MSE) ---
        generator.eval()
        val_loss = 0
        with torch.no_grad():
            for mag_noisy, mag_clean, _, _ in tqdm(val_loader, desc="Valid"):
                mag_noisy = mag_noisy.to(device)
                mag_clean = mag_clean.to(device)
                lengths = torch.ones(mag_noisy.shape[0]).to(device)
                mask = generator(mag_noisy, lengths)
                enhanced_mag = mask * mag_noisy
                val_loss += mse_loss(enhanced_mag, mag_clean).item()
        print(f"    Validation MSE: {val_loss/len(val_loader):.4f}")

        # --- Save generator checkpoint ---
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f"metricgan_gen_epoch{epoch}.pth"))

    print("Training complete!")

if __name__ == "__main__":
    train_metricgan(
        train_csv="train.csv",
        val_csv="val.csv",
        epochs=10,
        batch_size=8,
        lr=1e-4,
        checkpoint_dir="checkpoints"
    )
