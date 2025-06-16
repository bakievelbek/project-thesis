# train.py
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
from dataset import SpeechPairsDataset
from model import EnhancementGenerator, MetricDiscriminator
import os
import time

# --------- CONFIG -------------
CSV_PATH = "train.csv"
SAVE_DIR = "checkpoints_"
BATCH_SIZE = 8
LR = 1e-4
NUM_EPOCHS = 30
SAMPLE_RATE = 16000
SEGMENT_LENGTH = 3.0
PRINT_FREQ = 10  # Log every N batches

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ------- DATASET & LOADER -------
dataset = SpeechPairsDataset(CSV_PATH, sample_rate=SAMPLE_RATE, segment_length=SEGMENT_LENGTH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f"Total samples: {len(dataset)} | Batches per epoch: {len(loader)}")

# ------- MODELS -----------------
generator = EnhancementGenerator().to(DEVICE)
discriminator = MetricDiscriminator().to(DEVICE)
print(f"Generator params: {sum(p.numel() for p in generator.parameters())}")
print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters())}")

# ------- OPTIMIZERS -------------
optim_g = torch.optim.Adam(generator.parameters(), lr=LR)
optim_d = torch.optim.Adam(discriminator.parameters(), lr=LR)

# ------- LOSSES -----------------
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()

# ------- STFT -------------------
stft = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128, power=None).to(DEVICE)

def mag(x):
    spec = stft(x)
    return spec.abs()

def train_epoch(epoch):
    generator.train()
    discriminator.train()
    total_g_loss, total_d_loss = 0.0, 0.0

    start_time = time.time()
    for i, batch in enumerate(loader):
        noisy = batch['noisy'].to(DEVICE)  # [B, T]
        clean = batch['clean'].to(DEVICE)  # [B, T]

        # Debug info on first batch
        if i == 0:
            print(f"Epoch {epoch} | Batch shape: noisy {noisy.shape}, clean {clean.shape}")
            print(f"Sample values (noisy): {noisy[0][:10].cpu().numpy()}")

        # Convert to STFT mag
        noisy_mag = mag(noisy)
        clean_mag = mag(clean)
        lengths = torch.full((noisy_mag.shape[0],), noisy_mag.shape[2], dtype=torch.long, device=DEVICE)

        # -- Generator forward --
        mask = generator(noisy_mag.permute(0,2,1), lengths)
        enhanced_mag = noisy_mag.permute(0,2,1) * mask  # shape: [B, T, F]

        # -- Discriminator forward --
        d_in_fake = torch.stack([enhanced_mag, clean_mag.permute(0,2,1)], dim=1)
        d_in_real = torch.stack([clean_mag.permute(0,2,1), clean_mag.permute(0,2,1)], dim=1)

        pred_fake = discriminator(d_in_fake).squeeze(-1)
        pred_real = discriminator(d_in_real).squeeze(-1)

        # -- Discriminator loss --
        loss_d = bce(pred_real, torch.ones_like(pred_real)) + bce(pred_fake, torch.zeros_like(pred_fake))

        optim_d.zero_grad()
        loss_d.backward(retain_graph=True)
        optim_d.step()

        # -- Generator loss (adversarial + L1 mag) --
        pred_fake_for_g = discriminator(d_in_fake).squeeze(-1)
        loss_g_adv = bce(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
        loss_g_l1 = mse(enhanced_mag, clean_mag.permute(0,2,1))
        loss_g = loss_g_adv + 0.5 * loss_g_l1

        optim_g.zero_grad()
        loss_g.backward()
        optim_g.step()

        total_g_loss += loss_g.item()
        total_d_loss += loss_d.item()

        # ---- LOGGING ----
        if (i+1) % PRINT_FREQ == 0 or (i+1) == len(loader):
            elapsed = time.time() - start_time
            eta = (elapsed / (i+1)) * (len(loader)-(i+1))
            print(
                f"[Epoch {epoch}] Batch {i+1}/{len(loader)} | "
                f"G_loss: {loss_g.item():.4f} | D_loss: {loss_d.item():.4f} | "
                f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
            )
            print(f"  Learning rate: {optim_g.param_groups[0]['lr']:.6f}")
            print(f"  Example noisy path: {dataset.df.iloc[i]['noisy']}")

    avg_g = total_g_loss / len(loader)
    avg_d = total_d_loss / len(loader)
    print(f"=== EPOCH {epoch} SUMMARY ===")
    print(f"Avg Generator Loss: {avg_g:.4f} | Avg Discriminator Loss: {avg_d:.4f}")
    print(f"Epoch took {time.time() - start_time:.1f} seconds")
    return avg_g, avg_d

def main():
    # -------- TRAIN LOOP -------------
    os.makedirs(SAVE_DIR, exist_ok=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        g_loss, d_loss = train_epoch(epoch)
        torch.save(generator.state_dict(), f"{SAVE_DIR}/generator_epoch{epoch}.pth")
        torch.save(discriminator.state_dict(), f"{SAVE_DIR}/discriminator_epoch{epoch}.pth")
        print(f"[Checkpoint] Saved after epoch {epoch}")

    print("Training finished!")


if __name__ == "__main__":
    main()