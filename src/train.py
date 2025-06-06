# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SpeechEnhancementDataset
from model import HybridGenerator, MetricDiscriminator

from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
SEGMENT_LENGTH = 3.0  # seconds, as per your new chunk length
SAMPLE_RATE = 16000

def train_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer, recon_loss_fn, device, epoch, print_every=10):
    generator.train()
    discriminator.train()
    total_g_loss = 0
    total_d_loss = 0

    for batch_idx, (noisy, clean) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        noisy = noisy.to(device)
        clean = clean.to(device)

        # ------- Train Discriminator -------
        d_optimizer.zero_grad()
        enhanced = generator(noisy)
        real_input = torch.stack([clean, clean], dim=1).permute(0, 1, 3, 2)
        fake_input = torch.stack([clean, enhanced.detach()], dim=1).permute(0, 1, 3, 2)
        real_scores = discriminator(real_input)
        fake_scores = discriminator(fake_input)
        d_loss = torch.mean(torch.relu(1.0 - real_scores)) + torch.mean(torch.relu(1.0 + fake_scores))
        d_loss.backward()
        d_optimizer.step()
        total_d_loss += d_loss.item()

        # ------- Train Generator -------
        g_optimizer.zero_grad()
        enhanced = generator(noisy)
        fake_input = torch.stack([clean, enhanced], dim=1).permute(0, 1, 3, 2)
        fake_scores = discriminator(fake_input)
        adv_loss = -torch.mean(fake_scores)
        recon_loss = recon_loss_fn(enhanced, clean)
        g_loss = adv_loss + 0.1 * recon_loss
        g_loss.backward()
        g_optimizer.step()
        total_g_loss += g_loss.item()

        if batch_idx % print_every == 0:
            tqdm.write(f"Batch {batch_idx}/{len(train_loader)} | G_loss: {g_loss.item():.4f} | D_loss: {d_loss.item():.4f}")

    return total_g_loss / len(train_loader), total_d_loss / len(train_loader)

def validate(generator, val_loader, recon_loss_fn, device):
    generator.eval()
    total_loss = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            enhanced = generator(noisy)
            loss = recon_loss_fn(enhanced, clean)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    # Device selection for Apple Silicon/M1
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # DataLoader: Use num_workers=0 for Mac (change if you want)
    train_dataset = SpeechEnhancementDataset('train_index.csv', segment_length=SEGMENT_LENGTH, sample_rate=SAMPLE_RATE, augment=True)
    val_dataset = SpeechEnhancementDataset('val_index.csv', segment_length=SEGMENT_LENGTH, sample_rate=SAMPLE_RATE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Models
    generator = HybridGenerator().to(device)
    discriminator = MetricDiscriminator().to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # Loss
    recon_loss_fn = nn.L1Loss()

    # Training loop
    for epoch in range(EPOCHS):
        train_g_loss, train_d_loss = train_epoch(
            generator, discriminator, train_loader,
            g_optimizer, d_optimizer, recon_loss_fn,
            device, epoch, print_every=10
        )
        val_loss = validate(generator, val_loader, recon_loss_fn, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | G_loss: {train_g_loss:.4f} | D_loss: {train_d_loss:.4f} | Val L1: {val_loss:.4f} ")

        torch.save(generator.state_dict(), f"generator_epoch{epoch+1}.pt")
        torch.save(discriminator.state_dict(), f"discriminator_epoch{epoch+1}.pt")

if __name__ == "__main__":
    main()
