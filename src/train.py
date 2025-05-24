import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import SpeechReconstructionDataset
from model import TDNN
import datetime


def normalize(mel_spec):
    mean = mel_spec.mean()
    std = mel_spec.std()
    return (mel_spec - mean) / (std + 1e-9)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Загружаем датасеты ===
train_dataset = SpeechReconstructionDataset('train_index.csv', transform=normalize, segment_duration=3.0)
val_dataset = SpeechReconstructionDataset('val_index.csv', transform=normalize, segment_duration=3.0)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# === Модель, loss, optimizer ===
model = TDNN().to(device)
criterion = nn.MSELoss()  # можно комбинировать с L1 позже
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
best_val_loss = float('inf')

print("Start Training...")
print(f'Time: {datetime.datetime.now()}')
for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0
    for corrupted, clean in train_loader:
        corrupted = corrupted.to(device)
        clean = clean.to(device)
        optimizer.zero_grad()
        outputs = model(corrupted)
        loss = 0.7 * nn.MSELoss()(outputs, clean) + 0.3 * nn.L1Loss()(outputs, clean)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * corrupted.size(0)
    epoch_train_loss = running_loss / len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for corrupted, clean in val_loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            outputs = model(corrupted)
            loss = 0.7 * nn.MSELoss()(outputs, clean) + 0.3 * nn.L1Loss()(outputs, clean)
            val_loss += loss.item() * corrupted.size(0)
    epoch_val_loss = val_loss / len(val_dataset)

    print(f"Epoch {epoch + 1}/{epochs} Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # Save only if val_loss improved
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), f"../outputs/best_model_{epoch + 1}.pth")
        print("New best model saved.")

print("Training complete.")
print(f'Time: {datetime.datetime.now()}')
