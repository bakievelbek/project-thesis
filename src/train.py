import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import SpeechReconstructionDataset
from model import TDNN


def normalize(mel_spec):
    mean = mel_spec.mean()
    std = mel_spec.std()
    return (mel_spec - mean) / (std + 1e-9)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SpeechReconstructionDataset(
        'D:\\PyCharm projects\\project-thesis\\preprocess\\dataset_index.csv',
        transform=normalize, segment_duration=3.0)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TDNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for corrupted, clean in dataloader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()
            outputs = model(corrupted)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * corrupted.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} Loss: {epoch_loss:.4f}")
        torch.save(model.state_dict(), f"outputs\\model_epoch_{epoch + 1}.pth")


if __name__ == '__main__':
    main()
