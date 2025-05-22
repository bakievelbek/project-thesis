import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import SpeechReconstructionDataset

class TDNN(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super(TDNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, dilation=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.output_layer = nn.Conv1d(128, output_dim, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, 1, n_mels, time)
        x = x.squeeze(1)  # (batch, n_mels, time)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        x = x.unsqueeze(1)  # (batch, 1, n_mels, time)
        return x


dataset = SpeechReconstructionDataset(csv_path='D:\\PyCharm projects\\project-thesis\\preprocess\\dataset_index.csv',
                                      segment_duration=3.0)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = TDNN()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0
    for corrupted, clean in dataloader:
        optimizer.zero_grad()
        outputs = model(corrupted)
        loss = loss_fn(outputs, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * corrupted.size(0)
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(dataset):.4f}")
