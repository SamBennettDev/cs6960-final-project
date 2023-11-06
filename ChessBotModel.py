import torch
import torch.nn as nn

class ChessBotModel(nn.Module):
    def __init__(self):
        super(ChessBotModel, self).__init__()

        # Define the layers and architecture of your neural network
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)  # Output layer with move values

    def forward(self, board, legal_moves):
        x = self.conv1(board)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        return x
