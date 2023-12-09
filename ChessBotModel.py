"""
    Luna-Chess Reinforcement Neural Network Architecture
    
    NOTE
    Neural Network
    - Inputs:
        . b - serialized board
    - Outputs:
        . vθ(s) - a scalar value of the board state ∈ [-1,1] from the perspective of the current player
        . →pθ(s) - a policy that is a probability vector over all possible actions.

    Training
    (st,→πt,zt), where:
        - st is the state
        - →πt is an estimate of the probability from state st
        - zt final game outcome ∈ [-1,1]

    Loss function:
    l=∑t(vθ(st)-zt)2-→πt⋅log(→pθ(st))
"""

from __future__ import annotations
import torch
from torch import nn
from torch import optim
from torch.nn import functional as fn

# Luna
from utils import dotdict


class ChessBotModel(nn.Module):
    """Reinforcement Learning Neural Network"""

    # Optimizer
    optimizer: optim.Optimizer

    # Learning Rate
    learning_rate: float

    # Action size
    action_size: int

    # HyperParameter args
    args: dotdict

    def __init__(self):
        super(ChessBotModel, self).__init__()

        self.board_x, self.board_y, self.board_z = (8, 8, 6)
        self.action_size = 64 * 64

        # Define neural net
        self.define_architecture()
        self.learning_rate = 1e-3
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def define_architecture(self) -> None:
        """Define Net
            - Input: serialized chess.Board
            - Output:
                - predicted board value (tanh)
                - policy distribution over possible moves (softmax)
        """
        # Args shortcut
        num_channels = 128

        # Input
        self.conv1 = nn.Conv3d(1, num_channels, 3, stride=1, padding=1)

        ## Hidden
        self.conv2 = nn.Conv3d(num_channels, num_channels * 2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(num_channels * 2, num_channels * 2, 3, stride=1)
        self.conv4 = nn.Conv3d(num_channels * 2, num_channels * 2, 3, stride=1)
        self.conv5 = nn.Conv3d(num_channels * 2, num_channels, 1, stride=1)

        self.bn1 = nn.BatchNorm3d(num_channels)
        self.bn2 = nn.BatchNorm3d(num_channels * 2)
        self.bn3 = nn.BatchNorm3d(num_channels * 2)
        self.bn4 = nn.BatchNorm3d(num_channels * 2)
        self.bn5 = nn.BatchNorm3d(num_channels)

        self.fc1 = nn.Linear(num_channels * (self.board_x - 4) * (self.board_y - 4) * (self.board_z - 4),
                             1024)  # 4096 -> 1024
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)

        # output p dist        
        self.fc4 = nn.Linear(512, self.action_size)

        # output scalar
        self.fc5 = nn.Linear(512, 1)

    def forward(self, boards_and_valids):
        """Forward prop"""
        x, valids = boards_and_valids
        num_channels = 128
        dropout = 0.3

        x = x.view(-1, 1, self.board_x, self.board_y, self.board_z)
        x = fn.relu(self.bn1(self.conv1(x)))
        x = fn.relu(self.bn2(self.conv2(x)))
        x = fn.relu(self.bn3(self.conv3(x)))
        x = fn.relu(self.bn4(self.conv4(x)))
        x = fn.relu(self.bn5(self.conv5(x)))
        x = x.view(-1, num_channels * (self.board_x - 4) * (self.board_y - 4) * (self.board_z - 4))
        x = fn.dropout(fn.relu(self.fc_bn1(self.fc1(x))), p=dropout, training=self.training)
        x = fn.dropout(fn.relu(self.fc_bn2(self.fc2(x))), p=dropout, training=self.training)
        x = fn.dropout(fn.relu(self.fc_bn3(self.fc3(x))), p=dropout, training=self.training)

        pi = self.fc4(x)
        v = self.fc5(x)

        pi -= (1 - valids) * 1000
        return fn.log_softmax(pi, dim=1), torch.tanh(v)
