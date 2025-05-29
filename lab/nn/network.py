import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_conditions):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Embedding(num_conditions, num_features)
        self.beta = nn.Embedding(num_conditions, num_features)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x, condition):
        out = self.bn(x)
        gamma = self.gamma(condition).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(condition).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta


class ChessEvaluationCNN(nn.Module):
    def __init__(self, num_piece_channels=13, num_classes=1, num_conditions=2):
        super(ChessEvaluationCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(num_piece_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Conditional batch normalization for active player
        self.cbn1 = ConditionalBatchNorm2d(64, num_conditions)
        self.cbn2 = ConditionalBatchNorm2d(128, num_conditions)
        self.cbn3 = ConditionalBatchNorm2d(256, num_conditions)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 1024)  # Adjusted from 256 * 8 * 8 to 256
        self.fc2 = nn.Linear(1024 + 1, num_classes)  # Extra input for half move count

    def forward(self, board_tensor, active_player, half_move_clock):
        # First convolution + conditional batch norm + ReLU
        x = self.conv1(board_tensor)
        x = self.cbn1(x, active_player)
        x = F.relu(x)

        # Second convolution + conditional batch norm + ReLU
        x = self.conv2(x)
        x = self.cbn2(x, active_player)
        x = F.relu(x)

        # Third convolution + conditional batch norm + ReLU
        x = self.conv3(x)
        x = self.cbn3(x, active_player)
        x = F.relu(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Reduce to (batch_size, 256, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 256)

        # Fully connected layer
        x = F.relu(self.fc1(x))  # Input to fc1 is now (batch_size, 256)

        half_move_clock = half_move_clock.float()

        # Concatenate half move clock and pass through the final fully connected layer
        x = torch.cat([x, half_move_clock.unsqueeze(1)], dim=1)
        output = self.fc2(x)

        return output
