import torch.nn as nn

class EmotionClassifier(nn.Module):
    """
    A simple feedforward neural network for emotion classification based on facial landmarks.
    The network consists of two hidden layers with ReLU activations and dropout for regularization.

    The input dimension is set to 1434, which corresponds to the flattened facial landmarks,
    and the output dimension is set to 5, corresponding to the number of emotion classes.
    """
    def __init__(self, input_dim=1434, hidden_dim=512, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)