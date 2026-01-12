import torch.nn as nn

class MLPClassifier(nn.Module):
  def __init__(self, input_dim, num_classes):
    super().__init__()
    self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

  def forward(self, x):
    return self.net(x)
