import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from pytorch.FER2013Dataset import device


class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, 7)

    def forward(self, x):
        return self.mobilenet(x)


model = EmotionNet().to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
