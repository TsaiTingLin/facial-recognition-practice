import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn


# 定义你的 U-Net 模型
class UNet(nn.Module):
    def __init__(self, in_channels, start_neurons):
        super(UNet, self).__init__()

        # Down part
        self.conv1 = nn.Conv2d(in_channels, start_neurons, 3, padding=1)
        self.conv1_2 = nn.Conv2d(start_neurons, start_neurons, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.25)

        self.conv2 = nn.Conv2d(start_neurons, start_neurons * 2, 3, padding=1)
        self.conv2_2 = nn.Conv2d(start_neurons * 2, start_neurons * 2, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.5)

        self.conv3 = nn.Conv2d(start_neurons * 2, start_neurons * 4, 3, padding=1)
        self.conv3_2 = nn.Conv2d(start_neurons * 4, start_neurons * 4, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.5)

        self.conv4 = nn.Conv2d(start_neurons * 4, start_neurons * 8, 3, padding=1)
        self.conv4_2 = nn.Conv2d(start_neurons * 8, start_neurons * 8, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(0.5)

        # Middle
        self.convm1 = nn.Conv2d(start_neurons * 8, start_neurons * 16, 3, padding=1)
        self.convm2 = nn.Conv2d(start_neurons * 16, start_neurons * 16, 3, padding=1)

        # Up part
        self.deconv4 = nn.ConvTranspose2d(start_neurons * 16, start_neurons * 8, 2, stride=2)
        self.uconv4 = nn.Conv2d(start_neurons * 8 + start_neurons * 8, start_neurons * 8, 3, padding=1)
        self.uconv4_2 = nn.Conv2d(start_neurons * 8, start_neurons * 8, 3, padding=1)

        self.deconv3 = nn.ConvTranspose2d(start_neurons * 8, start_neurons * 4, 2, stride=2)
        self.uconv3 = nn.Conv2d(start_neurons * 4 + start_neurons * 4, start_neurons * 4, 3, padding=1)
        self.uconv3_2 = nn.Conv2d(start_neurons * 4, start_neurons * 4, 3, padding=1)

        self.deconv2 = nn.ConvTranspose2d(start_neurons * 4, start_neurons * 2, 2, stride=2)
        self.uconv2 = nn.Conv2d(start_neurons * 2 + start_neurons * 2, start_neurons * 2, 3, padding=1)
        self.uconv2_2 = nn.Conv2d(start_neurons * 2, start_neurons * 2, 3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(start_neurons * 2, start_neurons, 2, stride=2)
        self.uconv1 = nn.Conv2d(start_neurons + start_neurons, start_neurons, 3, padding=1)
        self.uconv1_2 = nn.Conv2d(start_neurons, start_neurons, 3, padding=1)

        self.output_layer = nn.Conv2d(start_neurons, 1, 1)

    def forward(self, x):
        # Down part
        conv1 = F.relu(self.conv1(x))
        conv1 = F.relu(self.conv1_2(conv1))
        pool1 = self.drop1(self.pool1(conv1))

        conv2 = F.relu(self.conv2(pool1))
        conv2 = F.relu(self.conv2_2(conv2))
        pool2 = self.drop2(self.pool2(conv2))

        conv3 = F.relu(self.conv3(pool2))
        conv3 = F.relu(self.conv3_2(conv3))
        pool3 = self.drop3(self.pool3(conv3))

        conv4 = F.relu(self.conv4(pool3))
        conv4 = F.relu(self.conv4_2(conv4))
        pool4 = self.drop4(self.pool4(conv4))

        # Middle
        convm = F.relu(self.convm1(pool4))
        convm = F.relu(self.convm2(convm))

        # Up part
        deconv4 = self.deconv4(convm)
        deconv4 = F.interpolate(deconv4, size=conv4.size()[2:], mode='bilinear', align_corners=True)
        uconv4 = torch.cat([deconv4, conv4], dim=1)
        uconv4 = F.dropout(uconv4, p=0.5, training=self.training)
        uconv4 = F.relu(self.uconv4(uconv4))
        uconv4 = F.relu(self.uconv4_2(uconv4))

        deconv3 = self.deconv3(uconv4)
        deconv3 = F.interpolate(deconv3, size=conv3.size()[2:], mode='bilinear', align_corners=True)
        uconv3 = torch.cat([deconv3, conv3], dim=1)
        uconv3 = F.dropout(uconv3, p=0.5, training=self.training)
        uconv3 = F.relu(self.uconv3(uconv3))
        uconv3 = F.relu(self.uconv3_2(uconv3))

        deconv2 = self.deconv2(uconv3)
        deconv2 = F.interpolate(deconv2, size=conv2.size()[2:], mode='bilinear', align_corners=True)
        uconv2 = torch.cat([deconv2, conv2], dim=1)
        uconv2 = F.dropout(uconv2, p=0.5, training=self.training)
        uconv2 = F.relu(self.uconv2(uconv2))
        uconv2 = F.relu(self.uconv2_2(uconv2))

        deconv1 = self.deconv1(uconv2)
        deconv1 = F.interpolate(deconv1, size=conv1.size()[2:], mode='bilinear', align_corners=True)
        uconv1 = torch.cat([deconv1, conv1], dim=1)
        uconv1 = F.dropout(uconv1, p=0.5, training=self.training)
        uconv1 = F.relu(self.uconv1(uconv1))
        uconv1 = F.relu(self.uconv1_2(uconv1))

        output_layer = torch.sigmoid(self.output_layer(uconv1))

        return output_layer
