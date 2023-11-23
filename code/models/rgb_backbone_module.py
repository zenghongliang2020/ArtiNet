import torch
import torch.nn as nn
import sys
import os


class CNN_Backbone(nn.Module):
    def __init__(self):
        super(CNN_Backbone, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 32 * 32, 256)

    def forward(self, img):
        features_conv1 = self.conv1(img)
        features_conv1_relu = self.relu1(features_conv1)
        features_conv1_pool = self.pool1(features_conv1_relu)

        features_conv2 = self.conv2(features_conv1_pool)
        features_conv2_relu = self.relu2(features_conv2)
        features_conv2_pool = self.pool2(features_conv2_relu)

        features_conv3 = self.conv3(features_conv2_pool)
        features_conv3_relu = self.relu3(features_conv3)
        features_conv3_pool = self.pool3(features_conv3_relu)
        features = self.flatten(features_conv3_pool)
        features = self.fc1(features)

        return features


if __name__=='__main__':
    img_backbone_net = CNN_Backbone().cuda()
    print(img_backbone_net)
    img_backbone_net.eval()
    out = img_backbone_net(torch.rand(16, 3, 256, 256).cuda())
    print(out.shape)