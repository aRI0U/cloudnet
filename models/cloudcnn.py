from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import XConv, fps, knn

from models.net import Net

class XConvolution(nn.Module):
    r"""
        CloudCNN: neural network inspired from PointCNN used for relocalization.
    """
    def __init__(self, C_in, C_mid1, C_mid2, C_out, n_points):
        super(XConvolution, self).__init__()
        self.C_mid1 = C_mid1
        self.C_out = C_out
        self.xconv = XConv(C_in, C_mid1, 3, 4, hidden_channels=max(C_in//4, 1))

        self.conv = nn.Conv2d(ceil(n_points/4), ceil(n_points/4), (4,1))

        self.fc = nn.Sequential(
            nn.Linear(C_mid1, C_mid2),
            nn.ReLU(),
            nn.Linear(C_mid2, C_out)
        )

    @staticmethod
    def _fps(B, N):
        L = ceil(N/4)
        row = torch.arange(L).expand((B,L))
        col = N*torch.arange(B).unsqueeze(1)
        return (row + col).view(-1).cuda()


    def forward(self, x, p, batch):
        B, N, d = x.shape
        pos = p.view(B*N, -1)
        features = x.view(B*N, -1)

        idx = fps(pos, batch, ratio=0.25, random_start=True)

        knn_idx = knn(pos, pos[idx], 4, batch_x=batch, batch_y=batch[idx])[1]

        x = self.xconv(features, pos, batch=batch)

        x = x[knn_idx].view(B, ceil(N/4), 4, self.C_mid1)
        x = self.fc(x)
        x = self.conv(x).view(B, ceil(N/4), self.C_out)
        return x, pos[idx]

class CloudCNN(Net):
    r"""
        CloudCNN: neural network inspired from PointCNN used for relocalization.
    """
    def __init__(self, input_nc, output_nc, n_points, use_gpu):
        super(CloudCNN, self).__init__(input_nc, output_nc, n_points, use_gpu)
        self.xconv1 = XConvolution(input_nc, 16, 24, 32, n_points)
        self.xconv2 = XConvolution(32, 64, 96, 128, ceil(n_points/4))
        self.xconv3 = XConvolution(128, 256, 384, 512, ceil(n_points/16))

        self.conv = nn.Conv1d(ceil(n_points/64), 1, 1)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, output_nc)
        )


    def forward(self, input):
        B, N, d = input.shape
        pos, features = self._split_point_cloud(input)
        # hierarchical X-convolutions
        x, p = self.xconv1(features, pos, self._batch_indicator(B, N, self.use_gpu))
        x, p = self.xconv2(x, p, self._batch_indicator(B, ceil(N/4), self.use_gpu))
        x, p = self.xconv3(x, p, self._batch_indicator(B, ceil(N/16), self.use_gpu))
        # merge into one output vector
        x = self.conv(x).view(B, 512)
        # assign to channels
        x = self.fc(x)
        # normalize quaterions
        output = torch.cat((x[...,:3], F.normalize(x[...,3:], p=2, dim=-1)), dim=-1)
        return output


if __name__ == '__main__':
    import torch
    net = CloudCNN(6, 4, 8)
    input = torch.tensor(
        [[[-1.,1,0],[0,1,0],[1,1,0],
        [-1,0,0],[0,0,0],[1,0,0],
        [-1,-1,0],[-1,1,0]]]).cuda()
    net(input)
