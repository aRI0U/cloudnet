import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import XConv, fps, knn

from models.net import Net

class CloudCNN(Net):
    r"""
        CloudCNN: neural network inspired from PointCNN used for relocalization.
    """
    def __init__(self, input_nc, output_nc, n_points):
        super(CloudCNN, self).__init__(input_nc, output_nc, n_points)
        self.xconv1 = XConv(input_nc, 16, 3, 4)

        self.conv = nn.Conv2d(n_points//4, 1, (4,1))

        self.fc = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_nc)
        )


    def forward(self, input):
        B, N, d = input.shape
        batch = self._batch_indicator(B, N)
        data = input.view(B*N, d)
        pos, features = self._split_point_cloud(data)

        idx = fps(pos, batch, ratio=0.25)

        knn_idx = knn(pos, pos[idx], 4, batch_x=batch, batch_y=batch[idx])[1]

        x = self.xconv1(features, pos, batch=batch)

        x = x[knn_idx].view(B, N//4, 4, 16)
        x = self.fc(x)
        x = self.conv(x).view(B, self.output_nc)
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
