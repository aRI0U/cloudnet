import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

import torch_geometric.transforms as T
from torch_geometric.nn import PointConv, fps, radius
from torch_geometric.nn.conv import XConv

import util.geometry as geometry

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    @staticmethod
    def _batch_indicator(batch_size, npoint):
        # type: (int, int) -> torch.cuda.LongTensor
        r"""
            Returns a tensor whose each index i is the batch of the point i

            Parameters
            ----------
            batch_size: int
                Size of the batch
            npoint: int
                Number of points per point cloud

            Returns
            -------
            torch.cuda.LongTensor
                Tensor containing the batch id of all points

            >>> model = Net()
            >>> model._batch_indicator(4,3)
            tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        """
        z = torch.zeros(batch_size, npoint, dtype=torch.long)
        return (z + torch.arange(batch_size).unsqueeze(1)).view(-1).cuda()

    @staticmethod
    def _split_point_cloud(pc):
        # type: torch.cuda.FloatTensor -> (torch.cuda.FloatTensor, torch.cuda.FloatTensor)
        r"""
            Split point cloud in features and pos

            Parameters
            ----------
            input: torch.cuda.FloatTensor
                (...,d) tensor containing point clouds

            Returns
            -------
            torch.cuda.FloatTensor
                (...,3) tensor containing the coordinates of points
            torch.cuda.FloatTensor
                (...,d) tensor containing the whole point cloud
        """
        return pc[...,:3].contiguous(), pc

    def forward(self, input):
        # type: (Net, torch.cuda.FloatTensor) -> torch.cuda.FloatTensor
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.cuda.FloatTensor
                (B,N,d) tensor containing point clouds and features

            Returns
            -------
            torch.cuda.FloatTensor
                (B,7) tensor containing estimated configurations
        """
        raise NotImplementedError


class CloudNet(Net):
    r"""
        CloudNet: neural network inpired from PointNet++ used for relocalization.
        Implementation taken from Pytorch Geometric
    """
    def __init__(self):
        super(CloudNet, self).__init__()

        # set abstraction levels
        self.local_sa1 = PointConv(
            Seq(Lin(3, 64), ReLU(), Lin(64, 64), ReLU(), Lin(64, 128)))

        self.local_sa2 = PointConv(
            Seq(Lin(131, 128), ReLU(), Lin(128, 128), ReLU(), Lin(128, 256)))

        self.global_sa = Seq(
            Lin(259, 256), ReLU(), Lin(256, 512), ReLU(), Lin(512, 1024))

        # linear layers
        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 7)

    def forward(self, input):
        # pos, batch = data.pos, data.batch
        B, N, d = input.shape
        batch = self._batch_indicator(B, N)
        data = input.view(B*N,d)
        pos, _ = self._split_point_cloud(data)
        idx = fps(pos, batch, ratio=0.5)  # 512 points
        row, col = radius(
            pos, pos[idx], 0.1, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)  # Transpose.
        x = F.relu(self.local_sa1(None, (pos, pos[idx]), edge_index))
        pos, batch = pos[idx], batch[idx]

        idx = fps(pos, batch, ratio=0.25)  # 128 points
        row, col = radius(
            pos, pos[idx], 0.2, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)  # Transpose.
        x = F.relu(self.local_sa2(x, (pos, pos[idx]), edge_index))
        pos, batch = pos[idx], batch[idx]

        x = self.global_sa(torch.cat([x, pos], dim=1))
        x = x.view(-1, 128, self.lin1.in_features).max(dim=1)[0]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5)
        x = self.lin3(x)
        b_output = x.view(B, -1, 7)
        b_output = torch.mean(b_output, dim=1)
        n_output = torch.cat((b_output[...,:3], F.normalize(b_output[...,3:], p=2, dim=-1)), dim=-1)
        # b_output[...,3:] = F.normalize(b_output[...,3:], p=2, dim=-1)
        return n_output

class CloudCNN(Net):
    r"""
        CloudCNN: neural network inspired from PointCNN used for relocalization.
    """
    def __init__(self, input_nc, n_points):
        super(CloudCNN, self).__init__()
        self.xconv1 = XConv(input_nc, 16, 3, 4)
        self.lin1 = Lin(n_points, n_points//4)
        self.xconv2 = XConv(16, 32, 3, 4)
        self.lin2 = Lin(n_points//4, n_points//16)
        self.xconv3 = XConv(32, 7, 3, 4)
        self.lin3 = Lin(n_points//16, n_points//64)
        self.fc = Lin(n_points//64, 1)

    def forward(self, input):
        B, N, d = input.shape
        data = input.view(B*N,d)
        pos, features = self._split_point_cloud(data)

        x = self.xconv1(features, pos, batch=self._batch_indicator(B,N))
        x = self.lin1(x.view(B, -1, 16).transpose(1,2)) # TODO: see if all .transpose are necessary
        x = x.transpose(1,2).contiguous().view(-1,16)
        pos = self.lin1(pos.view(B, -1, 3).transpose(1,2))
        pos = pos.transpose(1,2).contiguous().view(-1,3)

        x = self.xconv2(x, pos, batch=self._batch_indicator(B,N//4))
        x = self.lin2(x.view(B, -1, 32).transpose(1,2))
        x = x.transpose(1,2).contiguous().view(-1,32)
        pos = self.lin2(pos.view(B, -1, 3).transpose(1,2))
        pos = pos.transpose(1,2).contiguous().view(-1,3)

        x = self.xconv3(x, pos, batch=self._batch_indicator(B,N//16))
        x = self.lin3(x.view(B, -1, 7).transpose(1,2))

        output = self.fc(x)
        b_output = output.view(B, 7)
        n_output = torch.cat((b_output[...,:3], F.normalize(b_output[...,3:], p=2, dim=-1)), dim=-1)
        return n_output

# if __name__ == '__main__':
