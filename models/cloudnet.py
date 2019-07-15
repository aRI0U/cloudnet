import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
#
# import torch_geometric.transforms as T
from torch_geometric.nn import PointConv, fps, radius

from models.net import Net

class CloudNet(Net):
    r"""
        CloudNet: neural network inpired from PointNet++ used for relocalization.
        Implementation taken from Pytorch Geometric
    """
    def __init__(self, input_nc, output_nc, n_points):
        super(CloudNet, self).__init__(input_nc, output_nc, n_points)

        assert self.n_points > 1016, 'Point clouds must have size > 1016. Please change --n_points option.'
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
        self.lin3 = Lin(256, output_nc)

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
        b_output = x.view(B, -1, self.output_nc)
        b_output = torch.mean(b_output, dim=1)
        n_output = torch.cat((b_output[...,:3], F.normalize(b_output[...,3:], p=2, dim=-1)), dim=-1)
        # b_output[...,3:] = F.normalize(b_output[...,3:], p=2, dim=-1)
        return n_output
