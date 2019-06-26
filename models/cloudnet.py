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

class CloudCNN(Net):
    r"""
        CloudCNN: neural network inspired from PointCNN used for relocalization.
    """
    def __init__(self, input_nc, output_nc, n_points):
        super(CloudCNN, self).__init__(input_nc, output_nc, n_points)
        assert n_points >= 64, 'Point clouds must have size >= 64. Please modify --n_points.'
        from torch_geometric.nn import XConv
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
        try:
            x = self.xconv1(features, pos, batch=self._batch_indicator(B,N))
            assert x.shape == (B*N, 16), 'Invalid shape for x: %s' % str(x.shape)
            x = self.lin1(x.view(B, -1, 16).transpose(1,2)) # TODO: see if all .transpose are necessary
            assert x.shape == (B, 16, N//4), 'Invalid shape for x: %s' % str(x.shape)
            x = x.transpose(1,2).contiguous().view(-1,16)
            assert x.shape == (B*N//4, 16), 'Invalid shape for x: %s' % str(x.shape)
            pos = self.lin1(pos.view(B, -1, 3).transpose(1,2))
            assert pos.shape == (B, 3, N//4), 'Invalid shape for x: %s' % str(x.shape)
            pos = pos.transpose(1,2).contiguous().view(-1,3)
            assert pos.shape == (B*N//4, 3), 'Invalid shape for x: %s' % str(x.shape)
            x = self.xconv2(x, pos, batch=self._batch_indicator(B,N//4))
            assert x.shape == (B*N//4, 32), 'Invalid shape for x: %s' % str(x.shape)
            x = self.lin2(x.view(B, -1, 32).transpose(1,2))
            assert x.shape == (B, 32, N//16), 'Invalid shape for x: %s' % str(x.shape)
            x = x.transpose(1,2).contiguous().view(-1,32)
            assert x.shape == (B*N//16, 32), 'Invalid shape for x: %s' % str(x.shape)
            pos = self.lin2(pos.view(B, -1, 3).transpose(1,2))
            pos = pos.transpose(1,2).contiguous().view(-1,3)
        except RuntimeError:
            pos, features = self._split_point_cloud(data)
            print(176)
            print(features.shape, pos.shape)
            print(pos, features)
            x = self.xconv1(features, pos, batch=self._batch_indicator(B,N))
            print(180)
            print(x.shape, pos.shape)
            print(x, pos)
            assert x.shape == (B*N, 16), 'Invalid shape for x: %s' % str(x.shape)
            x = self.lin1(x.view(B, -1, 16).transpose(1,2)) # TODO: see if all .transpose are necessary
            print(185)
            print(x.shape, pos.shape)
            print(x, pos)
            assert x.shape == (B, 16, N//4), 'Invalid shape for x: %s' % str(x.shape)
            x = x.transpose(1,2).contiguous().view(-1,16)
            print(190)
            print(x.shape, pos.shape)
            print(x, pos)
            assert x.shape == (B*N//4, 16), 'Invalid shape for x: %s' % str(x.shape)
            pos = self.lin1(pos.view(B, -1, 3).transpose(1,2))
            print(195)
            print(x.shape, pos.shape)
            print(x, pos)
            assert pos.shape == (B, 3, N//4), 'Invalid shape for x: %s' % str(x.shape)
            pos = pos.transpose(1,2).contiguous().view(-1,3)
            print(200)
            print(x.shape, pos.shape)
            print(x, pos)
            assert pos.shape == (B*N//4, 3), 'Invalid shape for x: %s' % str(x.shape)
            x = self.xconv2(x, pos, batch=self._batch_indicator(B,N//4))
            print(205)
            print(x.shape, pos.shape)
            print(x, pos)
            assert x.shape == (B*N//4, 32), 'Invalid shape for x: %s' % str(x.shape)
            x = self.lin2(x.view(B, -1, 32).transpose(1,2))
            print(210)
            print(x.shape, pos.shape)
            print(x, pos)
            assert x.shape == (B, 32, N//16), 'Invalid shape for x: %s' % str(x.shape)
            x = x.transpose(1,2).contiguous().view(-1,32)
            print(215)
            print(x.shape, pos.shape)
            print(x, pos)
            assert x.shape == (B*N//16, 32), 'Invalid shape for x: %s' % str(x.shape)
            pos = self.lin2(pos.view(B, -1, 3).transpose(1,2))
            print(220)
            print(x.shape, pos.shape)
            print(x, pos)
            pos = pos.transpose(1,2).contiguous().view(-1,3)
            print(224)
            print(x.shape, pos.shape)
            print(x, pos)
        x = self.xconv3(x, pos, batch=self._batch_indicator(B,N//16))
        x = self.lin3(x.view(B, -1, 7).transpose(1,2))

        output = self.fc(x)
        b_output = output.view(B, 7)
        n_output = torch.cat((b_output[...,:3], F.normalize(b_output[...,3:], p=2, dim=-1)), dim=-1)
        return n_output

# if __name__ == '__main__':
