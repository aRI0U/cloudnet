import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_nc, output_nc, n_points):
        super(Net, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_points = n_points

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
