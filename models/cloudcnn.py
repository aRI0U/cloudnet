from torch_geometric.nn import XConv

from models.net import Net

class CloudCNN(Net):
    r"""
        CloudCNN: neural network inspired from PointCNN used for relocalization.
    """
    def __init__(self, input_nc, output_nc, n_points):
        super(CloudCNN, self).__init__(input_nc, output_nc, n_points)
        self.xconv1 = XConv(input_nc, 16, 3, 4)
