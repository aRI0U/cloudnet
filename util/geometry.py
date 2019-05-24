from math import cos, sin, sqrt, pi
import time
import unittest

import torch

def normalize(Y):
    Y[...,3:] = Y[...,3:] / torch.norm(Y[...,3:],p=2,dim=1).unsqueeze(1)


def euler_to_quaternion(theta, phi, psi):
    # type: (float, float, float) -> torch.FloatTensor
    r"""
        converts an orientation represented by Euler angles into a unitary quaternion
    """
    return torch.Tensor([
        cos(phi)*cos(theta)*cos(psi) + sin(phi)*sin(theta)*sin(psi),
        sin(phi)*cos(theta)*cos(psi) - cos(phi)*sin(theta)*sin(psi),
        cos(phi)*sin(theta)*cos(psi) + sin(phi)*cos(theta)*sin(psi),
        cos(phi)*cos(theta)*sin(psi) - sin(phi)*sin(theta)*cos(psi),
    ])

def _quaternion_to_matrix(Q):
    # type: (torch.FloatTensor) -> torch.FloatTensor
    r"""
        Converts a tensor of UNITARY quaternions into a tensor or rotation matrices

        Parameters
        ----------
        Q: torch.FloatTensor
            (B,4) tensor of quaternions

        Returns
        -------
        torch.FloatTensor
            (B,3,3) tensor of rotation matrices
    """
    qr, qi, qj, qk = Q[:,0], Q[:,1], Q[:,2], Q[:,3]

    return torch.stack([
        1 - 2*(qj**2 + qk**2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr),
        2*(qi*qj + qk*qr), 1 - 2*(qi**2 + qk**2), 2*(qj*qk - qi*qr),
        2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1 - 2*(qi**2 + qj**2)
    ]).transpose(0,1).reshape(-1,3,3)

def geometric_loss(input_X, input_Y, pred_Y, p=2):
    # type: (torch.cuda.FloatTensor, torch.cuda.FloatTensor, int) -> torch.cuda.FloatTensor
    r"""
        Geometric loss between two configurations on point cloud [input_X]

        Parameters
        ----------
        input_X: torch.cuda.FloatTensor
            (B,3,N) tensor containing the point clouds
        input_Y: torch.cuda.FloatTensor
            (B,7) tensor containing the actual configurations
        pred_Y: torch.cuda.FloatTensor
            (B,7) tensor containing the predicted configurations
        p: int = 2
            Norm used for the computation of the loss

        Returns
        -------
        torch.cuda.FloatTensor
            (1) tensor containing the geometric loss
    """
    batch_size = len(input_X)
    input_pos, pred_pos = input_Y[:,:3], pred_Y[:,:3]
    input_ori, pred_ori = input_Y[:,3:], pred_Y[:,3:]
    input_mat = _quaternion_to_matrix(input_ori).cuda()
    pred_mat = _quaternion_to_matrix(pred_ori).cuda()
    # computing estimated global coords of points
    # print(input_mat.shape, input_X.shape, input_pos.unsqueeze(2).shape)
    input_glob = input_mat @ input_X + input_pos.unsqueeze(2)
    pred_glob = pred_mat @ input_X + pred_pos.unsqueeze(2)
    diff = input_glob - pred_glob
    return torch.mean(torch.norm(diff, p=p, dim=1))

def geometric_loss_no_cuda(input_X, input_Y, pred_Y, p=2):
    batch_size = len(input_X)
    input_pos, pred_pos = input_Y[:,:3], pred_Y[:,:3]
    input_ori, pred_ori = input_Y[:,3:], pred_Y[:,3:]
    input_mat = _quaternion_to_matrix(input_ori)
    pred_mat = _quaternion_to_matrix(pred_ori)
    # computing estimated global coords of points
    input_glob = input_mat @ input_X + input_pos.unsqueeze(2)
    pred_glob = pred_mat @ input_X + pred_pos.unsqueeze(2)
    diff = input_glob - pred_glob
    return torch.mean(torch.norm(diff, p=p, dim=1))

## tests on loss function
class TestLossFunction(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLossFunction, self).__init__(*args, **kwargs)
        self.batch_size = 16
        self.X1 = torch.randn(self.batch_size,3,50)
        self.X2 = torch.randn(self.batch_size,3,50)
        self.Y1 = torch.randn(self.batch_size,7)
        normalize(self.Y1)
        self.Y2 = torch.randn(self.batch_size,7)
        normalize(self.Y2)
        self.loss = geometric_loss_no_cuda

    # tests
    def test_separation(self):
        r"""
            Verifies that loss(X,Y1,Y2) = 0 <=> Y1 = Y2
        """
        self.assertEqual(self.loss(self.X1,self.Y1,self.Y1), 0)
        while torch.all(self.Y1 == self.Y2):
            self.Y2 = torch.randn(self.batch_size,7)
            normalize(self.Y2)
        self.assertNotEqual(self.loss(self.X1,self.Y1,self.Y2), 0)

    def test_symmetry(self):
        r"""
            Verifies that loss(X,Y1,Y2) = loss(X,Y2,Y1)
        """
        self.assertEqual(self.loss(self.X1,self.Y1,self.Y2).item(), self.loss(self.X1,self.Y2,self.Y1).item())

    def test_unordered_set(self):
        r"""
            Verifies that loss(X,Y1,Y2) = loss(sigma(X),Y1,Y2), where sigma is a
            permutation of the points of X
        """
        shuffled = self.X1[:,:,torch.randperm(50)]
        self.assertEqual(self.loss(self.X1,self.Y1,self.Y2).item(), self.loss(shuffled, self.Y1,self.Y2).item())

    def test_translation_invariance(self):
        # loss(X,Y1,Y2) = loss(X,Y1+t,Y2+t)
        X1 = self.X1
        Y1 = self.Y1
        Y2 = self.Y2
        t = torch.randn(3)
        Y1t = Y1.clone().detach()
        Y1t[:,:3] = Y1t[:,:3] + t
        Y2t = Y2.clone().detach()
        Y2t[:,:3] = Y2t[:,:3] + t
        self.assertEqual(self.loss(X1,Y1,Y2).item(), self.loss(X1,Y1t,Y2t).item())


if __name__ == '__main__':
    unittest.main()
