"""A module for a mixture density network layer
For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math


ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)

class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, in_features//4),
            nn.ReLU(),
            nn.Linear(in_features//4, in_features//16),
            nn.ReLU(),
            nn.Linear(in_features//16, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            nn.ReLU(),
            nn.Linear(in_features//2, in_features//4),
            nn.ReLU(),
            nn.Linear(in_features//4, out_features*num_gaussians),
            nn.Softplus()
        )

        self.mu = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            nn.ReLU(),
            nn.Linear(in_features//2, in_features//4),
            nn.ReLU(),
            nn.Linear(in_features//4, out_features*num_gaussians)
        )


    def forward(self, input):
        pi = self.pi(input)
        sigma = self.sigma(input).view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(input).view(-1, self.num_gaussians, self.out_features)
        if not (mu[0,0,0] >= 0 or mu[0,0,0] < 0):
            print(minibatch)
            print(pi)
            print(sigma)
            print(mu)
            raise ValueError("NAN")
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(torch.clamp(-0.5 * ((target - mu) / sigma)**2, min=-20)) / sigma
    # print(ret)
    assert torch.all(ret <= 1)
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    # type: (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor) -> torch.FloatTensor
    r"""
        Calculates the error, given the MoG parameters and the target
        The loss is the negative log likelihood of the data given the MoG
        parameters.

        Parameters
        ----------
        pi: torch.FloatTensor
            (B, G) tensor containing the weights of the kernels
        sigma: torch.FloatTensor
            (B, G, n) tensor containing the standard deviation of ...
        mu: torch.FloatTensor
            (B, G, n) tensor containing the predicted values
        target: torch.FloatTensor
            (B, n) tensor containing the target

        Returns
        -------
        torch.FloatTensor
            the negative log-likelihood of the distribution
    """
    target = target.unsqueeze(1).expand_as(sigma)
    norms = ((mu-target)/sigma)**2
    # print(norms)
    norms = 0.5*torch.sum(norms, dim=2)
    values = torch.min(norms, dim=1).values
    exp = torch.exp(values.unsqueeze(1) - norms)
    weights = pi / torch.prod(sigma, dim=2)
    likelihood = torch.sum(weights*exp, dim=1)
    ll = torch.log(likelihood) - values
    # print('mdn')
    # print(mu[0])
    # print(norms[0])
    # print(weights[0])
    # print(weights*exp[0])
    # print(-torch.mean(ll))
    m = torch.mean(ll)
    if not (m >= 0 or m < 0):
        raise ValueError("NAN")
    return -torch.mean(ll)




def sample(pi, sigma, mu):
    """
        Draw samples from a MoG.
    """
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample

if __name__ == '__main__':
    import torch.nn.functional as F

    B, G, n = (8,5,3)
    pi = torch.randn(B, G)
    pi = F.softmax(pi, dim=1)
    sigma = torch.randn(B, G, n)
    sigma = torch.exp(sigma)*4
    mu = torch.randn(B, G, n)
    target = torch.randn(B, n)
    t = target.unsqueeze(1).expand_as(mu)
    lazy_loss = -torch.mean(torch.log(torch.sum(pi / torch.prod(sigma, dim=2) * torch.exp(-0.5*torch.sum(((mu-t)/sigma)**2, dim=2)), dim=1)))
    loss = mdn_loss(pi, sigma, mu, target)
    # print(((mu-t)/sigma)**2)
    print(torch.log(torch.sum(pi / torch.prod(sigma, dim=2) * torch.exp(-0.5*torch.sum(((mu-t)/sigma)**2, dim=2)), dim=1)))
    # print(torch.log(pi / torch.prod(sigma, dim=2) * torch.exp(-0.5*torch.sum(((mu-t)/sigma)**2, dim=2))))
    assert torch.eq(loss, lazy_loss), '%s\n%s' % (str(loss), str(lazy_loss))
