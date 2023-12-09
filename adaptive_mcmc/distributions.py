from abc import ABC, abstractmethod

import numpy as np
import numpy.random as rng
import torch
import torch.distributions as td
import torch.nn.functional as F
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from torch import nn


torchType = torch.float32


class Distribution(ABC):
    """
    Base class for a custom target distribution
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        self.torchType = torchType
        self.xlim, self.ylim = [-1, 1], [-1, 1]
        self.scale_2d_log_prob = 1
        # self.device_zero = torch.tensor(0., dtype=self.torchType, device=self.device)
        # self.device_one = torch.tensor(1., dtype=self.torchType, device=self.device)

    def prob(self, x):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        Output:
        density - p(x)
        """
        # You should define the class for your custom distribution
        return self.log_prob(x).exp()

    @abstractmethod
    def log_prob(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        log_density - log p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def energy(self, x):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        Output:
        energy = -log p(x)
        """
        # You should define the class for your custom distribution
        return -self.log_prob(x)

    def sample(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def __call__(self, x):
        return self.log_prob(x)

    def log_prob_2d_slice(self, z):
        raise NotImplementedError

    def plot_2d(self, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots()

        x = np.linspace(*self.xlim, 100)
        y = np.linspace(*self.ylim, 100)
        xx, yy = np.meshgrid(x, y)
        z = torch.FloatTensor(np.stack([xx, yy], -1))
        vals = (self.log_prob_2d_slice(z) / self.scale_2d_log_prob).exp()

        if ax is not None:
            ax.imshow(
                vals.flip(0),
                extent=[*self.xlim, *self.ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )
        else:
            plt.imshow(
                vals.flip(0),
                extent=[*self.xlim, *self.ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )

        return fig, self.xlim, self.ylim






class Funnel(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", "cpu")
        self.a = (kwargs.get("a", 1.0) * torch.ones(1)).to(self.device)
        self.b = kwargs.get("b", 0.5)
        self.dim = kwargs.get("dim", 16)
        self.distr1 = torch.distributions.Normal(torch.zeros(1).to(self.device), self.a)
        # self.distr2 = lambda z1: torch.distributions.MultivariateNormal(torch.zeros(self.dim-1), (2*self.b*z1).exp()*torch.eye(self.dim-1))
        # self.distr2 = lambda z1: -(z[...,1:]**2).sum(-1) * (-2*self.b*z1).exp() - np.log(self.dim) + 2*self.b*z1
        self.xlim = [-2, 10]
        self.ylim = [-30, 30]
        self.scale_2d_log_prob = 20  # 30.0

    def log_prob(self, z, x=None):
        logprob1 = self.distr1.log_prob(z[..., 0])
        z1 = z[..., 0]
        logprob2 = (
            -0.5 * (z[..., 1:] ** 2).sum(-1) * torch.exp(-2 * self.b * z1)
            - (self.dim - 1) * self.b * z1
        )
        return logprob1 + logprob2

    def log_prob_2d_slice(self, z, dim1=0, dim2=1):
        if dim1 == 0 or dim2 == 0:
            logprob1 = self.distr1.log_prob(z[..., 0])
            dim2 = dim2 if dim2 != 0 else dim1
            z1 = z[..., 0]
            # logprob2 = self.distr2(z[...,0])
            logprob2 = (
                -0.5 * (z[..., dim2] ** 2) * torch.exp(-2 * self.b * z1) - self.b * z1
            )
        # else:
        #     logprob2 = -(z[...,dim2]**2) * (-2*self.b*z1).exp() - np.log(self.dim) + 2*self.b*z1
        return logprob1 + logprob2

    def plot_2d_countour(self, ax):
        x = np.linspace(-15, 15, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        inp = torch.from_numpy(np.stack([X, Y], -1))
        Z = self.log_prob(inp.reshape(-1, 2)).reshape(inp.shape[:-1])

        # levels = np.quantile(Z, np.linspace(0.9, 0.99, 5))
        ax.contour(
            X,
            Y,
            Z.exp(),
            # levels = levels,
            levels=3,
            alpha=1.0,
            cmap="inferno",
        )

    def plot_2d(self, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots()

            xlim = [-2, 13]
            ylim = [-60, 60]
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        x = np.linspace(*xlim, 100)
        y = np.linspace(*ylim, 100)
        xx, yy = np.meshgrid(x, y)
        z = torch.FloatTensor(np.stack([xx, yy], -1))
        vals = (self.log_prob_2d_slice(z) / self.scale_2d_log_prob).exp()
        if ax is not None:
            ax.imshow(
                vals.flip(0),
                extent=[*xlim, *ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )
        else:
            plt.imshow(
                vals.flip(0),
                extent=[*xlim, *ylim],
                cmap="Greens",
                alpha=0.5,
                aspect="auto",
            )

        return fig, xlim, ylim


class HalfBanana(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Q = kwargs.get("Q", 0.01) * torch.ones(1)
        self.dim = kwargs.get("dim", 32)
        self.xlim = [-1, 9]
        self.ylim = [-2, 4]
        self.scale_2d_log_prob = 2.0
        # assert self.dim % 2 == 0, 'Dimension should be divisible by 2'

    def log_prob(self, z, x=None):
        # n = self.dim/2
        even = np.arange(0, self.dim, 2)
        odd = np.arange(1, self.dim, 2)

        ll = -((z[..., even] - z[..., odd] ** 2) ** 2) / self.Q - (z[..., odd] - 1) ** 2
        return ll.sum(-1)

    def log_prob_2d_slice(self, z, dim1=0, dim2=1):
        if dim1 % 2 == 0 and dim2 % 2 == 1:
            ll = (
                -((z[..., dim1] - z[..., dim2] ** 2) ** 2) / self.Q
                - (z[..., dim2] - 1) ** 2
            )
        return ll  # .sum(-1)


class Banana(Distribution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = kwargs.get("device", "cpu")
        self.b = kwargs.get("b", 0.02)  # * torch.ones(1).to(self.device)
        self.sigma = kwargs.get("sigma", 100.0)  # * torch.ones(1).to(self.device)
        self.dim = kwargs.get("dim", 32)
        self.xlim = [-1, 5]
        self.ylim = [-2, 2]
        self.scale_2d_log_prob = 2.0
        # assert self.dim % 2 == 0, 'Dimension should be divisible by 2'

    def log_prob(self, z, x=None):
        # n = self.dim/2
        even = np.arange(0, self.dim, 2)
        odd = np.arange(1, self.dim, 2)

        ll = -0.5 * (
            z[..., odd] - self.b * z[..., even] ** 2 + (self.sigma**2) * self.b
        ) ** 2 - ((z[..., even]) ** 2) / (2 * self.sigma**2)
        return ll.sum(-1)

    def log_prob_2d_slice(self, z, dim1=0, dim2=1):
        if dim1 % 2 == 0 and dim2 % 2 == 1:
            ll = (
                -((z[..., dim1] - z[..., dim2] ** 2) ** 2) / self.Q
                - (z[..., dim1] - 1) ** 2
            )
        return ll  # .sum(-1)


