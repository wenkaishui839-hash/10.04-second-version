"""Partial block-circulant NICE model.

This model keeps a partial-replacement architecture and uses
``BlockCirculantLinear`` in the selected hidden layers.
"""

import operator

import torch
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base


class AdditiveCouplingBlock(nn.Module):
    """Coupling block with part of its hidden MLP replaced by block-circulant layers."""

    def __init__(
        self,
        n_features,
        n_hidden_layers,
        n_hidden_features,
        reverse,
        block_size=256,
    ):
        super().__init__()
        self.reverse = reverse
        half_features = n_features // 2
        net = [
            nn.Linear(in_features=half_features, out_features=n_hidden_features),
            nn.ReLU(),
        ]
        for i in range(n_hidden_layers - 1):
            if i * 2 < n_hidden_layers:
                net.append(
                    nn.Linear(
                        in_features=n_hidden_features,
                        out_features=n_hidden_features,
                    )
                )
            else:
                net.append(
                    pg_nn.BlockCirculantLinear(
                        in_features=n_hidden_features,
                        out_features=n_hidden_features,
                        block_size=block_size,
                    )
                )
            net.append(nn.ReLU())
        net.append(nn.Linear(in_features=n_hidden_features, out_features=half_features))
        self.net = nn.Sequential(*net)

    def _couple(self, x, op):
        c = x.shape[1]
        h1, h2 = x[:, : c // 2], x[:, c // 2 :]
        if self.reverse:
            h1 = op(h1, self.net(h2))
        else:
            h2 = op(h2, self.net(h1))
        return torch.cat((h1, h2), dim=1)

    def forward(self, x):
        return self._couple(x, operator.add)

    def inverse(self, y):
        return self._couple(y, operator.sub)


class ScalingLayer(nn.Module):
    """Invertible scaling layer representing a diagonal scaling matrix S."""

    def __init__(self, n_features):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros((1, n_features)))

    def log_det_J(self):
        return torch.sum(self.log_scale)

    def _couple(self, x, sign):
        original_shape = x.shape
        x = x.view(original_shape[0], -1)
        h = x * torch.exp(sign * self.log_scale)
        return h.view(original_shape)

    def forward(self, x):
        return self._couple(x, 1)

    def inverse(self, y):
        return self._couple(y, -1)


class PartialBlockCirculantNICE(base.GenerativeModel):
    """NICE model with partial hidden layers replaced by block-circulant layers."""

    def __init__(
        self,
        n_features,
        n_coupling_blocks=4,
        n_hidden_layers=5,
        n_hidden_features=1024,
        block_size=256,
    ):
        super().__init__()
        net = []
        reverse = False
        for _ in range(n_coupling_blocks):
            net.append(
                AdditiveCouplingBlock(
                    n_features=n_features,
                    n_hidden_layers=n_hidden_layers,
                    n_hidden_features=n_hidden_features,
                    reverse=reverse,
                    block_size=block_size,
                )
            )
            reverse = not reverse
        self.net = nn.Sequential(*net)
        self.scaling = ScalingLayer(n_features)

    def forward(self, x):
        return self._forward(x), self.scaling.log_det_J()

    @base.auto_reshape
    def _forward(self, x):
        y = self.net(x)
        y = self.scaling(y)
        return y

    def sample(self, n_samples, temp=1.0):
        x = torch.randn((n_samples, self._c, self._h, self._w)) * temp
        x = x.to(self.device)
        return self._inverse(x)

    @base.auto_reshape
    def _inverse(self, x):
        x = self.scaling.inverse(x)
        for block in reversed(self.net):
            x = block.inverse(x)
        return x


class PaddedPartialBlockCirculantNICE(PartialBlockCirculantNICE):
    """Partial block-circulant NICE with 784-dimensional inputs padded to 1024."""

    def __init__(
        self,
        n_features=784,
        padded_features=1024,
        n_coupling_blocks=4,
        n_hidden_layers=5,
        n_hidden_features=1024,
        block_size=256,
    ):
        if padded_features < n_features:
            raise ValueError("padded_features must be greater than or equal to n_features")
        self.original_features = int(n_features)
        self.padded_features = int(padded_features)
        super().__init__(
            n_features=self.padded_features,
            n_coupling_blocks=n_coupling_blocks,
            n_hidden_layers=n_hidden_layers,
            n_hidden_features=n_hidden_features,
            block_size=block_size,
        )

    def _pad_flattened_input(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        if x.shape[1] != self.original_features:
            raise ValueError(
                f"Expected {self.original_features} input features, got {x.shape[1]}"
            )
        if self.padded_features == self.original_features:
            return x
        padding = x.new_zeros(batch_size, self.padded_features - self.original_features)
        return torch.cat((x, padding), dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self._pad_flattened_input(x)
        y = self.net(x)
        y = self.scaling(y)
        side = int(self.padded_features**0.5)
        if side * side == self.padded_features:
            y = y.view(batch_size, 1, side, side)
        else:
            y = y.view(batch_size, 1, 1, self.padded_features)
        return y, self.scaling.log_det_J()

    def sample(self, n_samples, temp=1.0):
        x = torch.randn((n_samples, self.padded_features), device=self.device) * temp
        x = self.scaling.inverse(x)
        for block in reversed(self.net):
            x = block.inverse(x)
        x = x[:, : self.original_features]
        return x.view(n_samples, self._c, self._h, self._w)
