"""Circulant linear layer implemented with FFT.

The layer stores one trainable vector and uses it as the first row of a circulant
matrix. For non-square projections, inputs are zero-padded to the circulant
dimension and outputs are sliced to ``out_features``.
"""

import math

import torch
from torch import nn


class CirculantLinear(nn.Module):
    """Linear-like layer backed by a circulant matrix.

    Args:
        in_features: Number of input features.
        out_features: Number of output features. If omitted, uses ``in_features``.
        bias: Whether to add a trainable bias.

    Notes:
        A true circulant matrix is square. To support NICE layers such as
        ``392 -> 1000`` and ``1000 -> 392``, this implementation creates a square
        circulant transform with ``max(in_features, out_features)`` dimensions,
        zero-pads shorter inputs, and slices the requested output width.
    """

    def __init__(self, in_features, out_features=None, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features or in_features)
        self.circulant_dim = max(self.in_features, self.out_features)

        bound = 1 / math.sqrt(self.in_features)
        self.weight = nn.Parameter(torch.empty(self.circulant_dim))
        nn.init.uniform_(self.weight, -bound, bound)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        if self.in_features < self.circulant_dim:
            pad_width = self.circulant_dim - self.in_features
            x = torch.nn.functional.pad(x, (0, pad_width))

        x_fft = torch.fft.rfft(x, n=self.circulant_dim, dim=-1)
        w_fft = torch.fft.rfft(self.weight, n=self.circulant_dim, dim=-1)
        y = torch.fft.irfft(x_fft * w_fft, n=self.circulant_dim, dim=-1)
        y = y[..., : self.out_features]

        if self.bias is not None:
            y = y + self.bias

        return y.reshape(*original_shape[:-1], self.out_features)


class BlockCirculantLinear(nn.Module):
    """Linear-like layer backed by a block-circulant matrix.

    The dense ``out_features x in_features`` weight is divided into
    ``n_block_rows x n_block_cols`` circulant blocks. Each block is represented
    by one trainable vector of length ``block_size`` and multiplication is done
    with FFTs, matching the block-circulant algorithm from the notebook export.
    """

    def __init__(self, in_features, out_features=None, bias=True, block_size=256):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features or in_features)
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

        self.n_block_cols = math.ceil(self.in_features / self.block_size)
        self.n_block_rows = math.ceil(self.out_features / self.block_size)
        self.padded_in_features = self.n_block_cols * self.block_size
        self.padded_out_features = self.n_block_rows * self.block_size

        bound = 1 / math.sqrt(self.in_features)
        self.weight = nn.Parameter(
            torch.empty(self.n_block_rows, self.n_block_cols, self.block_size)
        )
        nn.init.uniform_(self.weight, -bound, bound)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_size={self.block_size}, "
            f"blocks={self.n_block_rows}x{self.n_block_cols}, "
            f"bias={self.bias is not None}"
        )

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        if self.in_features < self.padded_in_features:
            pad_width = self.padded_in_features - self.in_features
            x = torch.nn.functional.pad(x, (0, pad_width))

        x_blocks = x.view(-1, self.n_block_cols, self.block_size)
        x_fft = torch.fft.rfft(x_blocks, n=self.block_size, dim=-1)
        w_fft = torch.fft.rfft(self.weight, n=self.block_size, dim=-1)

        y_fft = (x_fft[:, None, :, :] * w_fft[None, :, :, :]).sum(dim=2)
        y = torch.fft.irfft(y_fft, n=self.block_size, dim=-1)
        y = y.reshape(-1, self.padded_out_features)[..., : self.out_features]

        if self.bias is not None:
            y = y + self.bias

        return y.reshape(*original_shape[:-1], self.out_features)


def manual_circular_convolution(w, x):
    """Reference implementation for small square test cases."""
    n = w.shape[0]
    y = torch.zeros_like(x)
    for i in range(n):
        for j in range(n):
            y[..., i] = y[..., i] + w[j] * x[..., (i - j) % n]
    return y


def test_circulant_linear():
    torch.manual_seed(42)

    for in_features, out_features in [(8, 8), (5, 9), (9, 5)]:
        layer = CirculantLinear(in_features, out_features, bias=True)
        x = torch.randn(4, in_features)
        y = layer(x)
        assert y.shape == (4, out_features)

    layer = CirculantLinear(5, bias=False)
    x = torch.randn(3, 5)
    y = layer(x)
    y_ref = manual_circular_convolution(layer.weight, x)
    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)

    layer = CirculantLinear(8, 6, bias=True)
    x = torch.randn(2, 8, requires_grad=True)
    loss = layer(x).sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert x.grad is not None

    print("CirculantLinear tests passed.")


if __name__ == "__main__":
    test_circulant_linear()
