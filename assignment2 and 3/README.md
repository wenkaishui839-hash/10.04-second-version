# 大创任务二 / 任务三：Block-Circulant NICE 架构重构与训练

> 本目录基于上游 [pytorch-generative](https://github.com/EugenHotaj/pytorch-generative) 修改，
> 原始上游文档见文末。

## 本任务做了什么

1. **循环层算子**（`pytorch_generative/nn/circulant_linear.py`）：新增
   `CirculantLinear` 与 `BlockCirculantLinear`——把全连接层替换为循环/分块循环矩阵，
   用 `torch.fft.rfft/irfft` 做 O(N log N) 循环卷积。
2. **模型改造**（`pytorch_generative/models/flow/nice.py`）：新增
   `CirculantNICE` / `PaddedNICE` / **`PaddedBlockCirculantNICE`**（最终部署模型）——
   784 维 MNIST 输入补零对齐至 1024 维，契合基-2 FFT 的 256 分块要求；另见
   `flow/block_circ_nice.py` 的部分替换实验模型。
3. **训练与导出**（`test_dachuang/run_nice_experiment.py`）：统一实验脚本支持
   `original / circulant / partial-block-circ / padded-original / padded-block-circ`
   等模型类型，训练后自动导出权重 `.txt/.pt` 与 metrics.json、样例图。

## 训练产物归档（runs/）

| 目录 | 模型 | 参数量 | 训练耗时（50 epoch） | 用途 |
|------|------|--------|---------------------|------|
| `padded_block_circ_nice_e50/` | **PaddedBlockCirculantNICE（部署模型，1024 维）** | **105,472** | 456.6 s | 导出 `.bin` 供任务四 C/RVV 推理；校验和 1880.16069 已复核 |
| `padded_original_nice_e50/` | PaddedNICE（padded 密集基线，1024 维） | 20,995,072 | 481.9 s | 压缩率对照（99.50%） |
| `original_nice_e50/` | NICE（原版 784 维） | 19,158,352 | 671.6 s | 训练加速对照（34%） |
| `block_circulant_nice_e50/e150` | CirculantNICE（784 维，104,272 参数） | 104,272 | 443.0 s | 训练加速对照（34%） |
| `original_nice_e150/` | NICE e150 | 19,158,352 | 1559.9 s | 长程对照 |

## 关键指标口径（已复核）

- **参数压缩率**：105,472 / 20,995,072（padded 基线）= **99.50%**；
  按权重 .bin 字节计 = 99.44%（426,480 B vs 76.6 MB 非 padded 原版）。
- **训练加速 34%**：`original_nice_e50`（671.6 s）→ `block_circulant_nice_e50`
  （443.0 s），同 50 epoch、同批大小，数据来自各自 metrics.json。
- **部署权重校验和**：`padded_block_circ_nice_e50/` 导出的 `circulant_weights.bin`
  校验和 1880.16069，与任务四 `load_weights_test` 报告逐位一致（2026-09 复核）。

---

# pytorch-generative

`pytorch-generative` is a Python library which makes generative modeling in PyTorch easier by providing:

* high quality reference implementations of SOTA generative [models](https://github.com/EugenHotaj/pytorch-generative/tree/master/pytorch_generative/models) 
* useful abstractions of common [building blocks](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/nn) found in the literature
* utilities for [training](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/trainer.py), [debugging](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/debug.py), and working with [Google Colab](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/colab_utils.py)
* integration with TensorBoard for easy metrics visualization

To get started, click on one of the links below.
* [Installation](#installation)
* [Reproducing Results](#reproducing-results)
* [Google Colab](#google-colab)
* [Example - ImageGPT](#example---imagegpt)
* [Supported Algorithms](#supported-algorithms) 
  * [Autoregressive Models](#autoregressive-models)
  * [Variational Autoencoders](#variational-autoencoders)
  * [Normalizing Flows](#normalizing-flows)
  * [Miscellaneous](#miscellaneous)

## Installation

To install `pytorch-generative`, clone the repository and install the requirements:

```
git clone https://www.github.com/EugenHotaj/pytorch-generative
cd pytorch-generative
pip install -r requirements.txt
```

After installation, run the tests to sanity check that everything works:

```
python -m unittest discover
```

## Reproducing Results

All our [models](https://github.com/EugenHotaj/pytorch-generative/tree/master/pytorch_generative/models) implement a `reproduce` function with all the hyperparameters necessary to reproduce the results listed in the [supported algorithms](#supported-algorithms) section. This makes it very easy to reproduce any results using our [training script](https://github.com/EugenHotaj/pytorch-generative/tree/master/train.py), for example:

```
python train.py --model image_gpt --logdir /tmp/run --use-cuda
```

Training metrics will periodically be logged to TensorBoard for easy visualization. To view these metrics, launch a local TensorBoard server:

```
tensorboard --logdir /tmp/run
```

To run the model on a different dataset, with different hyperparameters, etc, simply modify its `reproduce` function and rerun the commands above.

## Google Colab

To use `pytorch-generative` in Google Colab, clone the repository and move it into the top-level directory:

```
!git clone https://www.github.com/EugenHotaj/pytorch-generative
!mv pytorch-generative/pytorch_generative pytorch-generative
```

You can then import `pytorch-generative` like any other library:

```python
import pytorch_generative as pg_nn
from pytorch_generative import models
...
```

## Example - ImageGPT

Supported models are implemented as PyTorch Modules and are easy to use:

```python
from pytorch_generative import models

... # Data loading code.

model = models.ImageGPT(in_channels=1, out_channels=1, in_size=28)
model(batch)
```

Alternatively, lower level building blocks in [pytorch_generative.nn](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/nn) can be used to write models from scratch. We show how to implement a convolutional [ImageGPT](https://openai.com/blog/image-gpt/) model below:

```python

from torch import nn

from pytorch_generative import nn as pg_nn


class TransformerBlock(nn.Module):
  """An ImageGPT Transformer block."""

  def __init__(self, 
               n_channels, 
               n_attention_heads):
    """Initializes a new TransformerBlock instance.
    
    Args:
      n_channels: The number of input and output channels.
      n_attention_heads: The number of attention heads to use.
    """
    super().__init__()
    self._ln1 = pg_nn.NCHWLayerNorm(n_channels)
    self._ln2 = pg_nn.NCHWLayerNorm(n_channels)
    self._attn = pg_nn.CausalAttention(
        in_channels=n_channels,
        embed_channels=n_channels,
        out_channels=n_channels,
        n_heads=n_attention_heads,
        mask_center=False)
    self._out = nn.Sequential(
        nn.Conv2d(
            in_channels=n_channels, 
            out_channels=4*n_channels, 
            kernel_size=1),
        nn.GELU(),
        nn.Conv2d(
            in_channels=4*n_channels, 
            out_channels=n_channels, 
            kernel_size=1))

  def forward(self, x):
    x = x + self._attn(self._ln1(x))
    return x + self._out(self._ln2(x))


class ImageGPT(nn.Module):
  """The ImageGPT Model."""
  
  def __init__(self,       
               in_channels,
               out_channels,
               in_size,
               n_transformer_blocks=8,
               n_attention_heads=4,
               n_embedding_channels=16):
    """Initializes a new ImageGPT instance.
    
    Args:
      in_channels: The number of input channels.
      out_channels: The number of output channels.
      in_size: Size of the input images. Used to create positional encodings.
      n_transformer_blocks: Number of TransformerBlocks to use.
      n_attention_heads: Number of attention heads to use.
      n_embedding_channels: Number of attention embedding channels to use.
    """
    super().__init__()
    self._pos = nn.Parameter(torch.zeros(1, in_channels, in_size, in_size))
    self._input = pg_nn.CausalConv2d(
        mask_center=True,
        in_channels=in_channels,
        out_channels=n_embedding_channels,
        kernel_size=3,
        padding=1)
    self._transformer = nn.Sequential(
        *[TransformerBlock(n_channels=n_embedding_channels,
                         n_attention_heads=n_attention_heads)
          for _ in range(n_transformer_blocks)])
    self._ln = pg_nn.NCHWLayerNorm(n_embedding_channels)
    self._out = nn.Conv2d(in_channels=n_embedding_channels,
                          out_channels=out_channels,
                          kernel_size=1)

  def forward(self, x):
    x = self._input(x + self._pos)
    x = self._transformer(x)
    x = self._ln(x)
    return self._out(x)
```

## Supported Algorithms

 `pytorch-generative` supports the following algorithms. 

We train likelihood based models on dynamically [Binarized MNIST](https://paperswithcode.com/sota/image-generation-on-binarized-mnist) and report the log likelihood in the tables below.

### Autoregressive Models

| Algorithm | [Binarized MNIST](https://paperswithcode.com/sota/image-generation-on-binarized-mnist) (nats) | Links |
| --- | ---| --- |
| PixelSNAIL | **78.61** | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/autoregressive/pixel_snail.py), [Paper](http://proceedings.mlr.press/v80/chen18h/chen18h.pdf) |
| ImageGPT | 79.17 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/autoregressive/image_gpt.py), [Paper](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)|
| Gated PixelCNN | 81.50 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/autoregressive/gated_pixel_cnn.py), [Paper](https://arxiv.org/abs/1606.05328) |
| PixelCNN | 81.45 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/autoregressive/pixel_cnn.py), [Paper](https://arxiv.org/abs/1601.06759) |
| MADE | 84.87 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/autoregressive/made.py), [Paper](https://arxiv.org/abs/1502.03509) |
| NADE | 85.65 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/autoregressive/nade.py), [Paper](http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf) |
| FVSBN | 96.58 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/autoregressive/fvbn.py), [Paper](https://www.semanticscholar.org/paper/Connectionist-Learning-of-Belief-Networks-Neal/a120c05ad7cd4ce2eb8fb9697e16c7c4877208a5) |

### Variational Autoencoders

NOTE: The results below are the (variational) upper bound on the negative log likelihod (or equivalently, the lower bound on the log likelihod). 

| Algorithm | [Binarized MNIST](https://paperswithcode.com/sota/image-generation-on-binarized-mnist) (nats) | Links |
| --- | ---| --- |
| VD-VAE | **<= 80.72** | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/vae/vd_vae.py), [Paper](https://arxiv.org/abs/2011.10650) |
| VAE | <= 86.77 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/vae/vae.py), [Paper](https://arxiv.org/abs/1312.6114) |
| BetaVAE | N/A | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/vae/beta_vae.py), [Paper](https://openreview.net/pdf?id=Sy2fzU9gl) |
| VQ-VAE | N/A | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/vae/vq_vae.py), [Paper](https://arxiv.org/abs/1711.00937) |
| VQ-VAE-2 | N/A | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/vae/vq_vae_2.py), [Paper](https://arxiv.org/abs/1906.00446) |


### Normalizing Flows

NOTE: Bits per dimension (bits/dim) can be calculated as `(nll / 784 + log(256)) / log(2)` where `784` is the MNIST dimension,
`log(256)` accounts for dequantizing pixel values, and `log(2.0)` converts from natural log to base 2.  

| Algorithm | [MNIST](https://paperswithcode.com/sota/image-generation-on-mnist) (bits/dim) | Links |
| --- | ---| --- |
| NICE | 4.34 | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/flow/nice.py), [Paper](https://arxiv.org/abs/1410.8516) |


### Miscellaneous

| Algorithm |  Links |
| --- | --- |
| Mixture Models | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/mixture_models.py), [Wiki](https://en.wikipedia.org/wiki/Mixture_model) |
| Kernel Density Estimators | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/kde.py), [Wiki](https://en.wikipedia.org/wiki/Kernel_density_estimation) |
| Nerual Style Transfer | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/style_transfer.ipynb), [Blog](https://towardsdatascience.com/how-to-get-beautiful-results-with-neural-style-transfer-75d0c05d6489), [Paper](https://arxiv.org/abs/1508.06576) |
| Compositional Pattern Producing Networks | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/cppn.ipynb), [Wiki](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) |
