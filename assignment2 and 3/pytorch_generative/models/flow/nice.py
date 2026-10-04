"""Implementation of the Non-linear Independent Components Estimation (NICE) [1].

References (used throughout the code):
  [1]: https://arxiv.org/abs/1410.8516
"""

import operator

import torch
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base


class AdditiveCouplingBlock(nn.Module):
    """Coupling block with an additive coupling law.

    Given inputs x1, x2 = split(x, 2) and coupling network m(.), the reverse mapping is
    defined as y1 = x1, y2 = x2 + m(x1) and the forward mapping is defined as
    x1 = y1, x2 = y2 - m(y1).
    """

    def __init__(
        self,
        n_features,
        n_hidden_layers,
        n_hidden_features,
        reverse,
        linear_cls=nn.Linear,
    ):
        """Initializes a new AdditiveCouplingBlock instance.

        Args:
            n_features: Number of input and output features.
            n_hidden_layers: Number of hidden layers in the coupling network.
            n_hidden_features: Number of features in each hidden layer.
            reverse: Whether to reverse which half of the input is transformed by the
                coupling network.
        """
        super().__init__()
        self.reverse = reverse
        half_features = n_features // 2 #把x分成两半，half_features是每一半的特征数，n_features是总特征数，half_features是一半维度
        #构造一个coupling network,第一层把 392 维映射到隐藏层
        net = [
            linear_cls(in_features=half_features, out_features=n_hidden_features),
            nn.ReLU(),
        ]
        #继续堆叠隐藏层，堆叠n_hidden_layers个隐藏层，接在第一层network的后面
        for _ in range(n_hidden_layers - 1):
            net.append(
                linear_cls(
                    in_features=n_hidden_features, out_features=n_hidden_features
                )
            )
            net.append(nn.ReLU())
        net.append(linear_cls(in_features=n_hidden_features, out_features=half_features)) #最后的输出层把隐藏层映射回half_features维
        self.net = nn.Sequential(*net) #把所有的层组合成一个Sequential模型

    def _couple(self, x, op): #执行耦合变换
        c = x.shape[1]#获取输入x的特征维度
        h1, h2 = x[:, : c // 2], x[:, c // 2 :]#将输入x分成两半，h1是前一半特征，h2是后一半特征
        if self.reverse:#如果reverse为True，则h1不变，h2通过耦合网络进行变换
            h1 = op(h1, self.net(h2))
        else:#如果reverse为False，则h2不变，h1通过耦合网络进行变换
            h2 = op(h2, self.net(h1))
        return torch.cat((h1, h2), dim=1)#torth为深度学习框架
    #torch.cat是将两个张量沿指定维度拼接起来，这里dim=1是沿着特征维度拼接h1和h2，得到最终的输出而样本数量不变

    def forward(self, x):#把输入x映射到潜在空间z
        """Inverse mapping from the inputs to the prior (X -> Z)."""
        return self._couple(x, operator.add)

    def inverse(self, y):#把潜在空间z映射回输入x
        """Forward mapping from the prior to the input (Z -> X)."""
        return self._couple(y, operator.sub)
    #forward 时，把 operator.add 这个“加法函数”传给 _couple
    #inverse 时，把 operator.sub 这个“减法函数”传给 _couple
    #对应前文函数op

class ScalingLayer(nn.Module):
    """Invertible scaling layer representing a diagonal scaling matrix S."""

    def __init__(self, n_features):
        """Initializes a new ScalingLayer instance.

        Args:
            n_features: Number of input and output features.
        """
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros((1, n_features)))
        #定义一个可训练参数 log_scale，形状是 [1, 784]。它不是普通矩阵，而是每个维度一个缩放因子。
        #并初始化为全零张量
    def log_det_J(self):
        """Returns the log determinant of the Jacobian of the scaling matrix S.

        Makes use of the identity log det(S) = Tr(log S).
        """
        return torch.sum(self.log_scale)#把log_scale的所有元素求和，得到对角缩放矩阵S的雅可比行列式的对数值
        #例如：log_scale=[0.2,0.5,-1]那么：sum=-0.3

    def _couple(self, x, sign):
        original_shape = x.shape
        x = x.view(original_shape[0], -1)#original_shape[0]是样本数量，-1表示自动计算特征维度
        #例如如果original_shape是[64, 1, 28, 28]，那么x.view(original_shape[0], -1)会把x变成[64, 784]，即每个样本展平为一维向量
        h = x * torch.exp(sign * self.log_scale)#根据sign的正负决定是乘以指数还是除以指数
        return h.view(original_shape)#把原来展平的张量h重新reshape回原来的形状

    def forward(self, x):
        """Inverse mapping from the inputs to the prior (X -> Z)."""
        return self._couple(x, 1)#x * exp(log_scale)

    def inverse(self, y):
        """Forward mapping from the prior to the input (Z -> X)."""
        return self._couple(y, -1)#y * exp(-log_scale) = y / exp(log_scale)


class NICE(base.GenerativeModel):
    """Non-linear Independent Component Estimation (NICE) model."""

    def __init__(
        self,
        n_features,
        n_coupling_blocks=4,
        n_hidden_layers=5,
        n_hidden_features=1000,
        linear_cls=nn.Linear,
    ):
        #输入特征数 耦合块数量 每个小网络隐藏层数 隐藏层宽度
        """Initializes a new NICE instance.

        Args:
            n_features: Number of input and output features.
            n_coupling_blocks: Number of coupling blocks. Should be at least 3 to allow
                all dimensions to influence one another.
            n_hidden_layers: Number of hidden layers per coupling block.
            n_hidden_features: Number of features in each hidden layer.
        """
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
                    linear_cls=linear_cls,
                ),
            )
            reverse = not reverse#交替前后部分相互控制
        self.net = nn.Sequential(*net)
        self.scaling = ScalingLayer(n_features)
        #循环创建多个可逆耦合层，让前后两半特征轮流被更新，最后把这些层串成一个完整网络

    def forward(self, x):#训练时从图片到潜在空间z，Trainer类中调用forward方法
        """Inverse mapping from the inputs to the prior (X -> Z)."""
        return self._forward(x), self.scaling.log_det_J()

    @base.auto_reshape
    def _forward(self, x):#处理输入x，返回潜在空间z
        y = self.net(x)
        y = self.scaling(y)
        return y

    def sample(self, n_samples, temp=1.0):#生成随机样本，Trainer类中调用sample方法
        """See the base class.

        Args:
            n_samples: Number of samples to generate.
            temp: What temperature to use when sampling. Lower temperature produces
                higher quality samples with lower diversity.
        """
        # NOTE: Technically this should sample from the logistic distribution when using
        # a logistic prior. However, we sample from the standard normal for both
        # logistic and normal priors and use the temperature to control the variance.
        x = torch.randn((n_samples, self._c, self._h, self._w)) * temp#生成随机噪声样本，形状为 [n_samples, channels, height, width]，并乘以温度参数temp来控制样本的多样性
        #temp越低，样本质量越高，但多样性越低；temp越高，样本多样性越高，但质量可能下降
        x = x.to(self.device)#把生成出来的随机噪声样本x移动到模型所在的设备上（CPU或GPU），以确保后续计算不会出现设备不匹配的错误
        return self._inverse(x)

    @base.auto_reshape
    def _inverse(self, x):#把潜在空间z映射回输入x，相当于是def _forward(self, x)的逆过程
        x = self.scaling.inverse(x)
        for block in reversed(self.net):
            x = block.inverse(x)
        return x


class CirculantNICE(NICE):
    """NICE model whose coupling networks use block-circulant linear layers."""

    def __init__(
        self, n_features, n_coupling_blocks=4, n_hidden_layers=5, n_hidden_features=1000
    ):
        super().__init__(
            n_features=n_features,
            n_coupling_blocks=n_coupling_blocks,
            n_hidden_layers=n_hidden_layers,
            n_hidden_features=n_hidden_features,
            linear_cls=pg_nn.BlockCirculantLinear,
        )


class PaddedNICE(NICE):
    """NICE that pads 28x28 MNIST vectors to a larger training dimension.

    The model keeps the ordinary NICE coupling network layers, but trains in a
    padded invertible space. Generated samples are cropped back to the original
    image dimensions.
    """

    def __init__(
        self,
        n_features=784,
        padded_features=1024,
        n_coupling_blocks=4,
        n_hidden_layers=5,
        n_hidden_features=1024,
        linear_cls=nn.Linear,
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
            linear_cls=linear_cls,
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
        """Inverse mapping from 784-dimensional MNIST inputs to 1024-dimensional z."""
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
        """Generate samples and crop the padded output back to the MNIST image size."""
        z = torch.randn((n_samples, self.padded_features), device=self.device) * temp
        x = self.scaling.inverse(z)
        for block in reversed(self.net):
            x = block.inverse(x)
        x = x[:, : self.original_features]
        return x.view(n_samples, self._c, self._h, self._w)


class PaddedBlockCirculantNICE(PaddedNICE):
    """Block-circulant NICE that pads 28x28 MNIST vectors to 1024 dimensions."""

    def __init__(
        self,
        n_features=784,
        padded_features=1024,
        n_coupling_blocks=4,
        n_hidden_layers=5,
        n_hidden_features=1024,
    ):
        super().__init__(
            n_features=n_features,
            padded_features=padded_features,
            n_coupling_blocks=n_coupling_blocks,
            n_hidden_layers=n_hidden_layers,
            n_hidden_features=n_hidden_features,
            linear_cls=pg_nn.BlockCirculantLinear,
        )


def reproduce(
    n_epochs=150,#训练的轮数
    batch_size=1024,#每次训练使用的样本数量
    log_dir="/tmp/run",#日志存放目录
    n_gpus=1,#使用的GPU数量，如果为0，则使用CPU
    device_id=0,#当前GPU的设备ID，当使用多个GPU训练模型时指定
    debug_loader=None,#调试用的DataLoader，如果不为None，则替换默认的训练和评估加载器。仅在编写单元测试时使用。
):
    """Training script with defaults to reproduce results.

    The code inside this function is self contained and can be used as a top level
    training script, e.g. by copy/pasting it into a Jupyter notebook.

    Args:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size to use for training and evaluation.
        log_dir: Directory where to log trainer state and TensorBoard summaries.
        n_gpus: Number of GPUs to use for training the model. If 0, uses CPU.
        device_id: The device_id of the current GPU when training on multiple GPUs.
        debug_loader: Debug DataLoader which replaces the default training and
            evaluation loaders if not 'None'. Do not use unless you're writing unit
            tests.
    """
    from torch import optim
    from torch.nn import functional as F

    from pytorch_generative import datasets, models, trainer

    train_loader, test_loader = debug_loader, debug_loader
    if train_loader is None:#如果没有提供调试加载器，则使用默认的MNIST数据集加载器
        train_loader, test_loader = datasets.get_mnist_loaders(
            batch_size, dequantize=True
        )

    model = models.NICE(#实例化NICE模型，指定输入特征数、耦合块数量、隐藏层数量和隐藏层宽度
        n_features=784, n_coupling_blocks=4, n_hidden_layers=5, n_hidden_features=1000
    )
    # NOTE: We found most hyperparameters from the paper give bad results so we only use
    # the learning rate.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)#创建Adam优化器，指定模型参数和学习率为0.001，让神经网络训练时根据梯度更新参数

    def loss_fn(x, _, preds):
        preds, log_det_J = preds
        log_prob = -(F.softplus(preds) + F.softplus(-preds)).sum(dim=(1, 2, 3))#计算对数似然概率，使用softplus函数来稳定计算softplus函数是log(1 + exp(x))的缩写，可以避免数值溢出
        loss = log_prob + log_det_J
        return {
            "loss": -loss.mean(),#对数似然损失取负号并求平均，作为训练的目标函数
            "prior_log_likelihood": log_prob.mean(),#先验对数似然的平均值，用于评估模型在潜在空间上的表现
            "log_det_J": log_det_J.mean(),#雅可比行列式的对数值的平均值，用于评估模型的可逆性和稳定性
        }

    trainer = trainer.Trainer(
        model=model,#需要训练的模型
        loss_fn=loss_fn,#计算损失的函数
        optimizer=optimizer,#优化器，用于更新模型参数
        train_loader=train_loader,#训练数据加载器，用于提供训练样本
        eval_loader=test_loader,#评估数据加载器，用于提供验证样本
        lr_scheduler=None,#学习率调度器，用于动态调整学习率，这里设置为None表示不使用调度器
        log_dir=log_dir,#日志目录，用于存储训练过程中的日志和模型检查点
        n_gpus=n_gpus,#使用的GPU数量，如果为0，则使用CPU进行训练
        device_id=device_id,#当前GPU的设备ID，当使用多个GPU训练模型时指定
    )
    trainer.interleaved_train_and_eval(n_epochs)#开始训练和评估模型，训练轮数为n_epochs
