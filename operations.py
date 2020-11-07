import torch
import torch.nn as nn
from packaging import version


def norm(C_in, norm_name="FixUp", affine=True, fixup_use_gamma=False):
    if norm_name == "BN":
        return nn.BatchNorm2d(C_in, affine=affine)
    if norm_name == "AN":
        return AttenNorm(C_in)
    if norm_name == "FixUp":
        return KataFixUp(C_in, fixup_use_gamma)
    raise NotImplementedError("Unknown feature norm name")


def act(activation, inplace=False):
    if activation == "ReLU":
        return nn.ReLU(inplace=inplace)
    if activation == "Hardswish":
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            return nn.Hardswish(inplace=inplace)
        else:
            return nn.Hardswish()
    if activation == "Identity":
        return nn.Identity()
    raise NotImplementedError("Unknown activation name")


def conv1x1(C_in, C_out):
    return nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False)


def conv3x3(C_in, C_out):
    return nn.Conv2d(C_in, C_out, 3, 1, 1, bias=False)


class AttenNorm(nn.BatchNorm2d):
    def __init__(
        self, num_features, K=5, eps=1e-5, momentum=0.1, track_running_stats=True
    ):
        super(AttenNorm, self).__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=False,
            track_running_stats=track_running_stats,
        )
        self.gamma = nn.Parameter(torch.Tensor(K, num_features))
        self.beta = nn.Parameter(torch.Tensor(K, num_features))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, K)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = super(AttenNorm, self).forward(x)
        size = output.size()
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y)
        y = self.sigmoid(y)
        gamma = y @ self.gamma
        beta = y @ self.beta
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand(size)
        beta = beta.unsqueeze(-1).unsqueeze(-1).expand(size)
        return gamma * output + beta


class KataFixUp(nn.Module):
    def __init__(self, C_in, use_gamma):
        super(KataFixUp, self).__init__()
        self.use_gamma = use_gamma
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.ones(1, C_in, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, C_in, 1, 1))

    def forward(self, x):
        if self.use_gamma:
            return x * self.gamma + self.beta
        else:
            return x + self.beta


class NormMask(nn.Module):
    def __init__(self, C_in, normalization="FixUp", affine=True, fixup_use_gamma=False):
        super(NormMask, self).__init__()
        self._norm = norm(C_in, normalization, affine, fixup_use_gamma)

    def forward(self, x, mask):
        return self._norm(x)
        # return self._norm(x) * mask


class NormMaskActConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        padding,
        affine=True,
        activation="ReLU",
        normalization="FixUp",
        fixup_use_gamma=False,
    ):
        super(NormMaskActConv, self).__init__()
        self.norm = NormMask(
            C_in, normalization, affine=affine, fixup_use_gamma=fixup_use_gamma
        )
        self.act = act(activation, inplace=False)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=False)

    def forward(self, x, mask):
        out = self.norm(x, mask)
        out = self.act(out)
        out = self.conv(out)

        return out


class NormMaskActConv3x3(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        affine=True,
        activation="ReLU",
        normalization="FixUp",
        fixup_use_gamma=False,
    ):
        super(NormMaskActConv3x3, self).__init__()
        self.op = NormMaskActConv(
            C_in,
            C_out,
            kernel_size=3,
            padding=1,
            affine=affine,
            activation=activation,
            normalization=normalization,
            fixup_use_gamma=fixup_use_gamma,
        )

    def forward(self, x, mask):
        return self.op(x, mask)


class NormMaskActConv1x1(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        affine=True,
        activation="ReLU",
        normalization="FixUp",
        fixup_use_gamma=False,
    ):
        super(NormMaskActConv1x1, self).__init__()
        self.op = NormMaskActConv(
            C_in,
            C_out,
            kernel_size=1,
            padding=0,
            affine=affine,
            activation=activation,
            normalization=normalization,
            fixup_use_gamma=fixup_use_gamma,
        )

    def forward(self, x, mask):
        return self.op(x, mask)
