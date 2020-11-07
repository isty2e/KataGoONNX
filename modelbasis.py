import torch
import torch.nn as nn

from operations import *


class KataGPool(nn.Module):
    def __init__(self):
        super(KataGPool, self).__init__()

    def forward(self, x, mask):
        mask_sum_hw = mask.sum(dim=(1, 2, 3))
        mask_sum_hw_sqrt = mask_sum_hw.sqrt()
        div = mask_sum_hw.reshape((-1, 1, 1, 1))
        div_sqrt = mask_sum_hw_sqrt.reshape((-1, 1, 1, 1))

        layer_mean = x.sum(dim=(2, 3), keepdim=True) / div
        # meh, this does not exist in PyTorch
        # layer_max = torch.max(x, dim=(2, 3), keepdim=True)
        # Instead,
        layer_max = x.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        out_pool1 = layer_mean
        out_pool2 = layer_mean * (div_sqrt - 14.0) / 10.0
        out_pool3 = layer_max

        out = torch.cat((out_pool1, out_pool2, out_pool3), 1)
        return out


class KataValueHeadGPool(nn.Module):
    def __init__(self):
        super(KataValueHeadGPool, self).__init__()

    def forward(self, x, mask):
        mask_sum_hw = mask.sum(dim=(1, 2, 3))
        mask_sum_hw_sqrt = mask_sum_hw.sqrt()
        div = mask_sum_hw.reshape((-1, 1, 1, 1))
        div_sqrt = mask_sum_hw_sqrt.reshape((-1, 1, 1, 1))

        layer_mean = x.sum(dim=(2, 3), keepdim=True) / div

        out_pool1 = layer_mean
        out_pool2 = layer_mean * (div_sqrt - 14.0) / 10.0
        out_pool3 = layer_mean * ((div_sqrt - 14.0) * (div_sqrt - 14.0) / 100.0 - 0.1)

        out = torch.cat((out_pool1, out_pool2, out_pool3), 1)
        return out


class KataGPoolCell(nn.Module):
    def __init__(self, C_in, C_gpool, C_regular, activation, normalization):
        super(KataGPoolCell, self).__init__()
        self.norm1 = NormMask(C_in, normalization)
        self.act1 = act(activation)
        self.conv1_3x3 = conv3x3(C_in, C_regular)
        self.conv2_3x3 = conv3x3(C_in, C_gpool)
        self.norm2 = NormMask(C_gpool, normalization)
        self.act2 = act(activation)
        self.gpool = KataGPool()
        self.linear = nn.Linear(3 * C_gpool, C_regular, bias=False)

    def forward(self, x, mask):
        out0 = self.norm1(x, mask)
        out0 = self.act1(out0)
        out1 = self.conv1_3x3(out0)
        out2 = self.conv2_3x3(out0)
        out2 = self.norm2(out2, mask)
        out2 = self.act2(out2)
        out3 = self.gpool(out2, mask).squeeze()
        out3 = self.linear(out3).unsqueeze(-1).unsqueeze(-1)
        out = out1 + out3
        return out


class ResBlock(nn.Module):
    def __init__(self, C_in, activation, normalization):
        super(ResBlock, self).__init__()
        self.conv1_3x3 = NormMaskActConv3x3(
            C_in, C_in, activation=activation, normalization=normalization
        )
        self.conv2_3x3 = NormMaskActConv3x3(
            C_in,
            C_in,
            activation=activation,
            normalization=normalization,
            fixup_use_gamma=True,
        )

    def forward(self, x, mask):
        residual = x
        out = self.conv1_3x3(x, mask)
        out = self.conv2_3x3(out, mask)
        out += residual
        return out


class GpoolResBlock(nn.Module):
    def __init__(self, C_in, C_gpool, C_regular, activation, normalization):
        super(GpoolResBlock, self).__init__()
        self.pool = KataGPoolCell(
            C_in, C_gpool, C_regular, activation=activation, normalization=normalization
        )
        self.conv1_3x3 = NormMaskActConv3x3(
            C_regular,
            C_in,
            activation=activation,
            normalization=normalization,
            fixup_use_gamma=True,
        )

    def forward(self, x, mask):
        residual = x
        out = self.pool(x, mask)
        out = self.conv1_3x3(out, mask)
        out += residual
        return out


class PolicyHead(nn.Module):
    def __init__(self, C_in, C_p, C_pg, activation, normalization):
        super(PolicyHead, self).__init__()
        self.conv1_1x1 = conv1x1(C_in, C_p)
        self.conv2_1x1 = conv1x1(C_in, C_pg)
        self.norm1 = NormMask(C_pg, normalization)
        self.act1 = act(activation)
        self.gpool = KataGPool()
        self.linear_pass = nn.Linear(3 * C_pg, 1, bias=False)
        self.linear = nn.Linear(3 * C_pg, C_p, bias=False)
        self.conv3_1x1 = NormMaskActConv1x1(
            C_p, 1, activation=activation, normalization=normalization
        )

    def forward(self, x, mask):
        out_p = self.conv1_1x1(x)
        out_g = self.conv2_1x1(x)
        out_g = self.norm1(out_g, mask)
        out_g = self.act1(out_g)
        out_pool = self.gpool(out_g, mask).squeeze()
        # pass policy subhead output
        out_pass = self.linear_pass(out_pool)

        out_pool = self.linear(out_pool).unsqueeze(-1).unsqueeze(-1)
        out_p += out_pool
        # policy subhead output
        out_policy = self.conv3_1x1(out_p, mask)
        out_policy = out_policy - (1.0 - mask) * 5000.0

        # postprocessing
        # out_policy = out_policy.reshape((-1, 1, 361))
        # mask_sum_hw = mask.sum(dim=(1, 2, 3))
        # out_policy = out_policy.reshape((-1, 1, mask_sum_hw[0].round().int().item()))
        # out_pass = out_pass.reshape((-1, 1, 1))
        out_policy = out_policy.flatten(start_dim=1)
        out_pass = out_pass.reshape((-1, 1))
        return torch.cat((out_policy, out_pass), -1)


class ValueHead(nn.Module):
    def __init__(self, C_in, C_v1, C_v2, activation, normalization):
        super(ValueHead, self).__init__()
        self.init_conv = conv1x1(C_in, C_v1)
        self.norm1 = NormMask(C_v1, normalization)
        self.act1 = act(activation)
        self.vh_gpool = KataValueHeadGPool()
        self.linear_after_pool = nn.Linear(3 * C_v1, C_v2)
        self.act_after_pool = act(activation)
        # value subhead
        self.linear_valuehead = nn.Linear(C_v2, 3)
        # misc value subhead
        self.linear_miscvaluehead = nn.Linear(C_v2, 4)
        # ownership subhead
        self.conv_ownership = conv1x1(C_v1, 1)

    def forward(self, x, mask):
        out_v1 = self.init_conv(x)
        out_v1 = self.norm1(out_v1, mask)
        out_v1 = self.act1(out_v1)
        out_pooled = self.vh_gpool(out_v1, mask).squeeze()
        out_pooled = self.linear_after_pool(out_pooled)
        out_v2 = self.act_after_pool(out_pooled)
        # value subhead output
        out_value = self.linear_valuehead(out_v2)
        # misc value subhead
        out_miscvalue = self.linear_miscvaluehead(out_v2)
        # ownership subhead output
        # out_ownership = self.conv_ownership(out_v1) * mask
        out_ownership = self.conv_ownership(out_v1)

        return out_value, out_miscvalue, out_ownership
