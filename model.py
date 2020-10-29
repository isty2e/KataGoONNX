import torch
import torch.nn as nn

from modelbasis import *
from operations import NormMask, act


class KataGoInferenceModel(nn.Module):
    def __init__(self, conf):
        super(KataGoInferenceModel, self).__init__()

        self.conf = conf
        self.block_kind = conf["config"]["block_kind"]
        self.C = conf["config"]["trunk_num_channels"]
        self.C_mid = conf["config"]["mid_num_channels"]
        self.C_gpool = conf["config"]["gpool_num_channels"]
        self.C_regular = conf["config"]["regular_num_channels"]
        self.C_p = conf["config"]["p1_num_channels"]
        self.C_pg = conf["config"]["g1_num_channels"]
        self.C_v1 = conf["config"]["v1_num_channels"]
        self.C_v2 = conf["config"]["v2_size"]
        self.activation = "ReLU"
        if conf["config"]["use_fixup"]:
            self.normalization = "FixUp"
        else:
            self.normalization = "BN"

        self.linear_ginput = nn.Linear(19, self.C, bias=False)
        self.conv1 = nn.Conv2d(22, self.C, 5, 1, 2, bias=False)

        self.blocks = nn.ModuleList()
        for block_conf in self.block_kind:
            if block_conf[1] == "regular":
                self.blocks += [ResBlock(self.C, self.activation, self.normalization)]
            elif block_conf[1] == "gpool":
                self.blocks += [
                    GpoolResBlock(
                        self.C,
                        self.C_gpool,
                        self.C_regular,
                        self.activation,
                        self.normalization,
                    )
                ]
            else:
                assert False

        self.norm1 = NormMask(self.C, self.normalization)
        self.act1 = act(self.activation)
        self.policy_head = PolicyHead(
            self.C, self.C_p, self.C_pg, self.activation, self.normalization
        )
        self.value_head = ValueHead(
            self.C, self.C_v1, self.C_v2, self.activation, self.normalization
        )

    def forward(self, input_binary, input_global):
        mask = input_binary[:, 0:1, :, :]

        x_bin = self.conv1(input_binary)
        x_global = self.linear_ginput(input_global).unsqueeze(-1).unsqueeze(-1)
        out = x_bin + x_global

        for block in self.blocks:
            out = block(out, mask)

        out = self.norm1(out, mask)
        out = self.act1(out)

        out_policy = self.policy_head(out, mask)
        (out_value, out_miscvalue, out_ownership) = self.value_head(out, mask)

        return (
            out_policy,
            out_value,
            out_miscvalue,
            out_ownership,
        )

    def fill_misc_weights(self, conv1_weight, linear_ginput_weight, norm1_weight):
        self.conv1.weight.data = conv1_weight
        self.linear_ginput.weight.data = linear_ginput_weight
        self.norm1._norm.beta.data = norm1_weight

    def fill_regular_block(self, block, block_dict):
        block.conv1_3x3.op.norm._norm.beta.data = block_dict["norm1"]["beta"]
        block.conv1_3x3.op.conv.weight.data = block_dict["conv1"]["weights"]
        block.conv2_3x3.op.norm._norm.gamma.data = block_dict["norm2"]["gamma"]
        block.conv2_3x3.op.norm._norm.beta.data = block_dict["norm2"]["beta"]
        block.conv2_3x3.op.conv.weight.data = block_dict["conv2"]["weights"]

    def fill_gpool_block(self, block, block_dict):
        block.pool.norm1._norm.beta.data = block_dict["norm1"]["beta"]
        block.pool.conv1_3x3.weight.data = block_dict["conv1"]["weights"]
        block.pool.conv2_3x3.weight.data = block_dict["conv2"]["weights"]
        block.pool.norm2._norm.beta.data = block_dict["norm2"]["beta"]
        block.pool.linear.weight.data = block_dict["matmul1"]["weights"]
        block.conv1_3x3.op.norm._norm.gamma.data = block_dict["norm3"]["gamma"]
        block.conv1_3x3.op.norm._norm.beta.data = block_dict["norm3"]["beta"]
        block.conv1_3x3.op.conv.weight.data = block_dict["conv3"]["weights"]

    def fill_policy_head(self, policy_dict):
        self.policy_head.conv1_1x1.weight.data = policy_dict["convp"]["weights"]
        self.policy_head.conv2_1x1.weight.data = policy_dict["convg"]["weights"]
        self.policy_head.norm1._norm.beta.data = policy_dict["normg"]["beta"]
        self.policy_head.linear_pass.weight.data = policy_dict["matmulpass"]["weights"]
        self.policy_head.linear.weight.data = policy_dict["matmulg"]["weights"]
        self.policy_head.conv3_1x1.op.norm._norm.beta.data = policy_dict["norm2"][
            "beta"
        ]
        self.policy_head.conv3_1x1.op.conv.weight.data = policy_dict["conv3"]["weights"]

    def fill_value_head(self, value_dict):
        self.value_head.init_conv.weight.data = value_dict["conv1"]["weights"]
        self.value_head.norm1._norm.beta.data = value_dict["norm1"]["beta"]
        self.value_head.linear_after_pool.weight.data = value_dict["matmul1"]["weights"]
        self.value_head.linear_after_pool.bias.data = value_dict["matbias1"]["weights"]
        self.value_head.linear_valuehead.weight.data = value_dict["matmul2"]["weights"]
        self.value_head.linear_valuehead.bias.data = value_dict["matbias2"]["weights"]
        self.value_head.linear_miscvaluehead.weight.data = value_dict["matmul3"][
            "weights"
        ]
        self.value_head.linear_miscvaluehead.bias.data = value_dict["matbias3"][
            "weights"
        ]
        self.value_head.conv_ownership.weight.data = value_dict["conv2"]["weights"]

    def fill_weights(self):
        print("Filling misc weights")
        self.fill_misc_weights(
            self.conf["initial_conv"]["weights"],
            self.conf["initial_matmul"]["weights"],
            self.conf["postprocess_norm"]["beta"],
        )
        for i, block_conf in enumerate(self.block_kind):
            print(f"Filling block {i} weights ({block_conf[1]})")
            if block_conf[1] == "regular":
                self.fill_regular_block(self.blocks[i], self.conf["blocks"][i])
            elif block_conf[1] == "gpool":
                self.fill_gpool_block(self.blocks[i], self.conf["blocks"][i])
            else:
                assert False
        print("Filling policy head weights")
        self.fill_policy_head(self.conf["policy_head"])
        print("Filling value head weights")
        self.fill_value_head(self.conf["value_head"])
