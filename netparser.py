import json
import struct

import numpy as np
import torch


def bin2str(binary_str):
    return binary_str.decode(encoding="ascii", errors="backslashreplace")


def read_header(lines, idx):
    header_dict = {
        "type": "header",
        "name": bin2str(lines[idx + 0]),
        "version": int(lines[idx + 1]),
        "num_bin_input_features": int(lines[idx + 2]),
        "num_global_input_features": int(lines[idx + 3]),
        # lines[idx + 4] == "trunk"
        "num_blocks": int(lines[idx + 5]),
        "num_channels": int(lines[idx + 6]),
        "num_mid_channels": int(lines[idx + 7]),
        "num_regular_channels": int(lines[idx + 8]),
        "num_dilated_channels": int(lines[idx + 9]),
        "num_gpool_channels": int(lines[idx + 10]),
    }

    return header_dict, idx + 11


def read_weights(lines, idx, shape):
    assert lines[idx][0:5] == b"@BIN@"
    buffer_size = struct.calcsize(f"<{np.prod(shape)}f")

    i_increment = 0
    buffer = lines[idx][5:]
    while len(buffer) < buffer_size:
        buffer += "\n".encode(encoding="ascii", errors="backslashreplace")
        if len(buffer) == buffer_size:
            break
        i_increment += 1
        buffer += lines[idx + i_increment]
    assert buffer_size == len(buffer)
    weights = np.array(struct.unpack(f"<{np.prod(shape)}f", buffer))
    weights = torch.tensor(weights.reshape(shape), dtype=torch.float)

    return weights, i_increment


def read_conv(lines, idx):
    diam_y = int(lines[idx + 1])
    diam_x = int(lines[idx + 2])
    C_in = int(lines[idx + 3])
    C_out = int(lines[idx + 4])
    weights, i_increment = read_weights(lines, idx + 7, (diam_y, diam_x, C_in, C_out))
    # print(f"conv weight mean: {torch.mean(weights)}")

    conv_dict = {
        "type": "conv",
        "name": bin2str(lines[idx]),
        "diam_y": diam_y,
        "diam_x": diam_x,
        "C_in": C_in,
        "C_out": C_out,
        "dil_y": int(lines[idx + 5]),
        "dil_x": int(lines[idx + 6]),
        # TF default NHWC:
        # (height, width, C_in, C_out)
        # PyTorch default NCHW:
        # (C_out, C_in, height, width)
        "weights": weights.permute(3, 2, 0, 1),
    }

    return conv_dict, idx + 8 + i_increment


def read_norm(lines, idx):
    C_in = int(lines[idx + 1])
    norm_dict = {
        "type": "norm",
        "name": bin2str(lines[idx]),
        "C_in": C_in,
        "eps": float(lines[idx + 2]),
        "has_scale": bool(int(lines[idx + 3])),
        "has_bias": bool(int(lines[idx + 4])),
    }

    idx_increment = 0

    moving_mean, i_increment = read_weights(lines, idx + 5 + idx_increment, (C_in))
    norm_dict["moving_mean"] = moving_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    idx_increment += 1 + i_increment

    moving_variance, i_increment = read_weights(lines, idx + 5 + idx_increment, (C_in))
    norm_dict["moving_variance"] = (
        moving_variance.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    )
    idx_increment += 1 + i_increment

    if norm_dict["has_scale"]:
        gamma, i_increment = read_weights(lines, idx + 5 + idx_increment, (C_in))
        norm_dict["gamma"] = gamma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        idx_increment += 1 + i_increment

    if norm_dict["has_bias"]:
        beta, i_increment = read_weights(lines, idx + 5 + idx_increment, (C_in))
        norm_dict["beta"] = beta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        idx_increment += 1 + i_increment

    return norm_dict, idx + 5 + idx_increment


def read_act(lines, idx):
    act_dict = {"type": "act", "act": bin2str(lines[idx])}

    return act_dict, idx + 1


def read_matmul(lines, idx):
    C_in = int(lines[idx + 1])
    C_out = int(lines[idx + 2])
    weights, i_increment = read_weights(lines, idx + 3, (C_in, C_out))

    matmul_dict = {
        "type": "matmul",
        "name": bin2str(lines[idx]),
        "C_in": C_in,
        "C_out": C_out,
        # TensorFlow (2D):
        # (C_in, C_out)
        # PyTorch:
        # (C_out, C_in)
        "weights": weights.permute(1, 0),
    }

    return matmul_dict, idx + 4 + i_increment


def read_matbias(lines, idx):
    C_in = int(lines[idx + 1])
    weights, i_increment = read_weights(lines, idx + 2, (C_in))

    matbias_dict = {
        "type": "matbias",
        "name": bin2str(lines[idx]),
        "C_in": C_in,
        "weights": weights,
    }

    return matbias_dict, idx + 3 + i_increment


def read_block(lines, idx):
    # block_type is "ordinary_block" or "gpool_block"
    # (dilated blocks are not really used)
    block_type = bin2str(lines[idx])
    name = bin2str(lines[idx + 1])
    head_idx = idx + 2

    if block_type == "ordinary_block":
        norm1, head_idx = read_norm(lines, head_idx)
        act1, head_idx = read_act(lines, head_idx)
        conv1, head_idx = read_conv(lines, head_idx)
        norm2, head_idx = read_norm(lines, head_idx)
        act2, head_idx = read_act(lines, head_idx)
        conv2, head_idx = read_conv(lines, head_idx)

        block_dict = {
            "type": block_type,
            "name": name,
            "norm1": norm1,
            "act1": act1,
            "conv1": conv1,
            "norm2": norm2,
            "act2": act2,
            "conv2": conv2,
        }

        return block_dict, head_idx
    if block_type == "gpool_block":
        norm1, head_idx = read_norm(lines, head_idx)
        act1, head_idx = read_act(lines, head_idx)
        conv1, head_idx = read_conv(lines, head_idx)
        conv2, head_idx = read_conv(lines, head_idx)
        norm2, head_idx = read_norm(lines, head_idx)
        act2, head_idx = read_act(lines, head_idx)
        matmul1, head_idx = read_matmul(lines, head_idx)
        norm3, head_idx = read_norm(lines, head_idx)
        act3, head_idx = read_act(lines, head_idx)
        conv3, head_idx = read_conv(lines, head_idx)

        block_dict = {
            "type": block_type,
            "name": name,
            "norm1": norm1,
            "act1": act1,
            "conv1": conv1,
            "conv2": conv2,
            "norm2": norm2,
            "act2": act2,
            "matmul1": matmul1,
            "norm3": norm3,
            "act3": act3,
            "conv3": conv3,
        }

        return block_dict, head_idx
    else:
        assert False


def read_policy_head(lines, idx):
    name = bin2str(lines[idx])
    head_idx = idx + 1

    conv1, head_idx = read_conv(lines, head_idx)
    conv2, head_idx = read_conv(lines, head_idx)
    norm1, head_idx = read_norm(lines, head_idx)
    act1, head_idx = read_act(lines, head_idx)
    matmul1, head_idx = read_matmul(lines, head_idx)
    norm2, head_idx = read_norm(lines, head_idx)
    act2, head_idx = read_act(lines, head_idx)
    conv3, head_idx = read_conv(lines, head_idx)
    matmul2, head_idx = read_matmul(lines, head_idx)

    policy_head_dict = {
        "type": "policy head",
        "name": name,
        "convp": conv1,
        "convg": conv2,
        "normg": norm1,
        "actg": act1,
        "matmulg": matmul1,
        "norm2": norm2,
        "act2": act2,
        "conv3": conv3,
        "matmulpass": matmul2,
    }

    return policy_head_dict, head_idx


def read_value_head(lines, idx):
    name = bin2str(lines[idx])
    head_idx = idx + 1

    conv1, head_idx = read_conv(lines, head_idx)
    norm1, head_idx = read_norm(lines, head_idx)
    act1, head_idx = read_act(lines, head_idx)
    matmul1, head_idx = read_matmul(lines, head_idx)
    matbias1, head_idx = read_matbias(lines, head_idx)
    act2, head_idx = read_act(lines, head_idx)
    matmul2, head_idx = read_matmul(lines, head_idx)
    matbias2, head_idx = read_matbias(lines, head_idx)
    matmul3, head_idx = read_matmul(lines, head_idx)
    matbias3, head_idx = read_matbias(lines, head_idx)
    conv2, head_idx = read_conv(lines, head_idx)

    value_head_dict = {
        "type": "value head",
        "name": name,
        "conv1": conv1,
        "norm1": norm1,
        "act1": act1,
        "matmul1": matmul1,
        "matbias1": matbias1,
        "act2": act2,
        "matmul2": matmul2,
        "matbias2": matbias2,
        "matmul3": matmul3,
        "matbias3": matbias3,
        "conv2": conv2,
    }

    return value_head_dict, head_idx


def read_model(model_path, json_config_path):
    print(f"Model config: {json_config_path}")
    with open(json_config_path) as f:
        model_config = json.load(f)
    assert model_config["version"] == 8
    assert model_config["support_japanese_rules"] == True
    assert model_config["use_fixup"] == True
    assert model_config["use_scoremean_as_lead"] == False

    print(f"Model: {model_path}")
    with open(model_path, "rb") as f:
        contents = f.read()
    lines = contents.split("\n".encode(encoding="ascii", errors="backslashreplace"))
    print(f"Model file loading completed.")

    head_idx = 0
    header, head_idx = read_header(lines, head_idx)
    assert header["version"] == 8

    initial_conv, head_idx = read_conv(lines, head_idx)
    initial_matmul, head_idx = read_matmul(lines, head_idx)

    blocks = []
    for i in range(header["num_blocks"]):
        print(f"Reading block {i}")
        block, head_idx = read_block(lines, head_idx)
        blocks.append(block)

    postprocess_norm, head_idx = read_norm(lines, head_idx)
    postprocess_act, head_idx = read_act(lines, head_idx)

    print("Reading policy head")
    policy_head, head_idx = read_policy_head(lines, head_idx)
    print("Reading value head")
    value_head, head_idx = read_value_head(lines, head_idx)

    model_dict = {
        "config": model_config,
        "initial_conv": initial_conv,
        "initial_matmul": initial_matmul,
        "blocks": blocks,
        "postprocess_norm": postprocess_norm,
        "postprocess_act": postprocess_act,
        "policy_head": policy_head,
        "value_head": value_head,
    }

    return model_dict


# KataGo .bin file structure
# model.name
# model.version == 8
# model.num_bin_input_features == 22
# model.num_global_input_features == 19
# trunk
#     "trunk"
#     len(blocks)
#     C_trunk
#     C_mid
#     C_regular
#     C_dilated
#     C_gpool
#     conv
#     matmul
#     blocks
#     if ordinary:
#         block[0] == "ordinary_block"
#         name
#         norm
#         act
#         conv
#         norm
#         act
#         conv
#     if gpool:
#         block[0] == "gpool_block"
#         name
#         norm
#         act
#         conv
#         conv
#         norm
#         act
#         matmul
#         norm
#         act
#         conv
#     norm
#     act
# policy_head
#     "policyhead"
#     conv
#     conv
#     norm
#     act
#     matmul
#     norm
#     act
#     conv
#     matmul
# value_head
#     "valuehead"
#     conv
#     norm
#     act
#     matmul
#     matbias
#     act
#     matmul
#     matbias
#     matmul
#     matbias
#     conv
# format:
# weights
#     "@BIN@{bin_data}\n"
# conv
#     name
#     diam_y
#     diam_x
#     C_in
#     C_out
#     dil_y
#     dil_x
#     weights
# norm
#     name
#     C_in
#     eps
#     has_scale ? 1 : 0
#     has_bias ? 1 : 0
#     zeros(C_in)
#     ones(C_in)
#     gamma if has_scale
#     beta if has_bias
# act
#     name
# matmul
#     name
#     C_in
#     C_out
#     weights
# matbias
#     name
#     C
#     weights
