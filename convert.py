from argparse import ArgumentParser

import numpy as np
import torch
import torch.onnx

from model import KataGoInferenceModel
from netparser import read_model


def main(args):
    model_spec = read_model(args.model, args.model_config)
    model = KataGoInferenceModel(model_spec)
    print("Model building completed")
    model.fill_weights()

    dummy_input_binary = torch.randn(10, 22, 19, 19)
    dummy_input_binary[:, 0, :, :] = 1.0
    dummy_input_global = torch.randn(10, 19)
    input_names = ["input_binary", "input_global"]
    output_names = [
        "output_policy",
        "output_value",
        "output_miscvalue",
        "output_ownership",
    ]

    torch.onnx.export(
        model,
        (dummy_input_binary, dummy_input_global),
        args.output,
        export_params=True,
        verbose=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_binary": {0: "batch_size", 2: "y_size", 3: "x_size"},
            "input_global": {0: "batch_size"},
            "output_policy": {0: "batch_size", 1: "board_area + 1"},
            "output_value": {0: "batch_size"},
            "output_miscvalue": {0: "batch_size"},
            "output_ownership": {0: "batch_size", 2: "y_size", 3: "x_size"},
        },
    )

    print(f"ONNX model saved in {args.output}")


if __name__ == "__main__":
    description = """
    Convert KataGo .bin model to .onnx file.
    """
    parser = ArgumentParser(description)
    parser.add_argument(
        "--model", type=str, required=True, help="KataGo .bin network file location"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="KataGo model.config.json file location (usually archived in the .zip file)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output .onnx network file location"
    )

    args = parser.parse_args()
    if args.output is None:
        args.output = args.model.replace(".bin", ".onnx")

    main(args)
