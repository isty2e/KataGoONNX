from argparse import ArgumentParser

import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16


def main(args):
    onnx_model = onnxmltools.utils.load_model(args.input)
    fp16_model = convert_float_to_float16(onnx_model)

    onnxmltools.utils.save_model(fp16_model, args.output)

    print(f"{args.input} is quantized and saved as {args.output}.")


if __name__ == "__main__":
    description = """
    Convert ONNX model to FP16.
    """
    parser = ArgumentParser(description)
    parser.add_argument("--input", type=str, required=True, help="ONNX file location")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to export FP16 .onnx network file",
    )

    args = parser.parse_args()
    if args.output is None:
        args.output = args.input.replace(".onnx", "_fp16.onnx")

    main(args)
