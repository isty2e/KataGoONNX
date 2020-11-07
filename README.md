# KataGoONNX
[KataGo](https://github.com/lightvector/KataGo) .bin network file to ONNX converter.

For the time being, only the following configuration is supported (essentially all >b10 networks from the g170 run):
```
{
    "version": 8,
    "support_japanese_rules": true,
    "use_fixup": true,
    "use_scoremean_as_lead": false
}
```
Some details about the output ONNX file:
* NCHW format (including input/output)
* inputs:
```
"input_binary": (-1, 22, y_size, x_size)
"input_global": (-1, 19)
```
* outputs:
```
"output_policy": (-1, y_size * x_size + 1)
"output_value": (-1, 3)
"output_miscvalue": (-1, 4)
"output_ownership": (-1, 1, y_size, x_size)
```
* Opset 10
## Requirements
* [PyTorch](https://pytorch.org/)
* [ONNXMLTools](https://github.com/onnx/onnxmltools) for FP16 quantization
## Usage
1. Download .zip file from https://d3dndmfyhecmj0.cloudfront.net/g170/neuralnets/index.html.
2. Unzip this .zip file and .bin.gz file. You will need both `model.bin` and `model.config.json` file.
3. Run convert.py:
```
python convert.py
    --model BIN_FILE_PATH
    --model-config CONFIG_JSON_FILE
    --output TARGET_OUTPUT_PATH
```
4. (Optional) Quantize to FP16 with quantize.py.
```
python quantize.py
    --input ONNX_FILE_PATH
    --output TARGET_OUTPUT_PATH
```
