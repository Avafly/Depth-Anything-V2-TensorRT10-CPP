# Depth-Anything-V2-TensorRT10-CPP

This project provides a C++ TensorRT implementation of [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) with dynamic shape support.

## Demo

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Avafly/ImageHostingService@master/uPic/Beatles_depth.png">
</p>

## Export

Models can be exported via the [Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX?tab=readme-ov-file#export-example) repository or downloaded from the [Releases](https://github.com/Avafly/Depth-Anything-V2-TensorRT10-CPP/releases/tag/models) page.

## Quick Start

### Usage

```bash
./main [OPTIONS]

OPTIONS:
  -h,     --help              Print this help message and exit 
  -m,     --model TEXT        Model path 
  -i,     --input TEXT        Image or folder path 
  -o,     --output TEXT [.]   Output directory 
  -s,     --size INT [518]    Target size 
  -b,     --batch INT [0]     Max batch size 
  -w,     --warmup INT [0]    Warmup rounds
```

### Examples

```bash
# single image inference
./main -m vitb.engine -i image.jpg

# batch inference
./main -m vitb.engine -i images/

# onnx model is also acceptable
./main -m vitb.onnx -i image.jpg
```

## Dependencies

|      Library      | Version |
| :---------------: | :-----: |
|     TensorRT      |  10.11  |
|       CUDA        |  12.9   |
|      OpenCV       | 4.12.0  |
| CLI11 (included)  |  2.6.1  |
| spdlog (included) | 1.17.0  |