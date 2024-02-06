# ONNX Runtime examples for Unity

[![upm](https://img.shields.io/npm/v/com.github.asus4.onnxruntime?label=upm)](https://www.npmjs.com/package/com.github.asus4.onnxruntime)

Examples of [ONNX Runtime Unity Plugin](https://github.com/asus4/onnxruntime-unity).

<https://github.com/asus4/onnxruntime-unity-examples/assets/357497/96ed9913-41b7-401d-a634-f0e2de4fc3c7>

## Tested environment

- Unity: 2022.3.16f1 (LTS)
- ONNX Runtime: 1.17.0
- macOS, iOS, Android, Windows, Linux.
  - Complete List for [:link: Supported Execution Providers](https://github.com/asus4/onnxruntime-unity?tab=readme-ov-file#execution-providers)

## How to Run

### Try all examples

> [!IMPORTANT]  
> Install [Git-LFS](https://git-lfs.github.com/) to try this repository.

ONNX examples are available in the `Assets/Examples` folder. Pull the repository and open the project in Unity.

Following demos are available:

- [MobileOne](https://github.com/apple/ml-mobileone): Image classification
- [Yolox](https://github.com/Megvii-BaseDetection/YOLOX): Object detection
- [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam/): Object Segmentation

### Install pre-built ONNX Runtime via Unity Package Manager (UPM)

Add the following `scopedRegistries` and `dependencies` in `Packages/manifest.json` to install the ONNX Runtime plugin into your project.  
Check out the [asus4/onnxruntime-unity](https://github.com/asus4/onnxruntime-unity) repository for more details.

```json
  "scopedRegistries": [
    {
      "name": "NPM",
      "url": "https://registry.npmjs.com",
      "scopes": [
        "com.github.asus4"
      ]
    }
  ]
  "dependencies": {
    // Core library
    "com.github.asus4.onnxruntime": "0.1.12",
    // (Optional) Utilities for Unity
    "com.github.asus4.onnxruntime.unity": "0.1.12",
    ... other dependencies
  }
```

## How to convert Onnx to Ort format

On the mobile platform, the *.onnx model is not recommended, although it's supported. Convert the Onnx model to Ort format.

Please refer to the [ORT model format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html) for more details.

```sh
# Recommend using python virtual environment
pip install onnx
pip install onnxruntime

# In general,
# Use --optimization_style Runtime, when running on mobile GPU
# Use --optimization_style Fixed, when running on mobile CPU
python -m onnxruntime.tools.convert_onnx_models_to_ort your_onnx_file.onnx --optimization_style Runtime
```

## Acknowledgements

- [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo): Some ORT models are converted from this repository.
