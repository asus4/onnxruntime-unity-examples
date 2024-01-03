# ONNX Runtime examples for Unity

[![npm](https://img.shields.io/npm/v/com.github.asus4.onnxruntime?label=npm)](https://www.npmjs.com/package/com.github.asus4.onnxruntime)

Examples of [ONNX Runtime Unity Plugin](https://github.com/asus4/onnxruntime-unity-examples).

<https://github.com/asus4/onnxruntime-unity-examples/assets/357497/96ed9913-41b7-401d-a634-f0e2de4fc3c7>

## Tested environment

- Unity: 2022.3.16f1 (LTS)
- ONNX Runtime: 1.16.3
- macOS, iOS, Android

## How to Run

### Test all examples

Pull this repository with **[Git-LFS](https://git-lfs.com/)**

### Install pre-built ONNX Runtime via Unity Package Manager (UPM)

Add the following `scopedRegistries` and `dependencies` in `Packages/manifest.json`.

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
    "com.github.asus4.onnxruntime": "0.1.3",
    // (Optional) Utilities for Unity
    "com.github.asus4.onnxruntime.unity": "0.1.3",
    // (Optional) GPU provider extensions for Windows/Linux (each 300mb+)
    // CPU for Windows/Linux is included in core library
    "com.github.asus4.onnxruntime.win-x64-gpu": "0.1.3",
    "com.github.asus4.onnxruntime.linux-x64-gpu": "0.1.3",
    ... other dependencies
  }
```

## How to convert Onnx to Ort format

On the mobile platform, the *.onnx model is not recommended, although it's supported on the desktop. Convert the Onnx model to Ort format.

Please refer to the [ORT model format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html) for more details.

**TL;DR;**

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
