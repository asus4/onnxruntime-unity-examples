# ONNX Runtime examples for Unity

Some examples and pre-built ONNX Runtime libraries for Unity.

## Tested environment

- Unity 2020.3.12f1 (LTS)
- macOS, iOS, Android

## Install via Unity Package Manager

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
    "com.github.asus4.onnxruntime": "0.1.0",
    // Utilities for Unity
    "com.github.asus4.onnxruntime.unity": "0.1.0",
    ... other dependencies
  }
```

## How to convert onnx to ort format

On the mobile platform, the *.onnx model is not supported, while it's supported on the desktop platform. Please convert the onnx model to ort format.

Please refer to the [ORT model format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html).

__TL;DR;__

```sh
# Recommend using python virtual environment

pip install onnx
pip install onnxruntime

python -m onnxruntime.tools.convert_onnx_models_to_ort mobileone_s4_224x224.onnx
```

## Links for libraries

- [macOS](https://github.com/microsoft/onnxruntime/releases/)
- [Android](https://central.sonatype.com/artifact/com.microsoft.onnxruntime/onnxruntime-mobile/versions)
- [iOS](https://github.com/CocoaPods/Specs/tree/master/Specs/3/d/f/onnxruntime-mobile-c)
