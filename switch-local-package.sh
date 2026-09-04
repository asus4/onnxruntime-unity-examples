#!/bin/bash -e

# This script switches between local and UPM package versions in the Unity manifest.json file.
# Useful for developing OnnxRuntime Unity packages locally.

MANIFEST="Packages/manifest.json"

function print_usage() {
    echo "Usage: $0 <version> OR $0 local"
    echo "Example: $0 1.2.3"
    echo "Example: $0 local"
}

# sed -i compatible with both BSD and GNU sed
function sed_inplace() {
    sed -i.bak "$1" "$MANIFEST" && rm -f "$MANIFEST.bak"
}

function switch_to_local() {
    sed_inplace 's|"'"$1"'": "[^"]*"|"'"$1"'": "file:../../onnxruntime-unity/'"$1"'"|'
}

function switch_to_npm() {
    sed_inplace 's|"'"$1"'": "[^"]*"|"'"$1"'": "'"$2"'"|'
}

# Validate input format
if [[ $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    switch_to_npm "com.github.asus4.onnxruntime" $1
    switch_to_npm "com.github.asus4.onnxruntime.unity" $1
    switch_to_npm "com.github.asus4.onnxruntime-extensions" $1
    switch_to_npm "com.github.asus4.onnxruntime-genai" $1
    switch_to_npm "com.github.asus4.onnxruntime.win-x64-gpu" $1
    switch_to_npm "com.github.asus4.onnxruntime.linux-x64-gpu" $1
elif [[ $1 == "local" ]]; then
    switch_to_local "com.github.asus4.onnxruntime"
    switch_to_local "com.github.asus4.onnxruntime.unity"
    switch_to_local "com.github.asus4.onnxruntime-extensions"
    switch_to_local "com.github.asus4.onnxruntime-genai"
    switch_to_local "com.github.asus4.onnxruntime.win-x64-gpu"
    switch_to_local "com.github.asus4.onnxruntime.linux-x64-gpu"
else
    print_usage
    exit 1
fi

echo "Done."
exit 0
