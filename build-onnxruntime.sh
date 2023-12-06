
cd ../onnxruntime

# Build for iOS simulator
./build.sh --config MinSizeRel --use_xcode --ios --ios_sysroot iphonesimulator --osx_arch x86_64 --apple_deploy_target 15 --use_coreml