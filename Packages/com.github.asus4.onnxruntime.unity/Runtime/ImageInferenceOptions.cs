
using System;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Unity
{
    public enum AspectMode
    {
        /// <summary>
        /// Resizes the image without keeping the aspect ratio.
        /// </summary>
        None = 0,
        /// <summary>
        /// Resizes the image to contain full area and padded black pixels.
        /// </summary>
        Fit = 1,
        /// <summary>
        /// Trims the image to keep aspect ratio.
        /// </summary>
        Fill = 2,
    }

    [System.Serializable]
    public class ImageInferenceOptions
    {
        public AspectMode aspectMode = AspectMode.Fit;
        public bool useGPU = false;

        public SessionOptions CreateSessionOptions()
        {
            SessionOptions options = new();
            if (useGPU)
            {
                try
                {
                    AppendExecutionProvider(Application.platform, options);
                }
                catch (Exception e)
                {
                    Debug.LogError($"Failed to setup GPU: {e.Message}");
                }
            }

            return options;
        }

        public void AppendExecutionProvider(RuntimePlatform platform, SessionOptions options)
        {
            switch (platform)
            {
                case RuntimePlatform.OSXEditor:
                case RuntimePlatform.OSXPlayer:
                case RuntimePlatform.OSXServer:
                case RuntimePlatform.IPhonePlayer:
                    options.AppendExecutionProvider_CoreML(
                        CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                    Debug.Log("CoreML is enabled");
                    break;
                case RuntimePlatform.Android:
                    options.AppendExecutionProvider_Nnapi(
                        NnapiFlags.NNAPI_FLAG_USE_FP16);
                    Debug.Log("NNAPI is enabled");
                    break;
                case RuntimePlatform.WindowsEditor:
                case RuntimePlatform.WindowsPlayer:
                case RuntimePlatform.WindowsServer:
                case RuntimePlatform.LinuxEditor:
                case RuntimePlatform.LinuxPlayer:
                case RuntimePlatform.LinuxServer:
                    options.AppendExecutionProvider_Tensorrt();
                    // options.AppendExecutionProvider_CUDA();
                    Debug.Log("TensorRT is enabled");
                    break;
                // TODO: Add WebGL build
                default:
                    Debug.LogWarning($"Execution provider is not supported on {platform}");
                    break;
            }
        }
    }
}
