using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime;

/// <summary>
/// Unit test for each execution provider. run this on each device.
/// </summary>
public class UnitTestAddSample : MonoBehaviour
{
    [SerializeField]
    private OrtAsset model;

    private void Start()
    {
        Assert.IsNotNull(model, "Please specify the model file");

        // Test on all platforms
        TestDefaultCPU();

        // XNNPACK: Android, iOS
        if (IsPlatform(RuntimePlatform.Android, RuntimePlatform.IPhonePlayer))
        {
            TestXNNPack();
        }

        // CoreML: iOS, macOS
        if (IsPlatform(RuntimePlatform.IPhonePlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer))
        {
            TestCoreML();
        }

        // NNAPI: Android
        if (IsPlatform(RuntimePlatform.Android))
        {
            TestNNAPI();
        }

        // ONNX Runtime Extensions: macOS, Linux x64, Windows x64, iOS, Android
        TestOrtExtensions();
    }

    private void OnDestroy()
    {

    }

    private void TestDefaultCPU()
    {
        Debug.Log("TestDefaultCPU");
        using SessionOptions options = new();
        RunSession(model.bytes, options);
    }

    private void TestXNNPack()
    {
        Debug.Log("TestXNNPack");
        // Recommended configuration for XNNPACK
        // https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html#recommended-configuration
        using SessionOptions options = new();
        options.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        int threads = Math.Clamp(SystemInfo.processorCount, 1, 4);
        options.AppendExecutionProvider("XNNPACK", new Dictionary<string, string>()
            {
                { "intra_op_num_threads", threads.ToString()},
            });
        options.IntraOpNumThreads = 1;
        RunSession(model.bytes, options);
    }

    private void TestCoreML()
    {
        Debug.Log("TestCoreML");
        using SessionOptions options = new();
        options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
        RunSession(model.bytes, options);
    }

    private void TestNNAPI()
    {
        Debug.Log("TestNNAPI");
        using SessionOptions options = new();
        options.AppendExecutionProvider_Nnapi();
        RunSession(model.bytes, options);
    }

    private void TestOrtExtensions()
    {
        Debug.Log("TestOrtExtensions");
        using SessionOptions options = new();
        try
        {
            options.RegisterOrtExtensions();
        }
        catch (Exception e)
        {
            Debug.LogError(e.Message);
        }
        Debug.Log("Successfully registered ONNX Runtime Extensions");
    }

    private static void RunSession(byte[] modelBytes, SessionOptions options)
    {
        using InferenceSession session = new(modelBytes, options);

        var inputNames = new List<string>() { "A", "B" };
        var inputTensors = new List<OrtValue>()
        {
            OrtValue.CreateTensorValueFromMemory( new float[] { 3 }, new long[] { 1 }),
            OrtValue.CreateTensorValueFromMemory( new float[] { 4 }, new long[] { 1 }),
        };
        var outputNames = new List<string>() { "C" };
        var outputTensors = new List<OrtValue>()
        {
            OrtValue.CreateTensorValueFromMemory( new float[] { 0 }, new long[] { 1 }),
        };

        try
        {
            session.Run(null, inputNames, inputTensors, outputNames, outputTensors);
            var output = outputTensors[0].GetTensorDataAsSpan<float>();
            Debug.Log($"A + B = C, 3 + 4 = {output[0]}");
            Debug.Assert(output[0] == 7, "The result should be 7");
        }
        catch (Exception e)
        {
            Debug.LogError(e.Message);
        }
        finally
        {
            foreach (var tensor in inputTensors)
            {
                tensor.Dispose();
            }
            foreach (var tensor in outputTensors)
            {
                tensor.Dispose();
            }
        }
    }

    private static bool IsPlatform(params RuntimePlatform[] platforms)
    {
        var currentPlatform = Application.platform;
        return platforms.Any(platform => platform == currentPlatform);
    }
}
