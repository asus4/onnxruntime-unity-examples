using UnityEngine;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;

namespace Microsoft.ML.OnnxRuntime.Editor
{
    /// <summary>
    /// Custom pre-process build for ONNX Runtime
    /// </summary>
    public class OrtPreProcessBuild : IPreprocessBuildWithReport
    {
        public int callbackOrder => 0;

        public void OnPreprocessBuild(BuildReport report)
        {
            switch (report.summary.platform)
            {
                case BuildTarget.iOS:
                    OrtBuildHelper.AddDefine(NamedBuildTarget.iOS, "__IOS__", "__ENABLE_COREML__");
                    break;
                case BuildTarget.Android:
                    OrtBuildHelper.RemoveDefine(NamedBuildTarget.Android, "__ENABLE_COREML__");
                    OrtBuildHelper.AddDefine(NamedBuildTarget.Android, "__ANDROID__");
                    break;
                case BuildTarget.StandaloneOSX:
                    OrtBuildHelper.AddDefine(NamedBuildTarget.Standalone, "__ENABLE_COREML__");
                    break;
                // TODO: Add support for other platforms
                default:
                    Debug.Log("OnnxPreProcessBuild.OnPreprocessBuild for target " + report.summary.platform + " is not supported");
                    break;
            }
        }
    }
}
