using System.IO;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;
#if UNITY_IOS
using UnityEditor.iOS.Xcode;
using UnityEditor.iOS.Xcode.Extensions;
using UnityEngine.Assertions;
#endif // UNITY_IOS

namespace Microsoft.ML.OnnxRuntime.Editor
{
    public class OnnxPreProcessBuild : IPreprocessBuildWithReport
    {
        private const string PACKAGE_PATH = "Packages/com.github.asus4.onnxruntime";

        public int callbackOrder => 0;

        public void OnPreprocessBuild(BuildReport report)
        {

            Debug.Log("OnnxPreProcessBuild.OnPreprocessBuild for target " + report.summary.platform + " at path " + report.summary.outputPath);
        }
    }
}
