using System.Linq;
using UnityEngine;
using UnityEditor;
using UnityEditor.Build;

namespace Microsoft.ML.OnnxRuntime.Editor
{
    public static class OrtBuildHelper
    {
        public const string PACKAGE_PATH = "Packages/com.github.asus4.onnxruntime";

        public static void AddDefine(NamedBuildTarget target, params string[] adds)
        {
            PlayerSettings.GetScriptingDefineSymbols(target, out string[] defines);
            var newDefines = defines.Concat(adds).Distinct().ToArray();
            PlayerSettings.SetScriptingDefineSymbols(target, newDefines);
        }

        public static void RemoveDefine(NamedBuildTarget target, params string[] removes)
        {
            PlayerSettings.GetScriptingDefineSymbols(target, out string[] defines);
            var newDefines = defines.Except(removes).ToArray();
            PlayerSettings.SetScriptingDefineSymbols(target, newDefines);
        }

        [InitializeOnLoadMethod]
        private static void Initialize()
        {
            RuntimePlatform platform = Application.platform;
            BuildTarget buildTarget = EditorUserBuildSettings.activeBuildTarget;

            Debug.Log($"Initialize platform:{platform}, target{buildTarget}");
            if (platform == RuntimePlatform.OSXEditor)
            {
                switch (buildTarget)
                {
                    case BuildTarget.StandaloneOSX:
                        AddDefine(NamedBuildTarget.Standalone, "__ENABLE_COREML__");
                        break;
                    case BuildTarget.iOS:
                        AddDefine(NamedBuildTarget.iOS, "__ENABLE_COREML__");
                        break;
                    case BuildTarget.Android:
                        AddDefine(NamedBuildTarget.Android, "__ENABLE_COREML__");
                        break;
                    default:
                        // Nothing
                        break;
                }
            }
        }
    }
}
