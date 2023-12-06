using System.IO;
using UnityEngine.Assertions;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
#if UNITY_IOS
using UnityEditor.iOS.Xcode;
using UnityEditor.iOS.Xcode.Extensions;
#endif // UNITY_IOS

namespace Microsoft.ML.OnnxRuntime.Editor
{
    public class OnnxPostProcessBuild : IPostprocessBuildWithReport
    {
        private const string PACKAGE_PATH = "Packages/com.github.asus4.onnxruntime";

        public int callbackOrder => 0;

        public void OnPostprocessBuild(BuildReport report)
        {
#if UNITY_IOS
            string pbxProjectPath = PBXProject.GetPBXProjectPath(report.summary.outputPath);
            PBXProject pbxProject = new();
            pbxProject.ReadFromFile(pbxProjectPath);

            // Copy XCFramework to in the "PROJECT/Libraries/onnxruntime.xcframework"
            string frameworkSrcPath = Path.Combine(PACKAGE_PATH, "Plugins/iOS~/onnxruntime.xcframework");
            string frameworkDstRelPath = "Libraries/onnxruntime.xcframework";
            string frameworkDstAbsPath = Path.Combine(report.summary.outputPath, frameworkDstRelPath);
            CopyDir(frameworkSrcPath, frameworkDstAbsPath);

            // Then add to Xcode project
            string frameworkGuid = pbxProject.AddFile(frameworkDstAbsPath, frameworkDstRelPath, PBXSourceTree.Source);
            string targetGuid = pbxProject.GetUnityMainTargetGuid();
            pbxProject.AddFileToEmbedFrameworks(targetGuid, frameworkGuid);

            pbxProject.WriteToFile(pbxProjectPath);
#endif // UNITY_IOS
        }

        private static void CopyDir(string srcPath, string dstPath)
        {
            srcPath = FileUtil.GetPhysicalPath(srcPath);
            Assert.IsTrue(Directory.Exists(srcPath), $"Framework not found at {srcPath}");

            if (Directory.Exists(dstPath))
            {
                FileUtil.DeleteFileOrDirectory(dstPath);
            }
            FileUtil.CopyFileOrDirectory(srcPath, dstPath);
        }
    }
}
