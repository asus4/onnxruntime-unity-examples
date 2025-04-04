using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;

#if UNITY_IOS
using UnityEditor.iOS.Xcode;
#endif // UNITY_IOS

namespace Microsoft.ML.OnnxRuntime.Examples
{
    public class CustomPostBuildProcess : IPostprocessBuildWithReport
    {
        public int callbackOrder => 0;

        public void OnPostprocessBuild(BuildReport report)
        {
            switch (report.summary.platform)
            {
                case BuildTarget.iOS:
                    SettingForIOS(report);
                    break;
            }
        }

        static void SettingForIOS(BuildReport report)
        {
#if UNITY_IOS
            var plist = new PlistDocument();
            string plistPath = Path.Combine(report.summary.outputPath, "Info.plist");
            plist.ReadFromFile(plistPath);

            PlistElementDict rootDict = plist.root;
            // Access to the documents directory to load big LLM models
            rootDict.SetBoolean("UIFileSharingEnabled", true);
            rootDict.SetBoolean("LSSupportsOpeningDocumentsInPlace", true);
            plist.WriteToFile(plistPath);
#endif // UNITY_IOS
        }
    }
}
