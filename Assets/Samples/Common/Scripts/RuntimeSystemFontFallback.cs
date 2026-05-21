using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    public static class RuntimeSystemFontFallback
    {
        static readonly string[] FontFamilies =
        {
            "Noto Sans",
            "Arial",
            "Arial Unicode MS",
            "Segoe UI",
            "DejaVu Sans",
            "FreeSans",

            "Noto Sans CJK JP",
            "Noto Sans JP",
            "Hiragino Sans",
            "Hiragino Kaku Gothic ProN",
            "Yu Gothic UI",
            "Yu Gothic",
            "Meiryo",

            "Noto Sans CJK KR",
            "Noto Sans KR",
            "Apple SD Gothic Neo",
            "Malgun Gothic",

            "Noto Sans CJK SC",
            "Noto Sans CJK TC",
            "PingFang SC",
            "PingFang TC",
            "Microsoft YaHei UI",
            "Microsoft JhengHei UI",

            "Noto Sans Arabic",
            "Geeza Pro",

            "Noto Sans Devanagari",
            "Kohinoor Devanagari",
            "Devanagari Sangam MN",
            "Nirmala UI",

            "Roboto",
            "Droid Sans",
        };

        static bool installed;

        public static void Install()
        {
            if (installed)
            {
                return;
            }

            installed = true;

            if (TMP_Settings.instance == null)
            {
                Debug.LogWarning("TMP Settings are not available. System font fallbacks were not installed.");
                return;
            }

            TMP_Settings.fallbackFontAssets ??= new List<TMP_FontAsset>();
            var fallbacks = TMP_Settings.fallbackFontAssets;

            // Remove empty/null entries in case there are any
            fallbacks.RemoveAll(fontAsset => fontAsset == null);

            foreach (string fontFamily in FontFamilies)
            {
                string assetName = $"Runtime Fallback Font - {fontFamily}";
                if (fallbacks.Any(fontAsset => fontAsset != null && fontAsset.name == assetName))
                {
                    continue;
                }

                var fontAsset = TMP_FontAsset.CreateFontAsset(fontFamily, "Regular", 90);
                if (fontAsset == null)
                {
                    continue;
                }

                fontAsset.name = assetName;
                fontAsset.isMultiAtlasTexturesEnabled = true;
                fallbacks.Add(fontAsset);
                Debug.Log($"Installed system font fallback: {fontFamily}");
            }
        }
    }
}
