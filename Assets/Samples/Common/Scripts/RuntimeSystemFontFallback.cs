using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    public static class RuntimeSystemFontFallback
    {
        readonly struct FontEntry
        {
            public readonly string Family;
            public readonly string Style;

            public FontEntry(string family, string style)
            {
                Family = family;
                Style = style;
            }

            public string AssetName => $"Runtime Fallback Font - {Family} {Style}";

            public bool TryCreateFontAsset(out TMP_FontAsset fontAsset)
            {
                fontAsset = TMP_FontAsset.CreateFontAsset(Family, Style, 90);
                if (fontAsset != null)
                {
                    fontAsset.name = AssetName;
                    fontAsset.isMultiAtlasTexturesEnabled = true;
                    return true;
                }
                return false;
            }
        }

        // iOS bundled fonts.
        // https://developer.apple.com/fonts/system-fonts/
        static readonly FontEntry[] IOSFonts =
        {
            new("Helvetica Neue", "Regular"),
            new("Arial", "Regular"),

            new("Hiragino Sans", "W3"),
            new("Apple SD Gothic Neo", "Regular"),

            new("PingFang SC", "Regular"),
            new("PingFang TC", "Regular"),

            new("Geeza Pro", "Regular"),

            new("Kohinoor Devanagari", "Regular"),
            new("Devanagari Sangam MN", "Regular"),
        };

        // macOS bundled fonts.
        // https://developer.apple.com/fonts/system-fonts/
        static readonly FontEntry[] MacOSFonts =
        {
            new("Helvetica Neue", "Regular"),
            new("Arial", "Regular"),

            new("Hiragino Sans", "W3"),
            new("Apple SD Gothic Neo", "Regular"),

            new("PingFang SC", "Regular"),
            new("PingFang TC", "Regular"),

            new("Geeza Pro", "Regular"),

            new("Kohinoor Devanagari", "Regular"),
            new("Devanagari Sangam MN", "Regular"),
        };

        // Android (7.0+) bundled fonts.
        // https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/data/fonts/fonts.xml
        static readonly FontEntry[] AndroidFonts =
        {
            new("Roboto", "Regular"),

            new("Noto Sans CJK JP", "Regular"),
            new("Noto Sans CJK KR", "Regular"),
            new("Noto Sans CJK SC", "Regular"),
            new("Noto Sans CJK TC", "Regular"),

            new("Noto Sans Arabic", "Regular"),
            new("Noto Sans Devanagari", "Regular"),
        };

        // Windows bundled fonts.
        // Source: https://learn.microsoft.com/en-us/typography/fonts/windows_11_font_list
        static readonly FontEntry[] WindowsFonts =
        {
            new("Segoe UI", "Regular"),
            new("Arial", "Regular"),

            new("Yu Gothic UI", "Regular"),
            new("Yu Gothic", "Regular"),
            new("Meiryo", "Regular"),

            new("Malgun Gothic", "Regular"),

            new("Microsoft YaHei UI", "Regular"),
            new("Microsoft JhengHei UI", "Regular"),

            new("Nirmala UI", "Regular"),
        };

        // Linux fonts commonly available via fontconfig (DejaVu / Noto).
        static readonly FontEntry[] LinuxFonts =
        {
            new("DejaVu Sans", "Book"),
            new("Noto Sans", "Regular"),

            new("Noto Sans CJK JP", "Regular"),
            new("Noto Sans CJK KR", "Regular"),
            new("Noto Sans CJK SC", "Regular"),
            new("Noto Sans CJK TC", "Regular"),

            new("Noto Sans Arabic", "Regular"),
            new("Noto Sans Devanagari", "Regular"),
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

            foreach (FontEntry entry in GetFontEntries())
            {
                string assetName = entry.AssetName;
                if (fallbacks.Any(fontAsset => fontAsset != null && fontAsset.name == assetName))
                {
                    continue;
                }

                if (entry.TryCreateFontAsset(out var fontAsset))
                {
                    fallbacks.Add(fontAsset);
                    Debug.Log($"Installed system font fallback: {entry.Family} {entry.Style}");
                }
            }
        }

        static FontEntry[] GetFontEntries()
        {
            return Application.platform switch
            {
                RuntimePlatform.IPhonePlayer => IOSFonts,
                RuntimePlatform.OSXPlayer or RuntimePlatform.OSXEditor => MacOSFonts,
                RuntimePlatform.Android => AndroidFonts,
                RuntimePlatform.WindowsPlayer or RuntimePlatform.WindowsEditor => WindowsFonts,
                RuntimePlatform.LinuxPlayer or RuntimePlatform.LinuxEditor => LinuxFonts,
                _ => MacOSFonts,
            };
        }
    }
}
