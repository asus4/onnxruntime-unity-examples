using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Simple remote file download and cache system.
    /// Not for production use.
    /// </summary>
    [Serializable]
    public class RemoteFile
    {
        public enum DownloadLocation
        {
            Persistent,
            Cache,
        }

        public string url;
        public DownloadLocation downloadLocation;

        public string LocalPath
        {
            get
            {
                string dir = downloadLocation switch
                {
                    DownloadLocation.Persistent => Application.persistentDataPath,
                    DownloadLocation.Cache => Application.temporaryCachePath,
                    _ => throw new Exception($"Unknown download location {downloadLocation}"),
                };
                // make hash from url
                string ext = GetExtension(url);
                string fileName = $"{url.GetHashCode():X8}{ext}";
                return Path.Combine(dir, fileName);
            }
        }

        public RemoteFile() { }

        public RemoteFile(string url, DownloadLocation location = DownloadLocation.Persistent)
        {
            this.url = url;
            downloadLocation = location;
        }

        public async Task<byte[]> Load()
        {
            string localPath = LocalPath;

            if (File.Exists(localPath))
            {
                Log($"Cache Loading file from local: {localPath}");
                return await LoadFromLocal(localPath);
            }
            else
            {
                Log($"Cache not found at {localPath}. Loading from: {url}");
                return await LoadFromRemote(url, localPath);
            }
        }

        // Need to use WebRequest for local file download in Android
        private static async Task<byte[]> LoadFromLocal(string localPath)
        {
            if (!localPath.StartsWith("file:/"))
            {
                localPath = $"file://{localPath}";
            }
            using var request = UnityWebRequest.Get(localPath);

            var operation = request.SendWebRequest();
            while (!operation.isDone)
            {
                await Task.Yield();
            }

            if (request.result != UnityWebRequest.Result.Success)
            {
                throw new Exception($"Failed to download {localPath}: {request.error}");
            }

            return request.downloadHandler.data;
        }

        private static async Task<byte[]> LoadFromRemote(string remotePath, string localPath)
        {
            using var handler = new DownloadHandlerFile(localPath);
            using var request = new UnityWebRequest(remotePath, "GET", handler, null);

            var operation = request.SendWebRequest();
            while (!operation.isDone)
            {
                await Task.Yield();
            }

            if (request.result != UnityWebRequest.Result.Success)
            {
                throw new Exception($"Failed to download {remotePath}: {request.error}");
            }

            return File.ReadAllBytes(localPath);
        }

        private static string GetExtension(string url)
        {
            string ext = Path.GetExtension(url);
            if (ext.Contains('?'))
            {
                ext = ext[..ext.IndexOf('?')];
            }
            return ext;
        }

        [Conditional("DEVELOPMENT_BUILD"), Conditional("UNITY_EDITOR")]
        private static void Log(string message)
        {
            UnityEngine.Debug.Log(message);
        }
    }
}
