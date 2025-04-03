using System;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.ML.OnnxRuntimeGenAI;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    public sealed class Phi4MultiModal : IDisposable
    {
        readonly OgaHandle ogaHandle;
        // readonly Config config;
        // Model model;

        bool disposed = false;

        private Phi4MultiModal()
        {
            ogaHandle = new OgaHandle();
            // config = new Config("aaaaa");
            // model = new Model(config);
        }

        ~Phi4MultiModal()
        {
            Dispose(false);
        }

        public static async Awaitable<Phi4MultiModal> CreateAsync(CancellationToken cancellationToken)
        {
            // Simulate async initialization
            await Awaitable.BackgroundThreadAsync();
            cancellationToken.ThrowIfCancellationRequested();

            Phi4MultiModal instance = null;
            try
            {
                instance = new Phi4MultiModal();
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to create Phi4MultiModal: {ex.Message}");
                throw ex;
            }

            await Awaitable.MainThreadAsync();
            cancellationToken.ThrowIfCancellationRequested();

            return instance;
        }

        public void Dispose()
        {
            Dispose(true);
            Debug.Log($"Phi4MultiModal disposed");
        }

        void Dispose(bool disposing)
        {
            if (disposed)
            {
                return;
            }
            if (disposing)
            {
                // config?.Dispose();
                ogaHandle?.Dispose();
                // model?.Dispose();
            }
            disposed = true;
        }

    }
}
