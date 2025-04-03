using Microsoft.ML.OnnxRuntime.Examples;
using UnityEngine;

public class Phi4MultiModalSample : MonoBehaviour
{
    Phi4MultiModal inference;

    async Awaitable Start()
    {
        inference = await Phi4MultiModal.CreateAsync(destroyCancellationToken);
    }

    void OnDestroy()
    {
        inference?.Dispose();
    }
}
