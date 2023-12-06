using UnityEngine;
using Microsoft.ML.OnnxRuntime;

public sealed class Test : MonoBehaviour
{
    [SerializeField]
    private TextAsset model;

    private void Start()
    {
        using var session = new InferenceSession(model.bytes);

        Debug.Log($"Session created: {session}");

        foreach (var meta in session.InputMetadata)
        {
            Debug.Log($"Input name: {meta.Key} shape: {string.Join(",", meta.Value.Dimensions)}, type: {meta.Value.ElementType}");
        }

        foreach (var meta in session.OutputMetadata)
        {
            Debug.Log($"Output name: {meta.Key} shape: {string.Join(",", meta.Value.Dimensions)}, type: {meta.Value.ElementType}");
        }
    }
}
