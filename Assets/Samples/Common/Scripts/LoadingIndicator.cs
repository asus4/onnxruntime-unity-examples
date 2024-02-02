using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    public sealed class LoadingIndicator : MonoBehaviour
    {
        [SerializeField]
        Transform target;

        private void OnEnable()
        {
            if (target == null)
            {
                target = transform;
            }
        }

        private void Update()
        {
            target.Rotate(0, 0, 180 * Time.deltaTime);
        }
    }
}
