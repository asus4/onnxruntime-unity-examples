using System.Text;
using UnityEngine;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    [RequireComponent(typeof(TMPro.TMP_Text))]
    public class FpsText : MonoBehaviour
    {
        private TMPro.TMP_Text text;
        private readonly StringBuilder sb = new();

        private void Start()
        {
            text = GetComponent<TMPro.TMP_Text>();
        }

        private void Update()
        {
            sb.Clear();
            sb.Append("FPS: ");
            float fps = 1f / Time.smoothDeltaTime;
            sb.Append(fps.ToString("F1"));
            text.SetText(sb);
        }
    }
}
