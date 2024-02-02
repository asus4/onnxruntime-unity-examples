using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// Loads a scene when the button is clicked.
    /// </summary>
    [RequireComponent(typeof(Button))]
    public sealed class GoToSceneButton : MonoBehaviour
    {
        [Tooltip("The name of the scene to load.")]
        public string sceneName;

        private void OnEnable()
        {
            if (TryGetComponent(out Button button))
            {
                button.onClick.AddListener(OnClick);
            }
        }

        private void OnDisable()
        {
            if (TryGetComponent(out Button button))
            {
                button.onClick.RemoveListener(OnClick);
            }
        }

        private void OnClick()
        {
            SceneManager.LoadScene(sceneName);
        }
    }
}
