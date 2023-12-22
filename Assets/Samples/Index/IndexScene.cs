using Microsoft.ML.OnnxRuntime.Examples;
using UnityEngine;

public class IndexScene : MonoBehaviour
{
    [SerializeField]
    private RectTransform buttonContainer;

    [SerializeField]
    private GoToSceneButton buttonPrefab;

    [SerializeField]
    private string[] sceneNames;


    private void Start()
    {
        foreach (var sceneName in sceneNames)
        {
            var button = Instantiate(buttonPrefab, buttonContainer);
            button.sceneName = sceneName;

            var text = button.GetComponentInChildren<TMPro.TMP_Text>();
            if (text != null)
            {
                text.text = sceneName;
            }
        }
    }
}
