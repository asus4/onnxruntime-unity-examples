using System.Linq;
using System.IO;
using Microsoft.ML.OnnxRuntime.Examples;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.SceneManagement;

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
        ValidateSceneNames();

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

    private void ValidateSceneNames()
    {
        int count = SceneManager.sceneCountInBuildSettings;
        // Check all scene names are valid
        var sceneNamesInBuild = Enumerable.Range(0, count)
            .Select(i => SceneUtility.GetScenePathByBuildIndex(i))
            .Select(path => Path.GetFileNameWithoutExtension(path));
        var notContains = sceneNames.Except(sceneNamesInBuild);

        Assert.AreEqual(0, notContains.Count(),
            $"The following scene names are not found in the build settings: {string.Join(", ", notContains)}");
    }
}
