using System.Collections;
using System.Linq;
using System.IO;
using Microsoft.ML.OnnxRuntime.Examples;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.SceneManagement;

/// <summary>
/// Entry point of the sample app
/// </summary>
public sealed class IndexScene : MonoBehaviour
{
    [SerializeField]
    private RectTransform buttonContainer;

    [SerializeField]
    private GoToSceneButton buttonPrefab;

    [SerializeField]
    private string[] sceneNames;

    private void Awake()
    {
        Application.targetFrameRate = 60;
        Screen.sleepTimeout = SleepTimeout.NeverSleep;
    }

    private IEnumerator Start()
    {
        // Need the WebCam Authorization before using Camera on mobile devices
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        }

        ValidateSceneNames();

        // Add buttons
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
