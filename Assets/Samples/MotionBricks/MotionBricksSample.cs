using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime.Examples;
using Microsoft.ML.OnnxRuntime.Unity;
using TMPro;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;

/// <summary>
/// Click-to-move demo of MotionBricks (https://github.com/NVlabs/motionbricks).
/// Click or tap the ground to walk there. Drag before releasing to choose the final facing.
///
/// Each MotionBricks call generates up to 1.9 s of motion from the four latest poses toward a target.
/// Far from the destination, the target is a waypoint ahead with a position, heading and walking speed
/// but no body pose, so the model chooses a walk. Near the destination, the target is the idle pose
/// placed there, so the model stops in it. The next motion is generated on a worker thread while the
/// current one plays.
/// </summary>
public sealed class MotionBricksSample : MonoBehaviour
{
    // The ONNX model (758 MB) from the original repository, pinned to a commit.
    // It is licensed under the NVIDIA Open Model Agreement.
    const string ModelUrl = "https://media.githubusercontent.com/media/NVlabs/motionbricks/"
        + "54fc563aeaaf819a753d72cc9fdc6713989cdd43/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx";

    // Waypoints move at walking speed (m/s), this many seconds ahead.
    const float WalkSpeed = 1.3f;
    const float LookAheadSeconds = 1f;
    // Inside this distance (m), the character walks to the idle pose and stops.
    const float StopRadius = 1.6f;
    // From standstill, generating straight to an idle pose further than this (m) makes the
    // character hop, so it starts with a walk.
    const float MinWalkDistance = 0.6f;

    [SerializeField]
    SomaSkeletonView characterView;

    [SerializeField]
    SomaSkeletonView goalView;

    [SerializeField]
    Camera mainCamera;

    [SerializeField]
    Vector3 cameraOffset = new(0f, 3.2f, -4.8f);

    [SerializeField]
    TMP_Text statusLabel;

    [SerializeField]
    GameObject loadingIndicator;

    MotionBricks model;
    double inferenceMilliseconds;

    // The current segment plays while the next one is generated.
    MotionSegment current = new();
    MotionSegment next = new();
    int currentStart; // Frame number of current.Frames[0], at 30 Hz
    bool currentIsWalk;
    double frameTime;
    Task generation;
    int nextStart;
    bool nextIsWalk;

    readonly SomaPose pose = new();
    readonly SomaPose goal = new();
    readonly SomaPose[] context = new SomaPose[4];
    readonly SomaPose[] targets = new SomaPose[4];
    bool hasGoal;
    bool goalChanged;

    bool dragging;
    Vector3 pressPoint;
    Vector3? dragFacing;
    readonly SomaPose preview = new();
    readonly List<RaycastResult> raycastResults = new();
    Vector3 cameraVelocity;

    async Awaitable Start()
    {
        for (int i = 0; i < 4; i++)
        {
            context[i] = new SomaPose();
            targets[i] = new SomaPose();
        }
        // Start standing, facing the camera (Unity -Z is SOMA +Z, so turn around).
        current.Frames[0].SetIdle(float2.zero, math.PI);
        current.FrameCount = 1;

        loadingIndicator.SetActive(true);
        var cancellationToken = destroyCancellationToken;
        try
        {
            var remoteFile = new RemoteFile(ModelUrl);
            remoteFile.OnDownloadProgress += p => statusLabel.SetText($"Downloading the model... {p:P0}");
            string path = await remoteFile.EnsureLocal(cancellationToken);

            statusLabel.SetText("Loading the model...");
            await Awaitable.BackgroundThreadAsync();
            var newModel = new MotionBricks(path);
            // The first calls are slower, and the last one tells how far ahead to generate.
            var idle = new SomaPose();
            idle.SetIdle(float2.zero, 0);
            var segment = new MotionSegment();
            for (int i = 0; i < 2; i++)
            {
                newModel.Generate(new[] { idle, idle, idle, idle }, new[] { idle }, true, 6, 16, segment);
            }
            await Awaitable.MainThreadAsync();
            if (cancellationToken.IsCancellationRequested)
            {
                newModel.Dispose();
                return;
            }

            model = newModel;
            inferenceMilliseconds = segment.InferenceMilliseconds;
            UpdateStatus();
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            await Awaitable.MainThreadAsync();
            statusLabel.SetText($"Failed to load the model: {ex.Message}");
            Debug.LogException(ex);
        }
        finally
        {
            if (loadingIndicator != null)
            {
                loadingIndicator.SetActive(false);
            }
        }
    }

    void OnDestroy()
    {
        WaitForGeneration();
        model?.Dispose();
    }

    void Update()
    {
        HandlePointer();

        frameTime += Time.deltaTime * MotionBricks.SampleRate;
        int frame = (int)frameTime;

        // Switch to the next segment at its first frame.
        if (generation != null && generation.IsCompleted && frame >= nextStart)
        {
            if (generation.IsFaulted)
            {
                Debug.LogException(generation.Exception);
                hasGoal = false;
            }
            else
            {
                (current, next) = (next, current);
                currentStart = nextStart;
                currentIsWalk = nextIsWalk;
                inferenceMilliseconds = current.InferenceMilliseconds;
                UpdateStatus();
            }
            generation = null;
        }

        if (generation == null && model != null && hasGoal)
        {
            // Start a generation early enough that it is ready when it should start playing.
            int lead = (int)math.ceil(inferenceMilliseconds / 1000 * 1.5 * MotionBricks.SampleRate) + 1;
            int halfway = currentStart + current.FrameCount / 2;
            if (goalChanged)
            {
                StartGeneration(math.max(frame + lead, currentStart + 3));
            }
            else if (currentIsWalk && frame >= halfway - lead)
            {
                // Play half of each walking segment, then continue toward the next waypoint.
                StartGeneration(math.max(frame + lead, halfway));
            }
            else if (!currentIsWalk && frame >= currentStart + current.FrameCount)
            {
                hasGoal = false; // Arrived
            }
        }

        // Play the current segment. Past its end, the last pose holds.
        float t = (float)(frameTime - currentStart);
        int a = math.clamp((int)t, 0, current.FrameCount - 1);
        int b = math.min(a + 1, current.FrameCount - 1);
        float blend = math.saturate(t - a);
        pose.Lerp(current.Frames[a], current.Frames[b], blend);
        characterView.Draw(pose, math.lerp(current.FootContacts[a], current.FootContacts[b], blend));

        if (dragging)
        {
            MakeGoalPose(pressPoint, dragFacing, preview);
            goalView.Draw(preview, float4.zero);
        }
        else if (hasGoal)
        {
            goalView.Draw(goal, float4.zero);
        }
    }

    void LateUpdate()
    {
        Vector3 hips = SomaSkeletonView.ToUnity(pose.RootPosition);
        var target = new Vector3(hips.x, 0f, hips.z);
        Vector3 position = Vector3.SmoothDamp(mainCamera.transform.position, target + cameraOffset, ref cameraVelocity, 0.4f);
        mainCamera.transform.SetPositionAndRotation(position, Quaternion.LookRotation(target + Vector3.up * 0.9f - position));
    }

    /// <summary>Generates from the poses that will show at the start frame toward the goal.</summary>
    void StartGeneration(int start)
    {
        for (int i = 0; i < 4; i++)
        {
            int f = math.clamp(start - 3 + i - currentStart, 0, current.FrameCount - 1);
            context[i].CopyFrom(current.Frames[f]);
        }
        float3 hips = context[3].RootPosition;
        float2 toGoal = goal.RootPosition.xz - hips.xz;
        float distance = math.length(toGoal);
        bool moving = math.distance(hips.xz, context[2].RootPosition.xz) * MotionBricks.SampleRate > 0.3f;
        bool walk = distance > StopRadius || (!moving && distance > MinWalkDistance);

        int targetCount = 1;
        if (walk)
        {
            // Four waypoints at walking speed. Only their root matters: the body pose is left to the model.
            float2 direction = toGoal / distance;
            float yaw = math.atan2(direction.x, direction.y);
            float ahead = math.min(WalkSpeed * LookAheadSeconds, distance);
            for (int i = 0; i < 4; i++)
            {
                targets[i].SetIdle(hips.xz + direction * (ahead + WalkSpeed * i / MotionBricks.SampleRate), yaw);
            }
            targetCount = 4;
        }
        else
        {
            targets[0].CopyFrom(goal);
        }

        goalChanged = false;
        nextStart = start;
        nextIsWalk = walk;
        var (runModel, segment) = (model, next);
        generation = Task.Run(() => runModel.Generate(
            context, targets.AsSpan(0, targetCount), !walk, walk ? 8 : 6, walk ? 12 : 16, segment));
    }

    void WaitForGeneration()
    {
        try
        {
            generation?.Wait();
        }
        catch (AggregateException)
        {
        }
        generation = null;
    }

    /// <summary>The idle pose at the destination, facing the drag direction or the way there.</summary>
    void MakeGoalPose(Vector3 destination, Vector3? facing, SomaPose output)
    {
        float3 p = SomaSkeletonView.ToSoma(destination);
        float2 direction = facing.HasValue ? SomaSkeletonView.ToSoma(facing.Value).xz : p.xz - pose.RootPosition.xz;
        float yaw = math.lengthsq(direction) > 1e-4f ? math.atan2(direction.x, direction.y) : pose.Heading;
        output.SetIdle(p.xz, yaw);
    }

    /// <summary>Mouse, pen and touch: press the ground to choose the destination, drag to choose the facing.</summary>
    void HandlePointer()
    {
        var pointer = Pointer.current;
        if (pointer == null)
        {
            return;
        }
        Vector2 screen = pointer.position.ReadValue();
        if (pointer.press.wasPressedThisFrame && !IsOverUI(screen) && TryGetGroundPoint(screen, out pressPoint))
        {
            dragging = true;
        }
        if (!dragging)
        {
            return;
        }
        dragFacing = null;
        if (TryGetGroundPoint(screen, out Vector3 point) && Vector3.Distance(point, pressPoint) > 0.3f)
        {
            dragFacing = point - pressPoint;
        }
        if (pointer.press.wasReleasedThisFrame)
        {
            dragging = false;
            MakeGoalPose(pressPoint, dragFacing, goal);
            hasGoal = true;
            goalChanged = true;
        }
    }

    bool TryGetGroundPoint(Vector2 screen, out Vector3 point)
    {
        var ray = mainCamera.ScreenPointToRay(screen);
        bool hit = new Plane(Vector3.up, Vector3.zero).Raycast(ray, out float enter);
        point = ray.GetPoint(enter);
        return hit;
    }

    bool IsOverUI(Vector2 screen)
    {
        raycastResults.Clear();
        EventSystem.current.RaycastAll(new PointerEventData(EventSystem.current) { position = screen }, raycastResults);
        return raycastResults.Count > 0;
    }

    void UpdateStatus()
    {
        statusLabel.SetText($"{inferenceMilliseconds:F0} ms per generation\n"
            + "<size=80%>Tap the ground to walk there. Drag to set the facing.</size>");
    }
}
