# FaceAuth.Maui

`FaceAuth.Maui` is a .NET MAUI plugin for on-device face embedding extraction and 1:1 face match for recurring login flows.

It uses the InsightFace model ecosystem (SCRFD for detection + ArcFace for embeddings) through ONNX Runtime.

## Features

- Android + iOS support with .NET MAUI
- Target frameworks: `net10.0-android` and `net10.0-ios`
- Offline pipeline (`detection -> alignment -> embedding`)
- `512` float embedding output
- Local 1:1 comparison with cosine similarity
- Embedding serialization to Base64 (`float32`, little-endian)
- NuGet packaging + GitHub Actions artifact pipeline

## Public API

```csharp
public record FaceEmbeddingResult(
    float[] Embedding,
    float QualityScore,
    float DetectionConfidence);

public record FaceMatchResult(
    float Score,
    bool IsMatch);

public sealed class FaceAuthOptions
{
    public float MatchThreshold { get; init; } = 0.55f;
    public bool RequireSingleFace { get; init; } = true;
    public int InputSize { get; init; } = 112;

    public float MinDetectionConfidence { get; init; } = 0.70f;
    public float MinQualityScore { get; init; } = 0.55f;
    public bool ThrowOnLowQuality { get; init; } = true;
}

public interface IFaceAuthService
{
    Task InitializeAsync(FaceAuthOptions? options = null);
    Task<FaceEmbeddingResult> ExtractEmbeddingAsync(byte[] imageBytes, CancellationToken ct = default);
    FaceMatchResult Compare(float[] probeEmbedding, float[] referenceEmbedding);
}
```

## Quick Start

```csharp
using FaceAuth.Maui;

var service = new FaceAuthService();

await service.InitializeAsync(new FaceAuthOptions
{
    MatchThreshold = 0.55f,
    RequireSingleFace = true,
    MinDetectionConfidence = 0.70f,
    MinQualityScore = 0.55f,
    ThrowOnLowQuality = true,
});

var enrollment = await service.ExtractEmbeddingAsync(imageBytes);
var serialized = FaceEmbeddingSerializer.ToBase64(enrollment.Embedding);

// send serialized embedding to server
```

Later (recurring login):

```csharp
var probe = await service.ExtractEmbeddingAsync(newSelfieBytes);
var reference = FaceEmbeddingSerializer.FromBase64(serverEmbeddingBase64);

var match = service.Compare(probe.Embedding, reference);
// match.Score + match.IsMatch
```

## Embedding Storage Format

`FaceEmbeddingSerializer` uses:

- `float32` values
- little-endian byte order
- Base64 string transport

That means backend stores a compact string without preserving original image.

## Quality and Threshold Tuning

### Match threshold

`MatchThreshold` controls acceptance in `Compare` (cosine similarity).

- Typical starting range: `0.45` to `0.60`
- Default in this package: `0.55`

Tune with your own dataset and FAR/FRR targets.

### Quality gate

`ExtractEmbeddingAsync` computes:

- `DetectionConfidence` from detector output
- `QualityScore` (deterministic):

```text
QualityScore = 0.60*detectionConfidence + 0.25*sharpness + 0.15*illumination
```

Control behavior with:

- `MinDetectionConfidence`
- `MinQualityScore`
- `ThrowOnLowQuality`

If `ThrowOnLowQuality = true`, extraction throws `LowFaceQualityException` on low quality/confidence.

## Failure Modes

`ExtractEmbeddingAsync` throws clear exceptions for:

- `FaceNotFoundException`
- `MultipleFacesDetectedException` (when `RequireSingleFace=true`)
- `LowFaceQualityException` (when threshold gate is active)
- `FaceAuthNotInitializedException` (when `InitializeAsync` was not called)

## Security Notes (Important)

- This package is **NOT** liveness/deepfake detection.
- For fraud resistance, combine with liveness checks and device integrity signals.
- Do not store selfies by default.
- Prefer storing only embedding + metadata + audit info.

## Model Assets and Licensing

This repository includes placeholders at:

- `src/FaceAuth.Maui/Models/scrfd.onnx`
- `src/FaceAuth.Maui/Models/arcface.onnx`

Replace with real ONNX binaries before production use.

Recommended source package for SCRFD + ArcFace:

- Official InsightFace Model Zoo link (Google Drive): [buffalo_l.zip](https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing)
- Mirror (SourceForge): [buffalo_l.zip](https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download)

After download/extract, use:

- `det_10g.onnx` (SCRFD)
- `w600k_r50.onnx` (ArcFace)

Copy/rename to:

- `src/FaceAuth.Maui/Models/scrfd.onnx` <- `det_10g.onnx`
- `src/FaceAuth.Maui/Models/arcface.onnx` <- `w600k_r50.onnx`

You are responsible for validating model provenance and license terms from the InsightFace ecosystem and keeping attribution/compliance in your distribution.

## Repository Layout

- `src/FaceAuth.Maui/` plugin package
- `sample/FaceAuth.SampleApp/` demo MAUI app (camera + gallery)
- `tests/FaceAuth.Maui.Tests/` unit/service tests
- `.github/workflows/pack.yml` CI for build + pack (artifact only)

## CI: build + pack only

Workflow triggers:

- push to `main`
- push tag `v*`
- manual `workflow_dispatch`

It builds/tests/packs and uploads `.nupkg`/`.snupkg` artifacts.

No `nuget push` is executed.
