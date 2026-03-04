using System.Numerics;

namespace FaceAuth.Maui;

internal sealed class RgbImage
{
    public RgbImage(int width, int height, byte[] rgb)
    {
        Width = width;
        Height = height;
        Rgb = rgb ?? throw new ArgumentNullException(nameof(rgb));

        var expectedLength = width * height * 3;
        if (rgb.Length != expectedLength)
        {
            throw new ArgumentException($"Invalid RGB buffer length. Expected {expectedLength} bytes, got {rgb.Length}.", nameof(rgb));
        }
    }

    public int Width { get; }
    public int Height { get; }
    public byte[] Rgb { get; }
}

internal readonly record struct BoundingBox(float XMin, float YMin, float XMax, float YMax)
{
    public float Width => MathF.Max(0f, XMax - XMin);
    public float Height => MathF.Max(0f, YMax - YMin);
    public float Area => Width * Height;
}

internal sealed record FaceDetection(BoundingBox BoundingBox, float Confidence, IReadOnlyList<Vector2> Landmarks);

internal sealed record EngineEmbeddingOutput(
    float[] Embedding,
    float DetectionConfidence,
    float SharpnessScore,
    float IlluminationScore);

internal interface IFaceEngine
{
    Task InitializeAsync(FaceAuthOptions options, CancellationToken ct = default);
    Task<EngineEmbeddingOutput> ExtractEmbeddingAsync(byte[] imageBytes, CancellationToken ct = default);
}
