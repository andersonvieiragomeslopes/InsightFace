namespace FaceAuth.Maui;

public sealed class FaceAuthOptions
{
    public float MatchThreshold { get; init; } = 0.55f;
    public bool RequireSingleFace { get; init; } = true;
    public int InputSize { get; init; } = 112;

    public float MinDetectionConfidence { get; init; } = 0.70f;
    public float MinQualityScore { get; init; } = 0.55f;
    public bool ThrowOnLowQuality { get; init; } = true;

    internal void Validate()
    {
        if (InputSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(InputSize), "InputSize must be greater than zero.");
        }

        if (MatchThreshold is < -1f or > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(MatchThreshold), "MatchThreshold must be between -1 and 1.");
        }

        if (MinDetectionConfidence is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(MinDetectionConfidence), "MinDetectionConfidence must be between 0 and 1.");
        }

        if (MinQualityScore is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(MinQualityScore), "MinQualityScore must be between 0 and 1.");
        }
    }
}
