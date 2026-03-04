namespace FaceAuth.Maui;

public sealed class LowFaceQualityException : FaceAuthException
{
    public LowFaceQualityException(float qualityScore, float detectionConfidence, float minQualityScore, float minDetectionConfidence)
        : base($"Face quality check failed. QualityScore={qualityScore:F3} (min={minQualityScore:F3}), DetectionConfidence={detectionConfidence:F3} (min={minDetectionConfidence:F3}).")
    {
        QualityScore = qualityScore;
        DetectionConfidence = detectionConfidence;
        MinQualityScore = minQualityScore;
        MinDetectionConfidence = minDetectionConfidence;
    }

    public float QualityScore { get; }
    public float DetectionConfidence { get; }
    public float MinQualityScore { get; }
    public float MinDetectionConfidence { get; }
}
