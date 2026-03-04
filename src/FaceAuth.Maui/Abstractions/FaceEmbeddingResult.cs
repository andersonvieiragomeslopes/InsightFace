namespace FaceAuth.Maui;

public record FaceEmbeddingResult(
    float[] Embedding,
    float QualityScore,
    float DetectionConfidence);
