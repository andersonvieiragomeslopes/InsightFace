namespace FaceAuth.Maui;

public interface IFaceAuthService
{
    Task InitializeAsync(FaceAuthOptions? options = null);

    Task<FaceEmbeddingResult> ExtractEmbeddingAsync(byte[] imageBytes, CancellationToken ct = default);

    FaceMatchResult Compare(float[] probeEmbedding, float[] referenceEmbedding);
}
