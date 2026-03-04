namespace FaceAuth.Maui;

public sealed class FaceAuthService : IFaceAuthService, IDisposable
{
    private readonly IFaceEngine _engine;
    private readonly SemaphoreSlim _initializeLock = new(1, 1);

    private bool _initialized;
    private FaceAuthOptions _options = new();

    public FaceAuthService()
        : this(new OnnxFaceEngine(new OnnxSessionManager(), ImageDecoderFactory.Create()))
    {
    }

    internal FaceAuthService(IFaceEngine engine)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
    }

    public async Task InitializeAsync(FaceAuthOptions? options = null)
    {
        var effectiveOptions = options ?? new FaceAuthOptions();
        effectiveOptions.Validate();

        await _initializeLock.WaitAsync().ConfigureAwait(false);
        try
        {
            await _engine.InitializeAsync(effectiveOptions).ConfigureAwait(false);
            _options = effectiveOptions;
            _initialized = true;
        }
        finally
        {
            _initializeLock.Release();
        }
    }

    public async Task<FaceEmbeddingResult> ExtractEmbeddingAsync(byte[] imageBytes, CancellationToken ct = default)
    {
        if (!_initialized)
        {
            throw new FaceAuthNotInitializedException();
        }

        if (imageBytes is null)
        {
            throw new ArgumentNullException(nameof(imageBytes));
        }

        if (imageBytes.Length == 0)
        {
            throw new ArgumentException("Image payload cannot be empty.", nameof(imageBytes));
        }

        var output = await _engine.ExtractEmbeddingAsync(imageBytes, ct).ConfigureAwait(false);
        var qualityScore = FaceQualityScoring.ComputeQualityScore(
            output.DetectionConfidence,
            output.SharpnessScore,
            output.IlluminationScore);

        var isLowQuality = output.DetectionConfidence < _options.MinDetectionConfidence || qualityScore < _options.MinQualityScore;
        if (isLowQuality && _options.ThrowOnLowQuality)
        {
            throw new LowFaceQualityException(
                qualityScore,
                output.DetectionConfidence,
                _options.MinQualityScore,
                _options.MinDetectionConfidence);
        }

        return new FaceEmbeddingResult(output.Embedding, qualityScore, output.DetectionConfidence);
    }

    public FaceMatchResult Compare(float[] probeEmbedding, float[] referenceEmbedding)
    {
        if (probeEmbedding is null)
        {
            throw new ArgumentNullException(nameof(probeEmbedding));
        }

        if (referenceEmbedding is null)
        {
            throw new ArgumentNullException(nameof(referenceEmbedding));
        }

        if (probeEmbedding.Length == 0 || referenceEmbedding.Length == 0)
        {
            throw new ArgumentException("Embeddings cannot be empty.");
        }

        if (probeEmbedding.Length != 512 || referenceEmbedding.Length != 512)
        {
            throw new ArgumentException("Embeddings must have exactly 512 floats.");
        }

        var score = FaceMath.CosineSimilarity(probeEmbedding, referenceEmbedding);
        var isMatch = score >= _options.MatchThreshold;

        return new FaceMatchResult(score, isMatch);
    }

    public void Dispose()
    {
        _initializeLock.Dispose();
        if (_engine is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }
}
