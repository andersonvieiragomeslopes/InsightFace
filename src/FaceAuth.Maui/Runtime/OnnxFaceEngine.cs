namespace FaceAuth.Maui;

internal sealed class OnnxFaceEngine : IFaceEngine, IDisposable
{
    private readonly OnnxSessionManager _sessionManager;
    private readonly IImageDecoder _imageDecoder;
    private readonly SemaphoreSlim _initializeLock = new(1, 1);

    private IFaceDetector? _detector;
    private IFaceEmbedder? _embedder;
    private FaceAuthOptions _options = new();
    private bool _initialized;

    public OnnxFaceEngine(OnnxSessionManager sessionManager, IImageDecoder imageDecoder)
    {
        _sessionManager = sessionManager;
        _imageDecoder = imageDecoder;
    }

    public async Task InitializeAsync(FaceAuthOptions options, CancellationToken ct = default)
    {
        await _initializeLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            await _sessionManager.InitializeAsync(ct).ConfigureAwait(false);
            _detector = new ScrfdFaceDetector(_sessionManager.DetectorSession);
            _embedder = new ArcFaceEmbedder(_sessionManager.RecognizerSession, options.InputSize);
            _options = options;
            _initialized = true;
        }
        finally
        {
            _initializeLock.Release();
        }
    }

    public Task<EngineEmbeddingOutput> ExtractEmbeddingAsync(byte[] imageBytes, CancellationToken ct = default)
    {
        if (!_initialized || _detector is null || _embedder is null)
        {
            throw new FaceAuthNotInitializedException();
        }

        var image = _imageDecoder.Decode(imageBytes);
        var detections = _detector.Detect(image, ct);

        if (detections.Count == 0)
        {
            throw new FaceNotFoundException();
        }

        if (_options.RequireSingleFace && detections.Count > 1)
        {
            throw new MultipleFacesDetectedException(detections.Count);
        }

        var selected = detections
            .OrderByDescending(static d => d.BoundingBox.Area)
            .ThenByDescending(static d => d.Confidence)
            .First();

        var aligned = ImageProcessing.AlignToArcFace(image, selected, _options.InputSize);
        var sharpness = FaceQualityScoring.ComputeSharpnessScore(aligned);
        var illumination = FaceQualityScoring.ComputeIlluminationScore(aligned);

        var embedding = _embedder.ExtractEmbedding(aligned, ct);
        var result = new EngineEmbeddingOutput(embedding, selected.Confidence, sharpness, illumination);
        return Task.FromResult(result);
    }

    public void Dispose()
    {
        _initializeLock.Dispose();
        _sessionManager.Dispose();
    }
}
