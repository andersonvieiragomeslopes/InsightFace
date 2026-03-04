using Microsoft.ML.OnnxRuntime;

namespace FaceAuth.Maui;

internal sealed class OnnxSessionManager : IDisposable
{
    private const string DetectorResourceName = "FaceAuth.Maui.Models.scrfd.onnx";
    private const string RecognizerResourceName = "FaceAuth.Maui.Models.arcface.onnx";

    private readonly SemaphoreSlim _initializeLock = new(1, 1);
    private InferenceSession? _detectorSession;
    private InferenceSession? _recognizerSession;
    private bool _initialized;

    public InferenceSession DetectorSession => _detectorSession ?? throw new FaceAuthNotInitializedException();
    public InferenceSession RecognizerSession => _recognizerSession ?? throw new FaceAuthNotInitializedException();

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        if (_initialized)
        {
            return;
        }

        await _initializeLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (_initialized)
            {
                return;
            }

            var detectorModel = LoadResourceBytes(DetectorResourceName);
            var recognizerModel = LoadResourceBytes(RecognizerResourceName);

            _detectorSession = CreateSession(detectorModel, DetectorResourceName);
            _recognizerSession = CreateSession(recognizerModel, RecognizerResourceName);
            _initialized = true;
        }
        finally
        {
            _initializeLock.Release();
        }
    }

    public void Dispose()
    {
        _detectorSession?.Dispose();
        _recognizerSession?.Dispose();
        _initializeLock.Dispose();
    }

    private static InferenceSession CreateSession(byte[] modelBytes, string modelName)
    {
        try
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount / 2),
                InterOpNumThreads = 1,
            };

            return new InferenceSession(modelBytes, options);
        }
        catch (Exception ex) when (ex is OnnxRuntimeException || ex is ArgumentException)
        {
            throw new FaceAuthException(
                $"Failed to initialize ONNX session for '{modelName}'. Replace placeholder model files with valid SCRFD/ArcFace ONNX binaries.",
                ex);
        }
    }

    private static byte[] LoadResourceBytes(string resourceName)
    {
        var assembly = typeof(OnnxSessionManager).Assembly;
        using var stream = assembly.GetManifestResourceStream(resourceName);
        if (stream is null)
        {
            var available = string.Join(", ", assembly.GetManifestResourceNames().OrderBy(static n => n));
            throw new FaceAuthException($"Embedded model resource '{resourceName}' was not found. Available resources: {available}");
        }

        using var memory = new MemoryStream();
        stream.CopyTo(memory);
        return memory.ToArray();
    }
}
