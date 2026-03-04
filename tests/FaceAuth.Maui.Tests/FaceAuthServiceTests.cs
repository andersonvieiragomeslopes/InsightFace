using FaceAuth.Maui;

namespace FaceAuth.Maui.Tests;

public class FaceAuthServiceTests
{
    [Fact]
    public async Task ExtractEmbeddingAsync_WithoutInitialize_Throws()
    {
        var service = new FaceAuthService(new StubFaceEngine());

        await Assert.ThrowsAsync<FaceAuthNotInitializedException>(async () =>
            await service.ExtractEmbeddingAsync(ReadFixture("selfie-a.jpg")));
    }

    [Fact]
    public async Task ExtractEmbeddingAsync_NoFace_ThrowsFaceNotFoundException()
    {
        var service = new FaceAuthService(new StubFaceEngine
        {
            ExceptionToThrow = new FaceNotFoundException(),
        });

        await service.InitializeAsync();

        await Assert.ThrowsAsync<FaceNotFoundException>(async () =>
            await service.ExtractEmbeddingAsync(ReadFixture("selfie-a.jpg")));
    }

    [Fact]
    public async Task ExtractEmbeddingAsync_MultipleFaces_ThrowsMultipleFacesDetectedException()
    {
        var service = new FaceAuthService(new StubFaceEngine
        {
            ExceptionToThrow = new MultipleFacesDetectedException(2),
        });

        await service.InitializeAsync();

        await Assert.ThrowsAsync<MultipleFacesDetectedException>(async () =>
            await service.ExtractEmbeddingAsync(ReadFixture("selfie-b.jpg")));
    }

    [Fact]
    public async Task ExtractEmbeddingAsync_LowQuality_ThrowsWhenEnabled()
    {
        var service = new FaceAuthService(new StubFaceEngine
        {
            Output = new EngineEmbeddingOutput(
                Embedding: CreateEmbedding(seed: 42),
                DetectionConfidence: 0.50f,
                SharpnessScore: 0.10f,
                IlluminationScore: 0.20f),
        });

        await service.InitializeAsync(new FaceAuthOptions
        {
            MinDetectionConfidence = 0.70f,
            MinQualityScore = 0.55f,
            ThrowOnLowQuality = true,
        });

        await Assert.ThrowsAsync<LowFaceQualityException>(async () =>
            await service.ExtractEmbeddingAsync(ReadFixture("selfie-a.jpg")));
    }

    [Fact]
    public async Task ExtractEmbeddingAsync_LowQuality_ReturnsWhenDisabled()
    {
        var service = new FaceAuthService(new StubFaceEngine
        {
            Output = new EngineEmbeddingOutput(
                Embedding: CreateEmbedding(seed: 7),
                DetectionConfidence: 0.50f,
                SharpnessScore: 0.10f,
                IlluminationScore: 0.20f),
        });

        await service.InitializeAsync(new FaceAuthOptions
        {
            MinDetectionConfidence = 0.70f,
            MinQualityScore = 0.55f,
            ThrowOnLowQuality = false,
        });

        var result = await service.ExtractEmbeddingAsync(ReadFixture("selfie-a.jpg"));

        Assert.Equal(512, result.Embedding.Length);
        Assert.InRange(result.DetectionConfidence, 0.49f, 0.51f);
        Assert.InRange(result.QualityScore, 0.0f, 1.0f);
    }

    [Fact]
    public void Compare_UsesThresholdAndNormalization()
    {
        var service = new FaceAuthService(new StubFaceEngine());

        var probe = CreateEmbedding(seed: 111, normalize: false);
        var reference = probe.ToArray();

        var match = service.Compare(probe, reference);

        Assert.True(match.IsMatch);
        Assert.InRange(match.Score, 0.999f, 1.001f);
    }

    [Fact]
    public void Compare_InvalidSize_Throws()
    {
        var service = new FaceAuthService(new StubFaceEngine());

        var probe = new float[128];
        var reference = new float[512];

        Assert.Throws<ArgumentException>(() => service.Compare(probe, reference));
    }

    private static float[] CreateEmbedding(int seed, bool normalize = true)
    {
        var random = new Random(seed);
        var values = new float[512];
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }

        return normalize ? FaceMath.NormalizeL2(values) : values;
    }

    private static byte[] ReadFixture(string name)
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", name);
        return File.ReadAllBytes(path);
    }

    private sealed class StubFaceEngine : IFaceEngine
    {
        public Exception? ExceptionToThrow { get; init; }
        public EngineEmbeddingOutput? Output { get; init; }

        public Task InitializeAsync(FaceAuthOptions options, CancellationToken ct = default)
        {
            return Task.CompletedTask;
        }

        public Task<EngineEmbeddingOutput> ExtractEmbeddingAsync(byte[] imageBytes, CancellationToken ct = default)
        {
            if (ExceptionToThrow is not null)
            {
                throw ExceptionToThrow;
            }

            if (Output is not null)
            {
                return Task.FromResult(Output);
            }

            return Task.FromResult(new EngineEmbeddingOutput(
                Embedding: CreateEmbedding(seed: 123),
                DetectionConfidence: 0.95f,
                SharpnessScore: 0.90f,
                IlluminationScore: 0.85f));
        }
    }
}
