using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FaceAuth.Maui;

internal interface IFaceEmbedder
{
    float[] ExtractEmbedding(RgbImage alignedFace, CancellationToken ct = default);
}

internal sealed class ArcFaceEmbedder : IFaceEmbedder
{
    private readonly InferenceSession _session;
    private readonly int _inputSize;

    public ArcFaceEmbedder(InferenceSession session, int inputSize)
    {
        _session = session;
        _inputSize = inputSize;
    }

    public float[] ExtractEmbedding(RgbImage alignedFace, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();

        var face = alignedFace.Width == _inputSize && alignedFace.Height == _inputSize
            ? alignedFace
            : ImageProcessing.ResizeBilinear(alignedFace, _inputSize, _inputSize);

        var tensor = new DenseTensor<float>(new[] { 1, 3, _inputSize, _inputSize });
        for (var y = 0; y < _inputSize; y++)
        {
            for (var x = 0; x < _inputSize; x++)
            {
                var index = ((y * _inputSize) + x) * 3;
                // ArcFace convention: normalize RGB to roughly [-1,1].
                tensor[0, 0, y, x] = (face.Rgb[index] - 127.5f) / 128f;
                tensor[0, 1, y, x] = (face.Rgb[index + 1] - 127.5f) / 128f;
                tensor[0, 2, y, x] = (face.Rgb[index + 2] - 127.5f) / 128f;
            }
        }

        var inputName = _session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor),
        };

        using var results = _session.Run(inputs);
        var vector = results.First().AsTensor<float>().ToArray();
        if (vector.Length != 512)
        {
            throw new FaceAuthException($"Unexpected ArcFace output size. Expected 512 floats, got {vector.Length}.");
        }

        return FaceMath.NormalizeL2(vector);
    }
}
