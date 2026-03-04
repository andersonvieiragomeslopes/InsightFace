using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FaceAuth.Maui;

internal interface IFaceDetector
{
    IReadOnlyList<FaceDetection> Detect(RgbImage image, CancellationToken ct = default);
}

internal sealed class ScrfdFaceDetector : IFaceDetector
{
    private readonly InferenceSession _session;
    private readonly int _inputSize;
    private readonly float _minConfidence;

    public ScrfdFaceDetector(InferenceSession session, int inputSize = 640, float minConfidence = 0.35f)
    {
        _session = session;
        _inputSize = inputSize;
        _minConfidence = minConfidence;
    }

    public IReadOnlyList<FaceDetection> Detect(RgbImage image, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();

        var resized = ImageProcessing.ResizeBilinear(image, _inputSize, _inputSize);
        var input = CreateInputTensor(resized);

        var inputName = _session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, input),
        };

        using var outputs = _session.Run(inputs);
        var parsed = ParseDetections(outputs, image.Width / (float)_inputSize, image.Height / (float)_inputSize);
        return ApplyNms(parsed, iouThreshold: 0.4f);
    }

    private DenseTensor<float> CreateInputTensor(RgbImage image)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, _inputSize, _inputSize });
        for (var y = 0; y < _inputSize; y++)
        {
            for (var x = 0; x < _inputSize; x++)
            {
                var offset = ((y * _inputSize) + x) * 3;
                tensor[0, 0, y, x] = image.Rgb[offset] / 255f;
                tensor[0, 1, y, x] = image.Rgb[offset + 1] / 255f;
                tensor[0, 2, y, x] = image.Rgb[offset + 2] / 255f;
            }
        }

        return tensor;
    }

    private List<FaceDetection> ParseDetections(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, float scaleX, float scaleY)
    {
        var tensors = new List<(string Name, Tensor<float> Tensor)>();
        foreach (var output in outputs)
        {
            try
            {
                tensors.Add((output.Name, output.AsTensor<float>()));
            }
            catch
            {
                // Ignore non-float outputs.
            }
        }

        var detections = TryParseDetsKpsPair(tensors, scaleX, scaleY);
        if (detections.Count > 0)
        {
            return detections;
        }

        detections = TryParseSinglePackedTensor(tensors, scaleX, scaleY);
        if (detections.Count > 0)
        {
            return detections;
        }

        detections = TryParseSplitBboxScoreKps(tensors, scaleX, scaleY);
        return detections;
    }

    private List<FaceDetection> TryParseDetsKpsPair(List<(string Name, Tensor<float> Tensor)> tensors, float scaleX, float scaleY)
    {
        var detTensor = tensors
            .Select(static t => t.Tensor)
            .FirstOrDefault(static t => t.Rank == 3 && t.Dimensions[2] >= 5);

        var kpsTensor = tensors
            .Select(static t => t.Tensor)
            .FirstOrDefault(static t => t.Rank == 3 && t.Dimensions[2] >= 10);

        if (detTensor is null || kpsTensor is null)
        {
            return new List<FaceDetection>();
        }

        var n = detTensor.Dimensions[1];
        if (kpsTensor.Dimensions[1] != n)
        {
            return new List<FaceDetection>();
        }

        var detData = detTensor.ToArray();
        var kpsData = kpsTensor.ToArray();
        var detStride = detTensor.Dimensions[2];
        var kpsStride = kpsTensor.Dimensions[2];

        var results = new List<FaceDetection>(n);
        for (var i = 0; i < n; i++)
        {
            var detOffset = i * detStride;
            var score = detData[detOffset + 4];
            if (score < _minConfidence)
            {
                continue;
            }

            var bbox = new BoundingBox(
                detData[detOffset] * scaleX,
                detData[detOffset + 1] * scaleY,
                detData[detOffset + 2] * scaleX,
                detData[detOffset + 3] * scaleY);

            var lmOffset = i * kpsStride;
            var landmarks = new List<Vector2>(5);
            for (var p = 0; p < 5; p++)
            {
                landmarks.Add(new Vector2(
                    kpsData[lmOffset + (p * 2)] * scaleX,
                    kpsData[lmOffset + (p * 2) + 1] * scaleY));
            }

            if (bbox.Area > 0)
            {
                results.Add(new FaceDetection(bbox, score, landmarks));
            }
        }

        return results;
    }

    private List<FaceDetection> TryParseSinglePackedTensor(List<(string Name, Tensor<float> Tensor)> tensors, float scaleX, float scaleY)
    {
        var packed = tensors
            .Select(static t => t.Tensor)
            .FirstOrDefault(static t =>
                (t.Rank == 2 && t.Dimensions[1] >= 15) ||
                (t.Rank == 3 && t.Dimensions[0] == 1 && t.Dimensions[2] >= 15));

        if (packed is null)
        {
            return new List<FaceDetection>();
        }

        var results = new List<FaceDetection>();
        var data = packed.ToArray();

        if (packed.Rank == 2)
        {
            var n = packed.Dimensions[0];
            var stride = packed.Dimensions[1];
            for (var i = 0; i < n; i++)
            {
                ParsePackedRow(data, i * stride, stride, scaleX, scaleY, results);
            }

            return results;
        }

        var count = packed.Dimensions[1];
        var packedStride = packed.Dimensions[2];
        for (var i = 0; i < count; i++)
        {
            ParsePackedRow(data, i * packedStride, packedStride, scaleX, scaleY, results);
        }

        return results;
    }

    private void ParsePackedRow(float[] data, int offset, int stride, float scaleX, float scaleY, List<FaceDetection> results)
    {
        var score = data[offset + 4];
        if (score < _minConfidence)
        {
            return;
        }

        var bbox = new BoundingBox(
            data[offset] * scaleX,
            data[offset + 1] * scaleY,
            data[offset + 2] * scaleX,
            data[offset + 3] * scaleY);

        if (bbox.Area <= 0)
        {
            return;
        }

        var landmarks = new List<Vector2>(5);
        var kpsStart = offset + 5;
        var available = stride - 5;
        if (available >= 10)
        {
            for (var p = 0; p < 5; p++)
            {
                landmarks.Add(new Vector2(
                    data[kpsStart + (2 * p)] * scaleX,
                    data[kpsStart + (2 * p) + 1] * scaleY));
            }
        }

        results.Add(new FaceDetection(bbox, score, landmarks));
    }

    private List<FaceDetection> TryParseSplitBboxScoreKps(List<(string Name, Tensor<float> Tensor)> tensors, float scaleX, float scaleY)
    {
        var bboxTensor = tensors
            .Select(static t => t.Tensor)
            .FirstOrDefault(static t =>
                (t.Rank == 2 && t.Dimensions[1] == 4) ||
                (t.Rank == 3 && t.Dimensions[0] == 1 && t.Dimensions[2] == 4));

        var scoreTensor = tensors
            .Select(static t => t.Tensor)
            .FirstOrDefault(static t =>
                (t.Rank == 1) ||
                (t.Rank == 2 && t.Dimensions[1] == 1) ||
                (t.Rank == 3 && t.Dimensions[2] == 1));

        var kpsTensor = tensors
            .Select(static t => t.Tensor)
            .FirstOrDefault(static t =>
                (t.Rank == 2 && t.Dimensions[1] >= 10) ||
                (t.Rank == 3 && t.Dimensions[0] == 1 && t.Dimensions[2] >= 10));

        if (bboxTensor is null || scoreTensor is null)
        {
            return new List<FaceDetection>();
        }

        var boxes = FlattenRows(bboxTensor, expectedRowWidth: 4);
        var scores = FlattenScores(scoreTensor);
        var landmarksRows = kpsTensor is null ? new List<float[]>() : FlattenRows(kpsTensor, expectedRowWidth: 10);

        var count = Math.Min(boxes.Count, scores.Count);
        var results = new List<FaceDetection>(count);
        for (var i = 0; i < count; i++)
        {
            var score = scores[i];
            if (score < _minConfidence)
            {
                continue;
            }

            var box = boxes[i];
            var bbox = new BoundingBox(box[0] * scaleX, box[1] * scaleY, box[2] * scaleX, box[3] * scaleY);
            if (bbox.Area <= 0)
            {
                continue;
            }

            var landmarks = new List<Vector2>(5);
            if (i < landmarksRows.Count)
            {
                var row = landmarksRows[i];
                for (var p = 0; p < 5; p++)
                {
                    landmarks.Add(new Vector2(row[p * 2] * scaleX, row[(p * 2) + 1] * scaleY));
                }
            }

            results.Add(new FaceDetection(bbox, score, landmarks));
        }

        return results;
    }

    private static List<float[]> FlattenRows(Tensor<float> tensor, int expectedRowWidth)
    {
        var rows = new List<float[]>();
        var data = tensor.ToArray();

        if (tensor.Rank == 2)
        {
            var rowCount = tensor.Dimensions[0];
            var width = tensor.Dimensions[1];
            if (width < expectedRowWidth)
            {
                return rows;
            }

            for (var i = 0; i < rowCount; i++)
            {
                var row = new float[expectedRowWidth];
                Array.Copy(data, i * width, row, 0, expectedRowWidth);
                rows.Add(row);
            }

            return rows;
        }

        if (tensor.Rank == 3 && tensor.Dimensions[0] == 1)
        {
            var rowCount = tensor.Dimensions[1];
            var width = tensor.Dimensions[2];
            if (width < expectedRowWidth)
            {
                return rows;
            }

            for (var i = 0; i < rowCount; i++)
            {
                var row = new float[expectedRowWidth];
                Array.Copy(data, i * width, row, 0, expectedRowWidth);
                rows.Add(row);
            }
        }

        return rows;
    }

    private static List<float> FlattenScores(Tensor<float> tensor)
    {
        var scores = new List<float>();
        var data = tensor.ToArray();

        if (tensor.Rank == 1)
        {
            scores.AddRange(data);
            return scores;
        }

        if (tensor.Rank == 2)
        {
            if (tensor.Dimensions[1] == 1)
            {
                for (var i = 0; i < tensor.Dimensions[0]; i++)
                {
                    scores.Add(data[i]);
                }
            }

            return scores;
        }

        if (tensor.Rank == 3 && tensor.Dimensions[0] == 1 && tensor.Dimensions[2] == 1)
        {
            for (var i = 0; i < tensor.Dimensions[1]; i++)
            {
                scores.Add(data[i]);
            }
        }

        return scores;
    }

    private static IReadOnlyList<FaceDetection> ApplyNms(List<FaceDetection> detections, float iouThreshold)
    {
        if (detections.Count <= 1)
        {
            return detections;
        }

        var sorted = detections
            .OrderByDescending(static d => d.Confidence)
            .ToList();

        var selected = new List<FaceDetection>(sorted.Count);
        while (sorted.Count > 0)
        {
            var current = sorted[0];
            selected.Add(current);
            sorted.RemoveAt(0);

            for (var i = sorted.Count - 1; i >= 0; i--)
            {
                if (ComputeIoU(current.BoundingBox, sorted[i].BoundingBox) > iouThreshold)
                {
                    sorted.RemoveAt(i);
                }
            }
        }

        return selected;
    }

    private static float ComputeIoU(BoundingBox a, BoundingBox b)
    {
        var interLeft = MathF.Max(a.XMin, b.XMin);
        var interTop = MathF.Max(a.YMin, b.YMin);
        var interRight = MathF.Min(a.XMax, b.XMax);
        var interBottom = MathF.Min(a.YMax, b.YMax);

        var interWidth = MathF.Max(0f, interRight - interLeft);
        var interHeight = MathF.Max(0f, interBottom - interTop);
        var interArea = interWidth * interHeight;
        if (interArea <= 0f)
        {
            return 0f;
        }

        var union = a.Area + b.Area - interArea;
        return union <= 0f ? 0f : interArea / union;
    }
}
