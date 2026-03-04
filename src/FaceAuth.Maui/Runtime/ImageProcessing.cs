using System.Numerics;

namespace FaceAuth.Maui;

internal static class ImageProcessing
{
    private static readonly Vector2[] CanonicalArcFacePoints =
    {
        new(38.2946f, 51.6963f),
        new(73.5318f, 51.5014f),
        new(56.0252f, 71.7366f),
        new(41.5493f, 92.3655f),
        new(70.7299f, 92.2041f),
    };

    public static RgbImage ResizeBilinear(RgbImage source, int targetWidth, int targetHeight)
    {
        if (source.Width == targetWidth && source.Height == targetHeight)
        {
            return source;
        }

        var dst = new byte[targetWidth * targetHeight * 3];
        var scaleX = source.Width / (float)targetWidth;
        var scaleY = source.Height / (float)targetHeight;

        for (var y = 0; y < targetHeight; y++)
        {
            var srcY = (y + 0.5f) * scaleY - 0.5f;
            var y0 = Clamp((int)MathF.Floor(srcY), 0, source.Height - 1);
            var y1 = Clamp(y0 + 1, 0, source.Height - 1);
            var wy = srcY - y0;

            for (var x = 0; x < targetWidth; x++)
            {
                var srcX = (x + 0.5f) * scaleX - 0.5f;
                var x0 = Clamp((int)MathF.Floor(srcX), 0, source.Width - 1);
                var x1 = Clamp(x0 + 1, 0, source.Width - 1);
                var wx = srcX - x0;

                var dstIndex = ((y * targetWidth) + x) * 3;
                for (var c = 0; c < 3; c++)
                {
                    var p00 = source.Rgb[((y0 * source.Width) + x0) * 3 + c];
                    var p01 = source.Rgb[((y0 * source.Width) + x1) * 3 + c];
                    var p10 = source.Rgb[((y1 * source.Width) + x0) * 3 + c];
                    var p11 = source.Rgb[((y1 * source.Width) + x1) * 3 + c];

                    var top = Lerp(p00, p01, wx);
                    var bottom = Lerp(p10, p11, wx);
                    var value = Lerp(top, bottom, wy);

                    dst[dstIndex + c] = (byte)Math.Clamp((int)MathF.Round(value), 0, 255);
                }
            }
        }

        return new RgbImage(targetWidth, targetHeight, dst);
    }

    public static RgbImage AlignToArcFace(RgbImage image, FaceDetection detection, int outputSize)
    {
        if (detection.Landmarks.Count >= 5)
        {
            var scale = outputSize / 112f;
            var destination = new[]
            {
                CanonicalArcFacePoints[0] * scale,
                CanonicalArcFacePoints[1] * scale,
                ((CanonicalArcFacePoints[3] + CanonicalArcFacePoints[4]) * 0.5f) * scale,
            };

            var source = new[]
            {
                detection.Landmarks[0],
                detection.Landmarks[1],
                (detection.Landmarks[3] + detection.Landmarks[4]) * 0.5f,
            };

            if (TrySolveAffine(source, destination, out var affine))
            {
                return WarpAffine(image, affine, outputSize, outputSize);
            }
        }

        return CropResizeFallback(image, detection.BoundingBox, outputSize);
    }

    private static RgbImage CropResizeFallback(RgbImage image, BoundingBox box, int outputSize)
    {
        var padX = box.Width * 0.1f;
        var padY = box.Height * 0.1f;

        var x0 = Clamp((int)MathF.Floor(box.XMin - padX), 0, image.Width - 1);
        var y0 = Clamp((int)MathF.Floor(box.YMin - padY), 0, image.Height - 1);
        var x1 = Clamp((int)MathF.Ceiling(box.XMax + padX), x0 + 1, image.Width);
        var y1 = Clamp((int)MathF.Ceiling(box.YMax + padY), y0 + 1, image.Height);

        var width = x1 - x0;
        var height = y1 - y0;
        var rgb = new byte[width * height * 3];

        for (var y = 0; y < height; y++)
        {
            Buffer.BlockCopy(
                image.Rgb,
                (((y0 + y) * image.Width) + x0) * 3,
                rgb,
                y * width * 3,
                width * 3);
        }

        var cropped = new RgbImage(width, height, rgb);
        return ResizeBilinear(cropped, outputSize, outputSize);
    }

    private static RgbImage WarpAffine(RgbImage image, float[] affine, int targetWidth, int targetHeight)
    {
        var dst = new byte[targetWidth * targetHeight * 3];

        var a = affine[0];
        var b = affine[1];
        var c = affine[2];
        var d = affine[3];
        var e = affine[4];
        var f = affine[5];

        var det = (a * e) - (b * d);
        if (MathF.Abs(det) < 1e-6f)
        {
            return ResizeBilinear(image, targetWidth, targetHeight);
        }

        for (var y = 0; y < targetHeight; y++)
        {
            for (var x = 0; x < targetWidth; x++)
            {
                var srcX = ((e * x) - (b * y) + (b * f) - (c * e)) / det;
                var srcY = ((-d * x) + (a * y) + (c * d) - (a * f)) / det;

                var dstIndex = ((y * targetWidth) + x) * 3;
                SampleBilinear(image, srcX, srcY, dst, dstIndex);
            }
        }

        return new RgbImage(targetWidth, targetHeight, dst);
    }

    private static void SampleBilinear(RgbImage source, float x, float y, byte[] dst, int dstOffset)
    {
        if (x < 0 || y < 0 || x > source.Width - 1 || y > source.Height - 1)
        {
            dst[dstOffset] = 0;
            dst[dstOffset + 1] = 0;
            dst[dstOffset + 2] = 0;
            return;
        }

        var x0 = Clamp((int)MathF.Floor(x), 0, source.Width - 1);
        var x1 = Clamp(x0 + 1, 0, source.Width - 1);
        var y0 = Clamp((int)MathF.Floor(y), 0, source.Height - 1);
        var y1 = Clamp(y0 + 1, 0, source.Height - 1);

        var wx = x - x0;
        var wy = y - y0;

        for (var c = 0; c < 3; c++)
        {
            var p00 = source.Rgb[((y0 * source.Width) + x0) * 3 + c];
            var p01 = source.Rgb[((y0 * source.Width) + x1) * 3 + c];
            var p10 = source.Rgb[((y1 * source.Width) + x0) * 3 + c];
            var p11 = source.Rgb[((y1 * source.Width) + x1) * 3 + c];

            var top = Lerp(p00, p01, wx);
            var bottom = Lerp(p10, p11, wx);
            var value = Lerp(top, bottom, wy);

            dst[dstOffset + c] = (byte)Math.Clamp((int)MathF.Round(value), 0, 255);
        }
    }

    private static bool TrySolveAffine(IReadOnlyList<Vector2> source, IReadOnlyList<Vector2> destination, out float[] affine)
    {
        affine = Array.Empty<float>();

        if (source.Count != 3 || destination.Count != 3)
        {
            return false;
        }

        var matrix = new double[6, 7];
        for (var i = 0; i < 3; i++)
        {
            var sx = source[i].X;
            var sy = source[i].Y;
            var dx = destination[i].X;
            var dy = destination[i].Y;

            var row = i * 2;
            matrix[row, 0] = sx;
            matrix[row, 1] = sy;
            matrix[row, 2] = 1;
            matrix[row, 6] = dx;

            matrix[row + 1, 3] = sx;
            matrix[row + 1, 4] = sy;
            matrix[row + 1, 5] = 1;
            matrix[row + 1, 6] = dy;
        }

        if (!GaussianElimination(matrix, 6))
        {
            return false;
        }

        affine = new[]
        {
            (float)matrix[0, 6],
            (float)matrix[1, 6],
            (float)matrix[2, 6],
            (float)matrix[3, 6],
            (float)matrix[4, 6],
            (float)matrix[5, 6],
        };

        return true;
    }

    private static bool GaussianElimination(double[,] matrix, int size)
    {
        for (var pivot = 0; pivot < size; pivot++)
        {
            var maxRow = pivot;
            var maxValue = Math.Abs(matrix[pivot, pivot]);
            for (var row = pivot + 1; row < size; row++)
            {
                var value = Math.Abs(matrix[row, pivot]);
                if (value > maxValue)
                {
                    maxValue = value;
                    maxRow = row;
                }
            }

            if (maxValue < 1e-10)
            {
                return false;
            }

            if (maxRow != pivot)
            {
                for (var col = pivot; col <= size; col++)
                {
                    (matrix[pivot, col], matrix[maxRow, col]) = (matrix[maxRow, col], matrix[pivot, col]);
                }
            }

            var pivotValue = matrix[pivot, pivot];
            for (var col = pivot; col <= size; col++)
            {
                matrix[pivot, col] /= pivotValue;
            }

            for (var row = 0; row < size; row++)
            {
                if (row == pivot)
                {
                    continue;
                }

                var factor = matrix[row, pivot];
                for (var col = pivot; col <= size; col++)
                {
                    matrix[row, col] -= factor * matrix[pivot, col];
                }
            }
        }

        return true;
    }

    private static int Clamp(int value, int min, int max) => Math.Min(Math.Max(value, min), max);

    private static float Lerp(float a, float b, float t) => a + ((b - a) * t);
}
