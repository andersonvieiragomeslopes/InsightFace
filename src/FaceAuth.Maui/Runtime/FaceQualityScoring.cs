namespace FaceAuth.Maui;

internal static class FaceQualityScoring
{
    public static float ComputeQualityScore(float detectionConfidence, float sharpnessScore, float illuminationScore)
    {
        var quality = (0.60f * detectionConfidence) + (0.25f * sharpnessScore) + (0.15f * illuminationScore);
        return Math.Clamp(quality, 0f, 1f);
    }

    public static float ComputeSharpnessScore(RgbImage image)
    {
        if (image.Width < 3 || image.Height < 3)
        {
            return 0f;
        }

        var gray = ToGrayscale(image);
        var width = image.Width;
        var height = image.Height;

        double sum = 0;
        double sumSquares = 0;
        var count = 0;

        for (var y = 1; y < height - 1; y++)
        {
            for (var x = 1; x < width - 1; x++)
            {
                var center = gray[(y * width) + x];
                var left = gray[(y * width) + (x - 1)];
                var right = gray[(y * width) + (x + 1)];
                var up = gray[((y - 1) * width) + x];
                var down = gray[((y + 1) * width) + x];

                var laplacian = (4f * center) - left - right - up - down;
                sum += laplacian;
                sumSquares += laplacian * laplacian;
                count++;
            }
        }

        if (count == 0)
        {
            return 0f;
        }

        var mean = sum / count;
        var variance = (sumSquares / count) - (mean * mean);
        var varianceF = (float)Math.Max(variance, 0d);

        // Empirical normalization for 112x112 aligned crops.
        return Math.Clamp(varianceF * 25f, 0f, 1f);
    }

    public static float ComputeIlluminationScore(RgbImage image)
    {
        var gray = ToGrayscale(image);
        if (gray.Length == 0)
        {
            return 0f;
        }

        double mean = 0;
        for (var i = 0; i < gray.Length; i++)
        {
            mean += gray[i];
        }

        mean /= gray.Length;
        var score = 1f - (MathF.Abs((float)mean - 0.5f) / 0.5f);
        return Math.Clamp(score, 0f, 1f);
    }

    private static float[] ToGrayscale(RgbImage image)
    {
        var gray = new float[image.Width * image.Height];
        var src = image.Rgb;

        for (var i = 0; i < gray.Length; i++)
        {
            var offset = i * 3;
            var r = src[offset] / 255f;
            var g = src[offset + 1] / 255f;
            var b = src[offset + 2] / 255f;
            gray[i] = (0.299f * r) + (0.587f * g) + (0.114f * b);
        }

        return gray;
    }
}
