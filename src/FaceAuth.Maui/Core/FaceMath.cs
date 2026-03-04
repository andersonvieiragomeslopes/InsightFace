namespace FaceAuth.Maui;

public static class FaceMath
{
    public static float[] NormalizeL2(float[] vector)
    {
        if (vector is null)
        {
            throw new ArgumentNullException(nameof(vector));
        }

        if (vector.Length == 0)
        {
            throw new ArgumentException("Embedding vector cannot be empty.", nameof(vector));
        }

        var sumSquares = 0f;
        for (var i = 0; i < vector.Length; i++)
        {
            sumSquares += vector[i] * vector[i];
        }

        if (sumSquares <= 0f)
        {
            throw new ArgumentException("Embedding vector norm cannot be zero.", nameof(vector));
        }

        var norm = MathF.Sqrt(sumSquares);
        var normalized = new float[vector.Length];
        for (var i = 0; i < vector.Length; i++)
        {
            normalized[i] = vector[i] / norm;
        }

        return normalized;
    }

    public static float CosineSimilarity(float[] left, float[] right)
    {
        if (left is null)
        {
            throw new ArgumentNullException(nameof(left));
        }

        if (right is null)
        {
            throw new ArgumentNullException(nameof(right));
        }

        if (left.Length == 0 || right.Length == 0)
        {
            throw new ArgumentException("Embedding vectors cannot be empty.");
        }

        if (left.Length != right.Length)
        {
            throw new ArgumentException("Embedding vectors must have the same size.");
        }

        var leftNorm = IsApproximatelyL2Normalized(left) ? left : NormalizeL2(left);
        var rightNorm = IsApproximatelyL2Normalized(right) ? right : NormalizeL2(right);

        var dot = 0f;
        for (var i = 0; i < leftNorm.Length; i++)
        {
            dot += leftNorm[i] * rightNorm[i];
        }

        return Math.Clamp(dot, -1f, 1f);
    }

    internal static bool IsApproximatelyL2Normalized(float[] vector, float tolerance = 1e-3f)
    {
        if (vector.Length == 0)
        {
            return false;
        }

        var sumSquares = 0f;
        for (var i = 0; i < vector.Length; i++)
        {
            sumSquares += vector[i] * vector[i];
        }

        return MathF.Abs(1f - sumSquares) <= tolerance;
    }
}
