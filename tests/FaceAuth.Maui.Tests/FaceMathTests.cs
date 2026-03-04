using FaceAuth.Maui;

namespace FaceAuth.Maui.Tests;

public class FaceMathTests
{
    [Fact]
    public void NormalizeL2_Twice_IsIdempotent()
    {
        var vector = Enumerable.Range(1, 512).Select(static x => (float)x).ToArray();

        var normalizedOnce = FaceMath.NormalizeL2(vector);
        var normalizedTwice = FaceMath.NormalizeL2(normalizedOnce);

        for (var i = 0; i < normalizedOnce.Length; i++)
        {
            Assert.Equal(normalizedOnce[i], normalizedTwice[i], precision: 5);
        }
    }

    [Fact]
    public void CosineSimilarity_IdenticalVectors_ReturnsOne()
    {
        var vector = FaceMath.NormalizeL2(Enumerable.Range(1, 512).Select(static x => (float)x).ToArray());
        var score = FaceMath.CosineSimilarity(vector, vector);

        Assert.InRange(score, 0.999f, 1.0001f);
    }

    [Fact]
    public void CosineSimilarity_OppositeVectors_ReturnsMinusOne()
    {
        var vector = FaceMath.NormalizeL2(Enumerable.Range(1, 512).Select(static x => (float)x).ToArray());
        var opposite = vector.Select(static x => -x).ToArray();

        var score = FaceMath.CosineSimilarity(vector, opposite);
        Assert.InRange(score, -1.0001f, -0.999f);
    }

    [Fact]
    public void CosineSimilarity_OrthogonalVectors_IsNearZero()
    {
        var left = new float[512];
        var right = new float[512];
        left[0] = 1f;
        right[1] = 1f;

        var score = FaceMath.CosineSimilarity(left, right);
        Assert.InRange(score, -0.001f, 0.001f);
    }
}
