using FaceAuth.Maui;

namespace FaceAuth.Maui.Tests;

public class FaceEmbeddingSerializerTests
{
    [Fact]
    public void ToBase64_FromBase64_RoundTripPreservesValues()
    {
        var input = Enumerable.Range(1, 512)
            .Select(static i => i / 1024f)
            .ToArray();

        var b64 = FaceEmbeddingSerializer.ToBase64(input);
        var restored = FaceEmbeddingSerializer.FromBase64(b64);

        Assert.Equal(input.Length, restored.Length);
        for (var i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i], restored[i], precision: 6);
        }
    }

    [Fact]
    public void FromBase64_InvalidLength_ThrowsFormatException()
    {
        var payload = Convert.ToBase64String(new byte[] { 0x01, 0x02, 0x03 });
        Assert.Throws<FormatException>(() => FaceEmbeddingSerializer.FromBase64(payload));
    }
}
