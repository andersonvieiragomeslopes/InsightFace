using System.Runtime.InteropServices;

namespace FaceAuth.Maui;

public static class FaceEmbeddingSerializer
{
    public static string ToBase64(float[] embedding)
    {
        if (embedding is null)
        {
            throw new ArgumentNullException(nameof(embedding));
        }

        if (embedding.Length == 0)
        {
            throw new ArgumentException("Embedding cannot be empty.", nameof(embedding));
        }

        var byteCount = embedding.Length * sizeof(float);
        var bytes = new byte[byteCount];

        if (BitConverter.IsLittleEndian)
        {
            embedding.AsSpan().CopyTo(MemoryMarshal.Cast<byte, float>(bytes.AsSpan()));
            return Convert.ToBase64String(bytes);
        }

        for (var i = 0; i < embedding.Length; i++)
        {
            var raw = BitConverter.GetBytes(embedding[i]);
            Array.Reverse(raw);
            Buffer.BlockCopy(raw, 0, bytes, i * sizeof(float), sizeof(float));
        }

        return Convert.ToBase64String(bytes);
    }

    public static float[] FromBase64(string b64)
    {
        if (string.IsNullOrWhiteSpace(b64))
        {
            throw new ArgumentException("Base64 embedding cannot be empty.", nameof(b64));
        }

        var bytes = Convert.FromBase64String(b64);
        if (bytes.Length == 0 || bytes.Length % sizeof(float) != 0)
        {
            throw new FormatException("Invalid embedding byte length. Expected multiple of 4 bytes.");
        }

        var count = bytes.Length / sizeof(float);
        var values = new float[count];

        if (BitConverter.IsLittleEndian)
        {
            MemoryMarshal.Cast<byte, float>(bytes.AsSpan()).CopyTo(values);
            return values;
        }

        for (var i = 0; i < count; i++)
        {
            var offset = i * sizeof(float);
            var raw = new byte[sizeof(float)];
            Buffer.BlockCopy(bytes, offset, raw, 0, sizeof(float));
            Array.Reverse(raw);
            values[i] = BitConverter.ToSingle(raw, 0);
        }

        return values;
    }
}
