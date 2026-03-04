namespace FaceAuth.Maui;

public sealed class MultipleFacesDetectedException : FaceAuthException
{
    public MultipleFacesDetectedException(int faceCount)
        : base($"Expected a single face, but {faceCount} faces were detected.")
    {
        FaceCount = faceCount;
    }

    public int FaceCount { get; }
}
