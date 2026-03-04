namespace FaceAuth.Maui;

public sealed class FaceNotFoundException : FaceAuthException
{
    public FaceNotFoundException()
        : base("No face was detected in the provided image.")
    {
    }
}
