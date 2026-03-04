namespace FaceAuth.Maui;

public sealed class FaceAuthNotInitializedException : FaceAuthException
{
    public FaceAuthNotInitializedException()
        : base("FaceAuthService is not initialized. Call InitializeAsync first.")
    {
    }
}
