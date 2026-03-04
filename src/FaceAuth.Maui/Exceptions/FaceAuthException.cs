namespace FaceAuth.Maui;

public class FaceAuthException : Exception
{
    public FaceAuthException(string message)
        : base(message)
    {
    }

    public FaceAuthException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
