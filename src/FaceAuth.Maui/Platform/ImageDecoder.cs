namespace FaceAuth.Maui;

internal interface IImageDecoder
{
    RgbImage Decode(byte[] imageBytes);
}

internal static class ImageDecoderFactory
{
    public static IImageDecoder Create() => new ImageDecoder();
}
