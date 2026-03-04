#if !ANDROID && !IOS
namespace FaceAuth.Maui;

internal sealed class ImageDecoder : IImageDecoder
{
    public RgbImage Decode(byte[] imageBytes)
    {
        throw new PlatformNotSupportedException(
            "Image decoding for FaceAuth.Maui is only available on Android and iOS targets.");
    }
}
#endif
