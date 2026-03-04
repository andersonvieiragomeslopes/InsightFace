#if IOS
using CoreGraphics;
using Foundation;
using UIKit;

namespace FaceAuth.Maui;

internal sealed class ImageDecoder : IImageDecoder
{
    public RgbImage Decode(byte[] imageBytes)
    {
        if (imageBytes is null)
        {
            throw new ArgumentNullException(nameof(imageBytes));
        }

        if (imageBytes.Length == 0)
        {
            throw new ArgumentException("Image payload cannot be empty.", nameof(imageBytes));
        }

        using var data = NSData.FromArray(imageBytes);
        using var loaded = UIImage.LoadFromData(data);
        if (loaded is null)
        {
            throw new FaceAuthException("Failed to decode image bytes on iOS.");
        }

        var normalized = NormalizeOrientation(loaded);
        try
        {
            var cgImage = normalized.CGImage ?? throw new FaceAuthException("Unable to get CGImage from decoded UIImage.");

            var width = (int)cgImage.Width;
            var height = (int)cgImage.Height;
            var bytesPerPixel = 4;
            var bytesPerRow = width * bytesPerPixel;

            var rgba = new byte[bytesPerRow * height];
            using var colorSpace = CGColorSpace.CreateDeviceRGB();
            using var context = new CGBitmapContext(
                rgba,
                width,
                height,
                8,
                bytesPerRow,
                colorSpace,
                CGBitmapFlags.ByteOrder32Big | CGBitmapFlags.PremultipliedLast);

            context.DrawImage(new CGRect(0, 0, width, height), cgImage);

            var rgb = new byte[width * height * 3];
            for (var i = 0; i < width * height; i++)
            {
                var srcOffset = i * 4;
                var dstOffset = i * 3;
                rgb[dstOffset] = rgba[srcOffset];
                rgb[dstOffset + 1] = rgba[srcOffset + 1];
                rgb[dstOffset + 2] = rgba[srcOffset + 2];
            }

            return new RgbImage(width, height, rgb);
        }
        finally
        {
            if (!ReferenceEquals(normalized, loaded))
            {
                normalized.Dispose();
            }
        }
    }

    private static UIImage NormalizeOrientation(UIImage image)
    {
        if (image.Orientation == UIImageOrientation.Up)
        {
            return image;
        }

        using var renderer = new UIGraphicsImageRenderer(image.Size);
        var rendered = renderer.CreateImage(_ =>
        {
            image.Draw(new CGRect(0, 0, image.Size.Width, image.Size.Height));
        });

        return rendered ?? image;
    }
}
#endif
