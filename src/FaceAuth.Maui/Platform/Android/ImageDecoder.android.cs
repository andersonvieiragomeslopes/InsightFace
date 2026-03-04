#if ANDROID
using Android.Graphics;
using Android.Media;

namespace FaceAuth.Maui;

internal sealed class ImageDecoder : IImageDecoder
{
    private const int OrientationNormal = 1;
    private const int OrientationRotate180 = 3;
    private const int OrientationRotate90 = 6;
    private const int OrientationRotate270 = 8;

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

        using var bitmap = BitmapFactory.DecodeByteArray(imageBytes, 0, imageBytes.Length);
        if (bitmap is null)
        {
            throw new FaceAuthException("Failed to decode image bytes on Android.");
        }

        var orientation = ReadOrientation(imageBytes);
        using var oriented = ApplyOrientation(bitmap, orientation);

        return ToRgbImage(oriented);
    }

    private static int ReadOrientation(byte[] imageBytes)
    {
        if (!OperatingSystem.IsAndroidVersionAtLeast(24))
        {
            return OrientationNormal;
        }

        try
        {
            using var stream = new MemoryStream(imageBytes);
            using var exif = new ExifInterface(stream);
            return exif.GetAttributeInt(ExifInterface.TagOrientation, OrientationNormal);
        }
        catch
        {
            return OrientationNormal;
        }
    }

    private static Bitmap ApplyOrientation(Bitmap source, int orientation)
    {
        if (orientation is not (OrientationRotate90 or OrientationRotate180 or OrientationRotate270))
        {
            var config = source.GetConfig() ?? Bitmap.Config.Argb8888!;
            return source.Copy(config, false) ?? Bitmap.CreateBitmap(source);
        }

        var matrix = new Matrix();
        switch (orientation)
        {
            case OrientationRotate90:
                matrix.PostRotate(90f);
                break;
            case OrientationRotate180:
                matrix.PostRotate(180f);
                break;
            case OrientationRotate270:
                matrix.PostRotate(270f);
                break;
        }

        return Bitmap.CreateBitmap(source, 0, 0, source.Width, source.Height, matrix, true);
    }

    private static RgbImage ToRgbImage(Bitmap bitmap)
    {
        var width = bitmap.Width;
        var height = bitmap.Height;

        var pixels = new int[width * height];
        bitmap.GetPixels(pixels, 0, width, 0, 0, width, height);

        var rgb = new byte[pixels.Length * 3];
        for (var i = 0; i < pixels.Length; i++)
        {
            var pixel = pixels[i];
            var offset = i * 3;
            rgb[offset] = (byte)((pixel >> 16) & 0xFF);
            rgb[offset + 1] = (byte)((pixel >> 8) & 0xFF);
            rgb[offset + 2] = (byte)(pixel & 0xFF);
        }

        return new RgbImage(width, height, rgb);
    }
}
#endif
