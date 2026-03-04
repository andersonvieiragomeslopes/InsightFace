using FaceAuth.Maui;
using Microsoft.Maui.ApplicationModel;
using Microsoft.Maui.Media;
using Microsoft.Maui.Storage;

namespace FaceAuth.SampleApp;

public partial class MainPage : ContentPage
{
    private readonly IFaceAuthService _faceAuthService = new FaceAuthService();

    private float[]? _referenceEmbedding;
    private float[]? _probeEmbedding;

    public MainPage()
    {
        InitializeComponent();
    }

    private async void OnInitializeClicked(object? sender, EventArgs e)
    {
        try
        {
            await _faceAuthService.InitializeAsync(new FaceAuthOptions
            {
                MatchThreshold = 0.55f,
                RequireSingleFace = true,
                MinDetectionConfidence = 0.70f,
                MinQualityScore = 0.55f,
                ThrowOnLowQuality = true,
            });

            StatusLabel.Text = "Status: initialized.";
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Status: init failed - {ex.Message}";
        }
    }

    private async void OnEnrollCameraClicked(object? sender, EventArgs e) => await EnrollAsync(fromCamera: true);

    private async void OnEnrollGalleryClicked(object? sender, EventArgs e) => await EnrollAsync(fromCamera: false);

    private async void OnProbeCameraClicked(object? sender, EventArgs e) => await ProbeAsync(fromCamera: true);

    private async void OnProbeGalleryClicked(object? sender, EventArgs e) => await ProbeAsync(fromCamera: false);

    private void OnCompareClicked(object? sender, EventArgs e)
    {
        try
        {
            if (_referenceEmbedding is null || _probeEmbedding is null)
            {
                ResultLabel.Text = "Result: both reference and probe embeddings are required.";
                return;
            }

            var match = _faceAuthService.Compare(_probeEmbedding, _referenceEmbedding);
            ResultLabel.Text = $"Result: score={match.Score:F4} | isMatch={match.IsMatch}";
        }
        catch (Exception ex)
        {
            ResultLabel.Text = $"Result: compare failed - {ex.Message}";
        }
    }

    private async Task EnrollAsync(bool fromCamera)
    {
        try
        {
            var bytes = await PickImageBytesAsync(fromCamera);
            if (bytes is null)
            {
                StatusLabel.Text = "Status: enroll canceled.";
                return;
            }

            var result = await _faceAuthService.ExtractEmbeddingAsync(bytes);
            _referenceEmbedding = result.Embedding;

            var serialized = FaceEmbeddingSerializer.ToBase64(result.Embedding);
            ReferenceLabel.Text = Preview(serialized);

            StatusLabel.Text =
                $"Status: enroll ok | conf={result.DetectionConfidence:F3} | quality={result.QualityScore:F3} | emb={result.Embedding.Length}";
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Status: enroll failed - {ex.Message}";
        }
    }

    private async Task ProbeAsync(bool fromCamera)
    {
        try
        {
            var bytes = await PickImageBytesAsync(fromCamera);
            if (bytes is null)
            {
                StatusLabel.Text = "Status: probe canceled.";
                return;
            }

            var result = await _faceAuthService.ExtractEmbeddingAsync(bytes);
            _probeEmbedding = result.Embedding;

            var serialized = FaceEmbeddingSerializer.ToBase64(result.Embedding);
            ProbeLabel.Text = Preview(serialized);

            StatusLabel.Text =
                $"Status: probe ok | conf={result.DetectionConfidence:F3} | quality={result.QualityScore:F3} | emb={result.Embedding.Length}";
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Status: probe failed - {ex.Message}";
        }
    }

    private static async Task<byte[]?> PickImageBytesAsync(bool fromCamera)
    {
        FileResult? file;
        if (fromCamera)
        {
            if (!MediaPicker.Default.IsCaptureSupported)
            {
                throw new InvalidOperationException("Camera capture is not supported on this device.");
            }

            file = await MediaPicker.Default.CapturePhotoAsync();
        }
        else
        {
            file = await MediaPicker.Default.PickPhotoAsync();
        }

        if (file is null)
        {
            return null;
        }

        await using var stream = await file.OpenReadAsync();
        using var memory = new MemoryStream();
        await stream.CopyToAsync(memory);
        return memory.ToArray();
    }

    private static string Preview(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return "(empty)";
        }

        return text.Length <= 96
            ? text
            : $"{text[..96]}...";
    }
}
