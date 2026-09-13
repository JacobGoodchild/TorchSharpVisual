using System;
using System.IO;
using TorchSharp;
using TorchSharpVisual.Extensions;
using static TorchSharp.torch;

namespace TorchSharpVisual.Tests;

public class ModuleExtensionsTests
{
    [Fact]
    public void ToDotGraph_ReturnsGraphvizSourceWithoutTouchingDisk()
    {
        var model = new SimpleMlp(inputFeatures: 4, hidden: 8, outputFeatures: 2);

        var dot = model.ToDotGraph(new long[] { 1, 4 });

        Assert.StartsWith("digraph", dot);
        Assert.Contains("Linear", dot);
        Assert.Contains("ReLU", dot);
    }

    [Fact]
    public void DrawGraph_EndToEnd_WritesAValidPngFile()
    {
        var model = new SimpleConvNet(inputChannels: 3, numClasses: 10);
        var path = Path.Combine(Path.GetTempPath(), $"tsv_end_to_end_{Guid.NewGuid():N}.png");

        try
        {
            model.DrawGraph(inputShape: new long[] { 1, 3, 8, 8 }, fileName: path);

            Assert.True(File.Exists(path));
            Assert.True(new FileInfo(path).Length > 0);
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public void DrawGraph_UnsupportedModuleShape_ThrowsNotSupportedException()
    {
        var bilinear = nn.Bilinear(4, 4, 2);
        var path = Path.Combine(Path.GetTempPath(), $"tsv_{Guid.NewGuid():N}.dot");

        Assert.Throws<NotSupportedException>(() =>
            bilinear.DrawGraph(new long[] { 1, 4 }, path));
    }
}
