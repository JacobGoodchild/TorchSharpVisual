using System;
using System.IO;
using TorchSharpVisual.Core;
using TorchSharpVisual.Renderers;

namespace TorchSharpVisual.Tests;

public class DotRendererTests
{
    private static Graph BuildSampleGraph()
    {
        var graph = new Graph { Name = "SampleModel" };
        graph.AddNode(new Node
        {
            Id = "input",
            Name = "input",
            TypeName = "Tensor",
            Kind = NodeKind.Input,
            OutputShape = new long[] { 1, 3, 224, 224 },
        });
        graph.AddNode(new Node
        {
            Id = "n0_conv1",
            Name = "conv1",
            TypeName = "Conv2d",
            Kind = NodeKind.Module,
            InputShape = new long[] { 1, 3, 224, 224 },
            OutputShape = new long[] { 1, 16, 224, 224 },
        });
        graph.AddNode(new Node
        {
            Id = "n1_relu",
            Name = "relu",
            TypeName = "ReLU",
            Kind = NodeKind.Operation,
            InputShape = new long[] { 1, 16, 224, 224 },
            OutputShape = new long[] { 1, 16, 224, 224 },
        });
        graph.AddNode(new Node
        {
            Id = "output",
            Name = "output",
            TypeName = "Tensor",
            Kind = NodeKind.Output,
            InputShape = new long[] { 1, 16, 224, 224 },
        });
        graph.AddEdge(new Edge { FromId = "input", ToId = "n0_conv1", Label = "(1, 3, 224, 224)" });
        graph.AddEdge(new Edge { FromId = "n0_conv1", ToId = "n1_relu", Label = "(1, 16, 224, 224)" });
        graph.AddEdge(new Edge { FromId = "n1_relu", ToId = "output", Label = "(1, 16, 224, 224)" });
        return graph;
    }

    [Fact]
    public void ToDot_ProducesValidDigraphWithAllNodesAndEdges()
    {
        var dot = DotRenderer.ToDot(BuildSampleGraph());

        Assert.StartsWith("digraph \"SampleModel\"", dot);
        Assert.Contains("\"input\"", dot);
        Assert.Contains("\"n0_conv1\"", dot);
        Assert.Contains("\"n1_relu\"", dot);
        Assert.Contains("\"output\"", dot);
        Assert.Contains("\"input\" -> \"n0_conv1\"", dot);
        Assert.Contains("\"n0_conv1\" -> \"n1_relu\"", dot);
        Assert.Contains("\"n1_relu\" -> \"output\"", dot);
        Assert.EndsWith("}\n", dot);
    }

    [Fact]
    public void ToDot_UsesTaskSpecifiedColorSchemeForEachNodeKind()
    {
        var dot = DotRenderer.ToDot(BuildSampleGraph());

        Assert.Contains("fillcolor=\"#FFF2CC\"", dot); // input/output
        Assert.Contains("fillcolor=\"#D5E8D4\"", dot); // module
        Assert.Contains("fillcolor=\"#DAE8FC\"", dot); // operation
    }

    [Fact]
    public void ToDot_EscapesQuotesAndBackslashesInLabels()
    {
        var graph = new Graph { Name = "Weird\"Name" };
        graph.AddNode(new Node { Id = "a", Name = "a\\b", TypeName = "Custom\"Type", Kind = NodeKind.Module });

        var dot = DotRenderer.ToDot(graph);

        Assert.Contains("Weird\\\"Name", dot);
        Assert.Contains("a\\\\b", dot);
        Assert.Contains("Custom\\\"Type", dot);
    }

    [Fact]
    public void Render_DotExtension_WritesRawDotSourceWithoutInvokingGraphviz()
    {
        var path = Path.Combine(Path.GetTempPath(), $"tsv_{Guid.NewGuid():N}.dot");
        try
        {
            DotRenderer.Render(BuildSampleGraph(), path);

            Assert.True(File.Exists(path));
            var content = File.ReadAllText(path);
            Assert.StartsWith("digraph", content);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Render_UnsupportedExtension_ThrowsNotSupportedException()
    {
        var path = Path.Combine(Path.GetTempPath(), $"tsv_{Guid.NewGuid():N}.jpg");
        Assert.Throws<NotSupportedException>(() => DotRenderer.Render(BuildSampleGraph(), path));
    }

    [Fact]
    public void Render_PngExtension_InvokesGraphvizAndProducesAValidPngFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"tsv_{Guid.NewGuid():N}.png");
        try
        {
            DotRenderer.Render(BuildSampleGraph(), path);

            Assert.True(File.Exists(path));
            var bytes = File.ReadAllBytes(path);
            Assert.True(bytes.Length > 8);
            // PNG magic number: 89 50 4E 47 0D 0A 1A 0A
            Assert.Equal(0x89, bytes[0]);
            Assert.Equal((byte)'P', bytes[1]);
            Assert.Equal((byte)'N', bytes[2]);
            Assert.Equal((byte)'G', bytes[3]);
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
    public void Render_SvgExtension_InvokesGraphvizAndProducesSvgMarkup()
    {
        var path = Path.Combine(Path.GetTempPath(), $"tsv_{Guid.NewGuid():N}.svg");
        try
        {
            DotRenderer.Render(BuildSampleGraph(), path);

            Assert.True(File.Exists(path));
            var content = File.ReadAllText(path);
            Assert.Contains("<svg", content);
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
    public void Render_MissingGraphvizExecutable_ThrowsInvalidOperationExceptionWithHelpfulMessage()
    {
        var path = Path.Combine(Path.GetTempPath(), $"tsv_{Guid.NewGuid():N}.png");
        var ex = Assert.Throws<InvalidOperationException>(() =>
            DotRenderer.Render(BuildSampleGraph(), path, dotExecutable: "definitely-not-a-real-dot-binary"));

        Assert.Contains("Graphviz", ex.Message);
    }
}
