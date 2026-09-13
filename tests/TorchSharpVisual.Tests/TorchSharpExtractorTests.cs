using System;
using System.Linq;
using TorchSharp;
using TorchSharpVisual.Core;
using TorchSharpVisual.Extractors;
using static TorchSharp.torch;

namespace TorchSharpVisual.Tests;

public class TorchSharpExtractorTests
{
    [Fact]
    public void Extract_SingleLinearModule_ProducesInputLinearAndOutputNodes()
    {
        var linear = nn.Linear(10, 5);

        var graph = TorchSharpExtractor.Extract(linear, new long[] { 2, 10 });

        Assert.Equal(3, graph.Nodes.Count);
        Assert.Equal(NodeKind.Input, graph.Nodes[0].Kind);
        Assert.Equal(NodeKind.Module, graph.Nodes[1].Kind);
        Assert.Equal("Linear", graph.Nodes[1].TypeName);
        Assert.Equal(NodeKind.Output, graph.Nodes[2].Kind);

        Assert.Equal(new long[] { 2, 10 }, graph.Nodes[1].InputShape);
        Assert.Equal(new long[] { 2, 5 }, graph.Nodes[1].OutputShape);
        Assert.Equal(2, graph.Edges.Count);
    }

    [Fact]
    public void Extract_Sequential_ProducesNodesInDefinitionOrder()
    {
        var model = SequentialModels.BuildMlp(10, 20, 3);

        var graph = TorchSharpExtractor.Extract(model, new long[] { 4, 10 });

        // input, fc1, relu, fc2, output
        Assert.Equal(5, graph.Nodes.Count);
        Assert.Equal("input", graph.Nodes[0].Name);
        Assert.Equal("Linear", graph.Nodes[1].TypeName);
        Assert.Equal("ReLU", graph.Nodes[2].TypeName);
        Assert.Equal("Linear", graph.Nodes[3].TypeName);
        Assert.Equal("output", graph.Nodes[4].Name);

        Assert.Equal(new long[] { 4, 20 }, graph.Nodes[1].OutputShape);
        Assert.Equal(new long[] { 4, 20 }, graph.Nodes[2].OutputShape);
        Assert.Equal(new long[] { 4, 3 }, graph.Nodes[3].OutputShape);
    }

    [Fact]
    public void Extract_CustomMultiLayerModule_TracksShapesAcrossReluAndClassifiesKinds()
    {
        var model = new SimpleMlp(inputFeatures: 8, hidden: 16, outputFeatures: 4);

        var graph = TorchSharpExtractor.Extract(model, new long[] { 3, 8 }, modelName: "MyMlp");

        Assert.Equal("MyMlp", graph.Name);

        var fc1 = graph.Nodes.Single(n => n.Name == "fc1");
        var relu = graph.Nodes.Single(n => n.Name == "relu");
        var fc2 = graph.Nodes.Single(n => n.Name == "fc2");

        Assert.Equal(NodeKind.Module, fc1.Kind); // has learnable parameters
        Assert.Equal(NodeKind.Operation, relu.Kind); // stateless activation
        Assert.Equal(NodeKind.Module, fc2.Kind);

        Assert.Equal(new long[] { 3, 8 }, fc1.InputShape);
        Assert.Equal(new long[] { 3, 16 }, fc1.OutputShape);
        Assert.Equal(new long[] { 3, 16 }, relu.InputShape);
        Assert.Equal(new long[] { 3, 16 }, relu.OutputShape);
        Assert.Equal(new long[] { 3, 4 }, fc2.OutputShape);
    }

    [Fact]
    public void Extract_Conv2dNetwork_ProducesCorrectShapesThroughPoolAndFlatten()
    {
        var model = new SimpleConvNet(inputChannels: 3, numClasses: 10);

        var graph = TorchSharpExtractor.Extract(model, new long[] { 1, 3, 8, 8 });

        var conv = graph.Nodes.Single(n => n.TypeName == "Conv2d");
        var pool = graph.Nodes.Single(n => n.TypeName == "MaxPool2d");
        var flatten = graph.Nodes.Single(n => n.TypeName == "Flatten");
        var fc = graph.Nodes.Single(n => n.TypeName == "Linear");

        Assert.Equal(new long[] { 1, 3, 8, 8 }, conv.InputShape);
        Assert.Equal(new long[] { 1, 8, 8, 8 }, conv.OutputShape); // padding=1 keeps spatial size
        Assert.Equal(new long[] { 1, 8, 4, 4 }, pool.OutputShape); // 2x2 max pool halves spatial dims
        Assert.Equal(new long[] { 1, 128 }, flatten.OutputShape); // 8 * 4 * 4
        Assert.Equal(new long[] { 1, 10 }, fc.OutputShape);

        Assert.Equal(NodeKind.Module, conv.Kind);
        Assert.Equal(NodeKind.Operation, pool.Kind);
        Assert.Equal(NodeKind.Operation, flatten.Kind);
    }

    [Fact]
    public void Extract_NestedSequentialInsideCustomModule_AssignsParentPathToNestedNodes()
    {
        var model = new NestedModel(inputFeatures: 5, hidden: 6, outputFeatures: 2);

        var graph = TorchSharpExtractor.Extract(model, new long[] { 1, 5 });

        var fc1 = graph.Nodes.Single(n => n.Name == "features.fc1");
        var relu = graph.Nodes.Single(n => n.Name == "features.relu");
        var classifier = graph.Nodes.Single(n => n.Name == "classifier");

        Assert.Equal("features", fc1.ParentPath);
        Assert.Equal("features", relu.ParentPath);
        Assert.Equal(string.Empty, classifier.ParentPath);
    }

    [Fact]
    public void Extract_LinearModule_CapturesInFeaturesAndOutFeaturesAttributes()
    {
        var linear = nn.Linear(10, 5);

        var graph = TorchSharpExtractor.Extract(linear, new long[] { 1, 10 });

        var node = graph.Nodes.Single(n => n.TypeName == "Linear");
        Assert.Equal("10", node.Attributes["in_features"]);
        Assert.Equal("5", node.Attributes["out_features"]);
    }

    [Fact]
    public void Extract_NullModel_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => TorchSharpExtractor.Extract(null!, new long[] { 1 }));
    }

    [Fact]
    public void Extract_EmptyInputShape_ThrowsArgumentException()
    {
        var linear = nn.Linear(10, 5);
        Assert.Throws<ArgumentException>(() => TorchSharpExtractor.Extract(linear, Array.Empty<long>()));
    }
}
