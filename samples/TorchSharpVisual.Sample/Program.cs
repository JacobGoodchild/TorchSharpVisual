// TorchSharpVisual sample: draws architecture diagrams for three different kinds of models —
// a plain Sequential stack, a custom class with a nested Sequential block, and a model with a
// genuine fork/join (a shared trunk feeding two branches that merge back together) — to show the
// library handling more than just a single straight chain of layers.
//
// Run from the repo root with: dotnet run --project samples/TorchSharpVisual.Sample
// Diagrams are written to samples/TorchSharpVisual.Sample/output/.

using TorchSharp;
using TorchSharpVisual.Extensions;
using TorchSharpVisual.Renderers;
using static TorchSharp.torch;

var outputDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "output");
Directory.CreateDirectory(outputDir);

Console.WriteLine("TorchSharpVisual sample — generating architecture diagrams...");
Console.WriteLine();

DrawConvNet(outputDir);
DrawCustomMultiLayerNet(outputDir);
DrawBranchingNet(outputDir);

Console.WriteLine();
Console.WriteLine($"Done. Diagrams written to: {Path.GetFullPath(outputDir)}");

static void DrawConvNet(string outputDir)
{
    // A small image classifier built entirely from a Sequential stack.
    var model = nn.Sequential(
        ("conv1", nn.Conv2d(3, 16, kernel_size: 3, padding: 1)),
        ("relu1", nn.ReLU()),
        ("pool1", nn.MaxPool2d(kernel_size: 2)),
        ("conv2", nn.Conv2d(16, 32, kernel_size: 3, padding: 1)),
        ("relu2", nn.ReLU()),
        ("pool2", nn.MaxPool2d(kernel_size: 2)),
        ("flatten", nn.Flatten()),
        ("fc", nn.Linear(32 * 8 * 8, 10)));

    var path = Path.Combine(outputDir, "conv-net.png");
    model.DrawGraph(inputShape: new long[] { 1, 3, 32, 32 }, fileName: path, modelName: "ConvNet");
    Console.WriteLine($"  wrote {path}");
}

static void DrawCustomMultiLayerNet(string outputDir)
{
    var model = new SimpleClassifier(inputFeatures: 20, hidden: 64, numClasses: 5);

    var path = Path.Combine(outputDir, "custom-mlp.svg");
    model.DrawGraph(inputShape: new long[] { 4, 20 }, fileName: path, modelName: "SimpleClassifier");
    Console.WriteLine($"  wrote {path}");
}

static void DrawBranchingNet(string outputDir)
{
    // A shared trunk feeding two sibling branches that merge back together — proves the extractor
    // tracks real tensor dataflow rather than assuming every model is one straight chain of layers.
    var model = new TwoBranchBlock(features: 32);

    var path = Path.Combine(outputDir, "branching-net.png");
    model.DrawGraph(
        inputShape: new long[] { 1, 32 },
        fileName: path,
        modelName: "TwoBranchBlock",
        options: new RenderOptions { LayoutDirection = GraphLayoutDirection.LeftToRight });
    Console.WriteLine($"  wrote {path}");
}

/// <summary>A small custom feed-forward classifier with two hidden layers.</summary>
internal sealed class SimpleClassifier : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> fc1;
    private readonly nn.Module<Tensor, Tensor> relu1;
    private readonly nn.Module<Tensor, Tensor> fc2;
    private readonly nn.Module<Tensor, Tensor> relu2;
    private readonly nn.Module<Tensor, Tensor> output;

    public SimpleClassifier(long inputFeatures, long hidden, long numClasses)
        : base(nameof(SimpleClassifier))
    {
        fc1 = nn.Linear(inputFeatures, hidden);
        relu1 = nn.ReLU();
        fc2 = nn.Linear(hidden, hidden);
        relu2 = nn.ReLU();
        output = nn.Linear(hidden, numClasses);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = fc1.call(input);
        x = relu1.call(x);
        x = fc2.call(x);
        x = relu2.call(x);
        x = output.call(x);
        return x;
    }
}

/// <summary>A shared trunk feeding two parallel branches whose outputs are added back together.</summary>
internal sealed class TwoBranchBlock : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> trunk;
    private readonly nn.Module<Tensor, Tensor> branchA;
    private readonly nn.Module<Tensor, Tensor> branchB;
    private readonly nn.Module<Tensor, Tensor> merge;

    public TwoBranchBlock(long features)
        : base(nameof(TwoBranchBlock))
    {
        trunk = nn.Linear(features, features);
        branchA = nn.Linear(features, features);
        branchB = nn.Linear(features, features);
        merge = nn.Linear(features, features);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var h = trunk.call(input);
        var a = branchA.call(h);
        var b = branchB.call(h);
        using var combined = a + b;
        return merge.call(combined);
    }
}
