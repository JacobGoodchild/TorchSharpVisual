using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharpVisual.Tests;

/// <summary>A minimal two-layer MLP with a ReLU activation, used to exercise custom forward passes.</summary>
internal sealed class SimpleMlp : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> fc1;
    private readonly nn.Module<Tensor, Tensor> relu;
    private readonly nn.Module<Tensor, Tensor> fc2;

    public SimpleMlp(long inputFeatures, long hidden, long outputFeatures)
        : base(nameof(SimpleMlp))
    {
        fc1 = nn.Linear(inputFeatures, hidden);
        relu = nn.ReLU();
        fc2 = nn.Linear(hidden, outputFeatures);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = fc1.call(input);
        x = relu.call(x);
        x = fc2.call(x);
        return x;
    }
}

/// <summary>A small convolutional network: Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear.</summary>
internal sealed class SimpleConvNet : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> conv1;
    private readonly nn.Module<Tensor, Tensor> relu;
    private readonly nn.Module<Tensor, Tensor> pool;
    private readonly nn.Module<Tensor, Tensor> flatten;
    private readonly nn.Module<Tensor, Tensor> fc;

    public SimpleConvNet(long inputChannels, long numClasses)
        : base(nameof(SimpleConvNet))
    {
        conv1 = nn.Conv2d(inputChannels, 8, kernel_size: 3, padding: 1);
        relu = nn.ReLU();
        pool = nn.MaxPool2d(kernel_size: 2);
        flatten = nn.Flatten();
        fc = nn.Linear(8 * 4 * 4, numClasses);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = conv1.call(input);
        x = relu.call(x);
        x = pool.call(x);
        x = flatten.call(x);
        x = fc.call(x);
        return x;
    }
}

/// <summary>A model built entirely from a single top-level <c>Sequential</c> container.</summary>
internal static class SequentialModels
{
    public static nn.Module<Tensor, Tensor> BuildMlp(long inputFeatures, long hidden, long outputFeatures) =>
        nn.Sequential(
            ("fc1", nn.Linear(inputFeatures, hidden)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(hidden, outputFeatures)));
}

/// <summary>A model that nests a <c>Sequential</c> block inside a custom module, to exercise recursive extraction.</summary>
internal sealed class NestedModel : nn.Module<Tensor, Tensor>
{
    private readonly nn.Module<Tensor, Tensor> features;
    private readonly nn.Module<Tensor, Tensor> classifier;

    public NestedModel(long inputFeatures, long hidden, long outputFeatures)
        : base(nameof(NestedModel))
    {
        features = nn.Sequential(
            ("fc1", nn.Linear(inputFeatures, hidden)),
            ("relu", nn.ReLU()));
        classifier = nn.Linear(hidden, outputFeatures);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = features.call(input);
        x = classifier.call(x);
        return x;
    }
}
