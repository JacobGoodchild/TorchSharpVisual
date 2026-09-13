using TorchSharp;
using TorchSharpVisual.Extractors;
using TorchSharpVisual.Renderers;

namespace TorchSharpVisual.Extensions;

/// <summary>
/// Convenience extension methods for drawing architecture diagrams directly from a
/// <see cref="TorchSharp.torch.nn.Module"/> instance.
/// </summary>
public static class ModuleExtensions
{
    /// <summary>
    /// Inspects this model with a dummy input of shape <paramref name="inputShape"/> and renders its
    /// architecture diagram to <paramref name="fileName"/>.
    /// </summary>
    /// <param name="model">The model to visualize.</param>
    /// <param name="inputShape">The shape of the dummy input tensor, e.g. <c>new long[] { 1, 3, 224, 224 }</c>.</param>
    /// <param name="fileName">
    /// Destination file path. The extension selects the output format: <c>.png</c> or <c>.svg</c> render
    /// an image via the local Graphviz <c>dot</c> executable; <c>.dot</c>/<c>.gv</c> write raw Graphviz
    /// source with no external dependency.
    /// </param>
    /// <param name="modelName">An optional display name for the diagram; defaults to the model's type name.</param>
    /// <param name="options">Layout and styling options (layout direction, legend, parameter counts, DPI). Defaults to <see cref="RenderOptions.Default"/>.</param>
    /// <param name="dotExecutable">The name or path of the Graphviz executable to use for image formats.</param>
    /// <example>
    /// <code>
    /// var model = new MyConvNet();
    /// model.DrawGraph(inputShape: new long[] { 1, 3, 224, 224 }, fileName: "architecture.png");
    /// </code>
    /// </example>
    public static void DrawGraph(
        this torch.nn.Module model,
        long[] inputShape,
        string fileName,
        string? modelName = null,
        RenderOptions? options = null,
        string dotExecutable = "dot")
    {
        var graph = TorchSharpExtractor.Extract(model, inputShape, modelName);
        DotRenderer.Render(graph, fileName, options ?? RenderOptions.Default, dotExecutable);
    }

    /// <summary>
    /// Inspects this model with a dummy input of shape <paramref name="inputShape"/> and returns its
    /// architecture diagram as raw Graphviz DOT source, without touching the file system or requiring
    /// Graphviz to be installed.
    /// </summary>
    /// <param name="model">The model to visualize.</param>
    /// <param name="inputShape">The shape of the dummy input tensor, e.g. <c>new long[] { 1, 3, 224, 224 }</c>.</param>
    /// <param name="modelName">An optional display name for the diagram; defaults to the model's type name.</param>
    /// <param name="options">Layout and styling options. Defaults to <see cref="RenderOptions.Default"/>.</param>
    public static string ToDotGraph(this torch.nn.Module model, long[] inputShape, string? modelName = null, RenderOptions? options = null)
    {
        var graph = TorchSharpExtractor.Extract(model, inputShape, modelName);
        return DotRenderer.ToDot(graph, options ?? RenderOptions.Default);
    }
}
