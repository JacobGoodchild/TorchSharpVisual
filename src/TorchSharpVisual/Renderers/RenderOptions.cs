namespace TorchSharpVisual.Renderers;

/// <summary>
/// Controls how <see cref="DotRenderer"/> lays out and decorates a diagram. All properties have
/// sensible defaults, so most callers only need to override the ones they care about.
/// </summary>
public sealed class RenderOptions
{
    /// <summary>The default options used when none are supplied.</summary>
    public static RenderOptions Default { get; } = new();

    /// <summary>The direction layers flow in the diagram. Defaults to top-to-bottom.</summary>
    public GraphLayoutDirection LayoutDirection { get; init; } = GraphLayoutDirection.TopToBottom;

    /// <summary>Whether to draw a legend explaining the node color scheme. Defaults to <c>true</c>.</summary>
    public bool ShowLegend { get; init; } = true;

    /// <summary>Whether to show each layer's learnable parameter count on its node. Defaults to <c>true</c>.</summary>
    public bool ShowParameterCounts { get; init; } = true;

    /// <summary>
    /// Whether to show a summary header above the diagram with the model name, layer count, and total
    /// parameter count. Defaults to <c>true</c>.
    /// </summary>
    public bool ShowSummaryHeader { get; init; } = true;

    /// <summary>The image resolution (dots per inch) used for raster output like PNG. Defaults to <c>96</c>.</summary>
    public int Dpi { get; init; } = 96;

    /// <summary>The font family used for node and edge labels. Defaults to <c>"Helvetica,Arial,sans-serif"</c>.</summary>
    public string FontName { get; init; } = "Helvetica,Arial,sans-serif";
}
