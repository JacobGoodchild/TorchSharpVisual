namespace TorchSharpVisual.Renderers;

/// <summary>The direction layers flow in a rendered diagram.</summary>
public enum GraphLayoutDirection
{
    /// <summary>Top-to-bottom (the default — reads like a typical layer stack).</summary>
    TopToBottom,

    /// <summary>Left-to-right (useful for wide models with many layers, or landscape slides).</summary>
    LeftToRight,
}
