using System.Linq;

namespace TorchSharpVisual.Core;

/// <summary>
/// Formats tensor shapes for display, e.g. <c>(1, 3, 224, 224)</c>.
/// </summary>
public static class ShapeFormatter
{
    /// <summary>Formats a shape array as <c>(d0, d1, ...)</c>, or <c>"?"</c> when the shape is unknown.</summary>
    public static string Format(long[]? shape) =>
        shape is null ? "?" : $"({string.Join(", ", shape.Select(d => d.ToString()))})";
}
