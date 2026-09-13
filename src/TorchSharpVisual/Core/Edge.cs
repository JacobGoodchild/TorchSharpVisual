namespace TorchSharpVisual.Core;

/// <summary>
/// A directed connection between two nodes in the diagram, representing tensor flow.
/// </summary>
public sealed class Edge
{
    /// <summary>The <see cref="Node.Id"/> the tensor flows from.</summary>
    public required string FromId { get; init; }

    /// <summary>The <see cref="Node.Id"/> the tensor flows into.</summary>
    public required string ToId { get; init; }

    /// <summary>An optional label to draw on the edge, typically the tensor shape flowing across it.</summary>
    public string? Label { get; init; }
}
