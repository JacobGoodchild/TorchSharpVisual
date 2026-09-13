using System.Collections.Generic;

namespace TorchSharpVisual.Core;

/// <summary>
/// A single box in the architecture diagram: a tensor (input/output) or a layer/operation.
/// </summary>
public sealed class Node
{
    /// <summary>Unique identifier used to reference this node from an <see cref="Edge"/>.</summary>
    public required string Id { get; init; }

    /// <summary>The instance name of the module (e.g. <c>"fc1"</c>, or its dotted path such as <c>"features.0"</c>).</summary>
    public required string Name { get; init; }

    /// <summary>The CLR type name of the module or tensor this node represents (e.g. <c>"Linear"</c>, <c>"Conv2d"</c>, <c>"Tensor"</c>).</summary>
    public required string TypeName { get; init; }

    /// <summary>How this node should be classified for styling purposes.</summary>
    public required NodeKind Kind { get; init; }

    /// <summary>The shape of the tensor flowing into this node, if known.</summary>
    public long[]? InputShape { get; init; }

    /// <summary>The shape of the tensor flowing out of this node, if known.</summary>
    public long[]? OutputShape { get; init; }

    /// <summary>The number of learnable scalar parameters owned by this layer (0 for input/output/stateless operation nodes).</summary>
    public long ParameterCount { get; init; }

    /// <summary>
    /// The dotted path of the immediate container module this node lives in (e.g. <c>"features"</c>
    /// for a node named <c>"features.0"</c>), or an empty string when the node sits directly under
    /// the model root. Renderers may use this to group related nodes visually.
    /// </summary>
    public string ParentPath { get; init; } = string.Empty;

    /// <summary>Extra descriptive attributes surfaced from the underlying module (e.g. <c>in_features</c>, <c>kernel_size</c>).</summary>
    public IReadOnlyDictionary<string, string> Attributes { get; init; } = new Dictionary<string, string>();
}
