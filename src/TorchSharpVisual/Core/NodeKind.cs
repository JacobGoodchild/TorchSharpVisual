namespace TorchSharpVisual.Core;

/// <summary>
/// Classifies a <see cref="Node"/> so renderers can apply consistent, meaningful styling.
/// </summary>
public enum NodeKind
{
    /// <summary>The tensor(s) fed into the model.</summary>
    Input,

    /// <summary>The tensor(s) produced by the model.</summary>
    Output,

    /// <summary>
    /// A learnable layer such as <c>Linear</c>, <c>Conv2d</c>, or <c>BatchNorm2d</c> — anything that
    /// owns trainable parameters.
    /// </summary>
    Module,

    /// <summary>
    /// A stateless operation such as an activation function, pooling layer, or reshape — anything
    /// without trainable parameters.
    /// </summary>
    Operation,
}
