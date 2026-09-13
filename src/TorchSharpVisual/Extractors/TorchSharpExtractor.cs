using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using TorchSharp;
using TorchSharpVisual.Core;
using static TorchSharp.torch;

namespace TorchSharpVisual.Extractors;

/// <summary>
/// Inspects a <see cref="TorchSharp.torch.nn.Module"/> and turns it into a renderer-agnostic
/// <see cref="Graph"/> describing its layers, their parameter counts, and the tensor shapes flowing
/// between them.
/// </summary>
/// <remarks>
/// <para>
/// The extractor works by recursively walking the module tree to find "leaf" modules (those with no
/// children of their own) and attaching a forward hook to each one. It then runs a single dummy
/// forward pass with a zero-filled tensor of the requested input shape.
/// </para>
/// <para>
/// Edges are built from actual tensor identity, not execution order: when a hook fires for a module,
/// it looks up which earlier node actually produced the exact tensor object being passed in. This
/// correctly diagrams models where a single tensor fans out to several sibling layers (a shared trunk
/// feeding two branches, for example) instead of assuming everything runs in one straight chain.
/// When a tensor arrives that no tracked layer produced directly — typically because it passed through
/// an untracked raw tensor operation between modules, such as a residual <c>+</c> — the extractor falls
/// back to connecting from whichever layers are still "dangling" (produced a tensor that nothing has
/// consumed yet), which recovers the common fan-in/merge case. A skip connection that reuses a tensor
/// long after it was already consumed elsewhere cannot be reconstructed this way, since that requires
/// tracing the model's actual computation graph rather than its module calls; it is the one topology
/// this extractor cannot see.
/// </para>
/// </remarks>
public static class TorchSharpExtractor
{
    private const string InputNodeId = "input";
    private const string OutputNodeId = "output";

    private static readonly HashSet<string> AttributeBlockList = new(StringComparer.OrdinalIgnoreCase)
    {
        "training",
    };

    /// <summary>
    /// The subset of a module's properties that are actually useful to show on a diagram, in display
    /// order. Keeps common layers (Linear, Conv*, BatchNorm*, Embedding, Dropout, RNN/LSTM/GRU) readable
    /// instead of dumping every public property (transposed flags, output padding, etc.) onto the node.
    /// </summary>
    private static readonly string[] PreferredAttributeOrder =
    {
        "in_features", "out_features",
        "in_channels", "out_channels",
        "kernel_size", "stride", "padding", "dilation", "groups",
        "num_features", "eps", "momentum",
        "num_embeddings", "embedding_dim",
        "input_size", "hidden_size", "num_layers",
        "p",
    };

    /// <summary>
    /// Extracts a <see cref="Graph"/> from <paramref name="model"/> by running a single dummy
    /// forward pass with a zero-filled tensor of shape <paramref name="inputShape"/>.
    /// </summary>
    /// <param name="model">The model to inspect. Must accept and return a single <see cref="Tensor"/>.</param>
    /// <param name="inputShape">The shape of the dummy input tensor, e.g. <c>[1, 3, 224, 224]</c>.</param>
    /// <param name="modelName">An optional display name for the graph; defaults to the model's type name.</param>
    /// <exception cref="NotSupportedException">
    /// Thrown when <paramref name="model"/> (or one if its leaf modules) does not implement the
    /// single-tensor-in/single-tensor-out <c>Module&lt;Tensor, Tensor&gt;</c> shape that TorchSharpVisual
    /// currently supports.
    /// </exception>
    public static Graph Extract(nn.Module model, long[] inputShape, string? modelName = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(inputShape);
        if (inputShape.Length == 0)
        {
            throw new ArgumentException("Input shape must have at least one dimension.", nameof(inputShape));
        }

        if (model is not nn.Module<Tensor, Tensor> callableModel)
        {
            throw new NotSupportedException(
                $"TorchSharpVisual currently supports models whose forward pass takes a single Tensor " +
                $"and returns a single Tensor. '{model.GetType().Name}' does not match that shape.");
        }

        var graph = new Graph { Name = modelName ?? model.GetType().Name };
        var tracker = new DataFlowTracker();

        var inputNode = new Node
        {
            Id = InputNodeId,
            Name = "input",
            TypeName = "Tensor",
            Kind = NodeKind.Input,
            OutputShape = inputShape,
        };
        graph.AddNode(inputNode);

        var unhook = HookLeafModules(model, graph, tracker);

        var wasTraining = model.training;
        Node[] outputParents;
        try
        {
            model.eval();
            using (no_grad())
            {
                using var dummyInput = zeros(inputShape);
                tracker.RegisterProducer(dummyInput, inputNode);

                using var modelOutput = callableModel.call(dummyInput);
                outputParents = tracker.ResolveParents(modelOutput);
            }
        }
        finally
        {
            unhook();
            if (wasTraining)
            {
                model.train();
            }
        }

        var outputNode = new Node
        {
            Id = OutputNodeId,
            Name = "output",
            TypeName = "Tensor",
            Kind = NodeKind.Output,
            InputShape = outputParents.FirstOrDefault()?.OutputShape,
        };
        graph.AddNode(outputNode);
        foreach (var parent in outputParents)
        {
            graph.AddEdge(new Edge
            {
                FromId = parent.Id,
                ToId = outputNode.Id,
                Label = ShapeFormatter.Format(parent.OutputShape),
            });
        }

        return graph;
    }

    private static Action HookLeafModules(nn.Module model, Graph graph, DataFlowTracker tracker)
    {
        var removers = new List<Action>();
        var order = 0;

        // TorchSharp's named_modules()/modules() do not include the module itself, only descendants,
        // so a bare leaf module passed in directly (e.g. a single Linear) is handled by treating the
        // root as its own single leaf when it has no children.
        var candidates = model.children().Any()
            ? model.named_modules()
            : new[] { (name: string.Empty, module: model) };

        foreach (var (dottedName, submodule) in candidates)
        {
            if (submodule.children().Any())
            {
                // Container module (e.g. Sequential, or a custom composite block) — recurse via its
                // own children rather than treating it as a node.
                continue;
            }

            if (submodule is not nn.Module<Tensor, Tensor> hookable)
            {
                throw new NotSupportedException(
                    $"TorchSharpVisual currently supports leaf modules whose forward pass takes a single " +
                    $"Tensor and returns a single Tensor. '{submodule.GetType().Name}' " +
                    $"(at '{dottedName}') does not match that shape.");
            }

            var name = dottedName;
            var remover = hookable.register_forward_hook((m, input, output) =>
            {
                var typeName = m.GetType().Name;
                var parameterCount = m.parameters().Sum(p => p.numel());
                var node = new Node
                {
                    Id = $"n{order}_{Sanitize(name, typeName)}",
                    Name = string.IsNullOrEmpty(name) ? typeName : name,
                    TypeName = typeName,
                    Kind = parameterCount > 0 ? NodeKind.Module : NodeKind.Operation,
                    InputShape = input.shape,
                    OutputShape = output.shape,
                    ParentPath = GetParentPath(name),
                    ParameterCount = parameterCount,
                    Attributes = ExtractAttributes(m),
                };
                graph.AddNode(node);

                foreach (var parent in tracker.ResolveParents(input))
                {
                    graph.AddEdge(new Edge
                    {
                        FromId = parent.Id,
                        ToId = node.Id,
                        Label = ShapeFormatter.Format(input.shape),
                    });
                }

                tracker.RegisterProducer(output, node);
                order++;
                return output;
            });
            removers.Add(remover.remove);
        }

        return () =>
        {
            foreach (var remove in removers)
            {
                remove();
            }
        };
    }

    private static string GetParentPath(string dottedName)
    {
        var separatorIndex = dottedName.LastIndexOf('.');
        return separatorIndex < 0 ? string.Empty : dottedName[..separatorIndex];
    }

    private static string Sanitize(string dottedName, string fallback)
    {
        var basis = string.IsNullOrEmpty(dottedName) ? fallback : dottedName;
        var chars = basis.Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray();
        return new string(chars);
    }

    private static IReadOnlyDictionary<string, string> ExtractAttributes(nn.Module module)
    {
        var attributes = new Dictionary<string, string>();
        foreach (var property in module.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (AttributeBlockList.Contains(property.Name) || !property.CanRead || property.GetIndexParameters().Length > 0)
            {
                continue;
            }

            // Tensors and sub-modules (weights, biases, nested layers) aren't useful as flat text.
            if (typeof(Tensor).IsAssignableFrom(property.PropertyType) ||
                typeof(nn.Module).IsAssignableFrom(property.PropertyType))
            {
                continue;
            }

            object? value;
            try
            {
                value = property.GetValue(module);
            }
            catch
            {
                continue;
            }

            if (value is null)
            {
                continue;
            }

            var formatted = FormatAttributeValue(value);
            if (!string.IsNullOrEmpty(formatted))
            {
                attributes[property.Name] = formatted;
            }
        }

        return CurateAttributes(attributes);
    }

    /// <summary>
    /// Narrows a module's full attribute set down to the handful that are actually informative on a
    /// diagram (see <see cref="PreferredAttributeOrder"/>). Falls back to the unfiltered set for
    /// unrecognized module types, so custom user layers still surface something useful.
    /// </summary>
    private static IReadOnlyDictionary<string, string> CurateAttributes(Dictionary<string, string> attributes)
    {
        var curated = new Dictionary<string, string>();
        foreach (var name in PreferredAttributeOrder)
        {
            if (attributes.TryGetValue(name, out var value))
            {
                curated[name] = value;
            }
        }

        return curated.Count > 0 ? curated : attributes;
    }

    private static string FormatAttributeValue(object value) => value switch
    {
        long[] longs => "[" + string.Join(", ", longs) + "]",
        int[] ints => "[" + string.Join(", ", ints) + "]",
        Array array => "[" + string.Join(", ", array.Cast<object>()) + "]",
        _ => value.ToString() ?? string.Empty,
    };

    /// <summary>
    /// Tracks which node produced each tensor object flowing through the model, by reference identity
    /// (a <see cref="Tensor"/>'s own <c>Equals</c>/<c>GetHashCode</c> perform elementwise comparison,
    /// which is not what "is this the same tensor" means here). Used to resolve the real parent(s) of
    /// each node as the forward pass executes.
    /// </summary>
    private sealed class DataFlowTracker
    {
        private readonly Dictionary<Tensor, Node> producedBy = new(ReferenceEqualityComparer.Instance);
        private readonly HashSet<Node> dangling = new();

        public void RegisterProducer(Tensor tensor, Node node)
        {
            producedBy[tensor] = node;
            dangling.Add(node);
        }

        /// <summary>
        /// Finds the node(s) that produced <paramref name="tensor"/>. If it was produced directly by a
        /// tracked layer, that single layer is the parent. Otherwise (the tensor passed through an
        /// untracked operation between modules) every node whose output hasn't yet been claimed by a
        /// direct consumer is used as an approximation of a merge/fan-in point.
        /// </summary>
        public Node[] ResolveParents(Tensor tensor)
        {
            if (producedBy.TryGetValue(tensor, out var direct))
            {
                dangling.Remove(direct);
                return new[] { direct };
            }

            var fallback = dangling.ToArray();
            dangling.ExceptWith(fallback);
            return fallback;
        }

        private sealed class ReferenceEqualityComparer : IEqualityComparer<Tensor>
        {
            public static readonly ReferenceEqualityComparer Instance = new();

            public bool Equals(Tensor? x, Tensor? y) => ReferenceEquals(x, y);

            public int GetHashCode(Tensor obj) => RuntimeHelpers.GetHashCode(obj);
        }
    }
}
