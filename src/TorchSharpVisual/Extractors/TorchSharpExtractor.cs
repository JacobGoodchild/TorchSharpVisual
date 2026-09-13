using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using TorchSharp;
using TorchSharpVisual.Core;
using static TorchSharp.torch;

namespace TorchSharpVisual.Extractors;

/// <summary>
/// Inspects a <see cref="TorchSharp.torch.nn.Module"/> and turns it into a renderer-agnostic
/// <see cref="Graph"/> describing its layers and the tensor shapes flowing between them.
/// </summary>
/// <remarks>
/// The extractor works by recursively walking the module tree to find "leaf" modules (those with
/// no children of their own), attaching a forward hook to each one, then running a single dummy
/// forward pass with a zero tensor of the requested input shape. The hooks record, in execution
/// order, the input/output shape of every leaf module — which is then linearised into a chain of
/// nodes. This mirrors the approach used by tools like torchview, and works well for the common
/// case of models built from <c>Sequential</c> containers and straight-line custom forward passes.
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

        var executionLog = new List<ExecutionRecord>();
        var unhook = HookLeafModules(model, executionLog);

        var wasTraining = model.training;
        try
        {
            model.eval();
            using (no_grad())
            using (var dummyInput = zeros(inputShape))
            {
                using var _ = callableModel.call(dummyInput);
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

        return BuildGraph(modelName ?? model.GetType().Name, inputShape, executionLog);
    }

    private static Action HookLeafModules(nn.Module model, List<ExecutionRecord> executionLog)
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
                executionLog.Add(new ExecutionRecord(
                    DottedName: name,
                    Module: m,
                    InputShape: input.shape,
                    OutputShape: output.shape,
                    Order: order++));
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

    private static Graph BuildGraph(string modelName, long[] inputShape, List<ExecutionRecord> executionLog)
    {
        var graph = new Graph { Name = modelName };

        var inputNode = new Node
        {
            Id = InputNodeId,
            Name = "input",
            TypeName = "Tensor",
            Kind = NodeKind.Input,
            OutputShape = inputShape,
        };
        graph.AddNode(inputNode);

        var previous = inputNode;
        foreach (var record in executionLog.OrderBy(r => r.Order))
        {
            var typeName = record.Module.GetType().Name;
            var hasParameters = record.Module.parameters().Any();
            var node = new Node
            {
                Id = $"n{record.Order}_{Sanitize(record.DottedName, typeName)}",
                Name = string.IsNullOrEmpty(record.DottedName) ? typeName : record.DottedName,
                TypeName = typeName,
                Kind = hasParameters ? NodeKind.Module : NodeKind.Operation,
                InputShape = record.InputShape,
                OutputShape = record.OutputShape,
                ParentPath = GetParentPath(record.DottedName),
                Attributes = ExtractAttributes(record.Module),
            };
            graph.AddNode(node);
            graph.AddEdge(new Edge
            {
                FromId = previous.Id,
                ToId = node.Id,
                Label = ShapeFormatter.Format(record.InputShape),
            });
            previous = node;
        }

        var outputNode = new Node
        {
            Id = OutputNodeId,
            Name = "output",
            TypeName = "Tensor",
            Kind = NodeKind.Output,
            InputShape = previous.OutputShape,
        };
        graph.AddNode(outputNode);
        graph.AddEdge(new Edge
        {
            FromId = previous.Id,
            ToId = outputNode.Id,
            Label = ShapeFormatter.Format(previous.OutputShape),
        });

        return graph;
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

    private sealed record ExecutionRecord(string DottedName, nn.Module Module, long[] InputShape, long[] OutputShape, int Order);
}
