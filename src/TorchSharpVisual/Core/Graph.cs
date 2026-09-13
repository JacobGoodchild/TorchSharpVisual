using System.Collections.Generic;

namespace TorchSharpVisual.Core;

/// <summary>
/// A plain, renderer-agnostic representation of a model's architecture: a list of nodes
/// (tensors and layers) connected by edges (tensor flow).
/// </summary>
public sealed class Graph
{
    /// <summary>A human-readable name for the graph, typically the model's type name.</summary>
    public string Name { get; set; } = "Model";

    /// <summary>The nodes in the graph, in the order they were discovered/executed.</summary>
    public List<Node> Nodes { get; } = new();

    /// <summary>The edges connecting the nodes, in the order tensors flowed through the model.</summary>
    public List<Edge> Edges { get; } = new();

    /// <summary>Appends a node to the graph.</summary>
    public void AddNode(Node node) => Nodes.Add(node);

    /// <summary>Appends an edge to the graph.</summary>
    public void AddEdge(Edge edge) => Edges.Add(edge);
}
