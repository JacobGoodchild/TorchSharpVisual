using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using TorchSharpVisual.Core;

namespace TorchSharpVisual.Renderers;

/// <summary>
/// Renders a <see cref="Graph"/> as Graphviz DOT syntax, and optionally shells out to the local
/// <c>dot</c> executable to produce a PNG or SVG image.
/// </summary>
public static class DotRenderer
{
    private const string InputOutputColor = "#FFF2CC"; // light yellow / amber
    private const string ModuleColor = "#D5E8D4"; // soft green
    private const string OperationColor = "#DAE8FC"; // soft blue

    /// <summary>Builds the Graphviz DOT source for <paramref name="graph"/> as a string.</summary>
    public static string ToDot(Graph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);

        var sb = new StringBuilder();
        sb.Append("digraph \"").Append(Escape(graph.Name)).Append("\" {\n");
        sb.Append("  rankdir=TB;\n");
        sb.Append("  fontname=\"Helvetica,Arial,sans-serif\";\n");
        sb.Append("  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica,Arial,sans-serif\", fontsize=11];\n");
        sb.Append("  edge [fontname=\"Helvetica,Arial,sans-serif\", fontsize=9, color=\"#666666\"];\n\n");

        var (clusteredParentPaths, nodesByParent) = GroupByParent(graph.Nodes);

        var clusterIndex = 0;
        foreach (var parentPath in clusteredParentPaths)
        {
            sb.Append("  subgraph \"cluster_").Append(clusterIndex).Append("\" {\n");
            sb.Append("    label=\"").Append(Escape(parentPath)).Append("\";\n");
            sb.Append("    style=\"rounded\";\n");
            sb.Append("    color=\"#999999\";\n");
            sb.Append("    fontsize=10;\n");
            foreach (var node in nodesByParent[parentPath])
            {
                sb.Append("    ").Append(RenderNode(node)).Append('\n');
            }

            sb.Append("  }\n");
            clusterIndex++;
        }

        foreach (var node in graph.Nodes.Where(n => string.IsNullOrEmpty(n.ParentPath)))
        {
            sb.Append("  ").Append(RenderNode(node)).Append('\n');
        }

        sb.Append('\n');
        foreach (var edge in graph.Edges)
        {
            sb.Append("  \"").Append(edge.FromId).Append("\" -> \"").Append(edge.ToId).Append('"');
            if (!string.IsNullOrEmpty(edge.Label))
            {
                sb.Append(" [label=\"").Append(Escape(edge.Label)).Append("\"]");
            }

            sb.Append(";\n");
        }

        sb.Append('}').Append('\n');
        return sb.ToString();
    }

    /// <summary>
    /// Renders <paramref name="graph"/> to <paramref name="outputPath"/>. The output format is
    /// inferred from the file extension: <c>.dot</c>/<c>.gv</c> writes raw Graphviz source directly,
    /// while <c>.png</c>/<c>.svg</c> shell out to the local Graphviz <c>dot</c> executable.
    /// </summary>
    /// <param name="graph">The graph to render.</param>
    /// <param name="outputPath">The destination file path.</param>
    /// <param name="dotExecutable">
    /// The name or path of the Graphviz executable to invoke for image formats. Defaults to <c>"dot"</c>,
    /// which must be resolvable on <c>PATH</c>.
    /// </param>
    /// <exception cref="NotSupportedException">The file extension is not one of <c>.dot</c>, <c>.gv</c>, <c>.png</c>, or <c>.svg</c>.</exception>
    /// <exception cref="InvalidOperationException">The Graphviz executable could not be started or exited with an error.</exception>
    public static void Render(Graph graph, string outputPath, string dotExecutable = "dot")
    {
        ArgumentNullException.ThrowIfNull(graph);
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("Output path must not be empty.", nameof(outputPath));
        }

        var dotSource = ToDot(graph);
        var extension = Path.GetExtension(outputPath).TrimStart('.').ToLowerInvariant();

        switch (extension)
        {
            case "dot":
            case "gv":
                File.WriteAllText(outputPath, dotSource);
                return;
            case "png":
            case "svg":
                RunDot(dotSource, outputPath, extension, dotExecutable);
                return;
            default:
                throw new NotSupportedException(
                    $"Unsupported output extension '.{extension}'. Use .png, .svg, .dot, or .gv.");
        }
    }

    private static (List<string> order, Dictionary<string, List<Node>> byParent) GroupByParent(IEnumerable<Node> nodes)
    {
        var order = new List<string>();
        var byParent = new Dictionary<string, List<Node>>();
        foreach (var node in nodes)
        {
            if (string.IsNullOrEmpty(node.ParentPath))
            {
                continue;
            }

            if (!byParent.TryGetValue(node.ParentPath, out var list))
            {
                list = new List<Node>();
                byParent[node.ParentPath] = list;
                order.Add(node.ParentPath);
            }

            list.Add(node);
        }

        return (order, byParent);
    }

    private static string RenderNode(Node node)
    {
        var color = node.Kind switch
        {
            NodeKind.Input or NodeKind.Output => InputOutputColor,
            NodeKind.Module => ModuleColor,
            NodeKind.Operation => OperationColor,
            _ => "#FFFFFF",
        };

        var label = BuildLabel(node);
        return $"\"{node.Id}\" [label=\"{label}\", fillcolor=\"{color}\"];";
    }

    private static string BuildLabel(Node node)
    {
        var lines = new List<string>();
        if (node.Kind == NodeKind.Input)
        {
            lines.Add("Input");
            lines.Add(ShapeFormatter.Format(node.OutputShape));
        }
        else if (node.Kind == NodeKind.Output)
        {
            lines.Add("Output");
            lines.Add(ShapeFormatter.Format(node.InputShape));
        }
        else
        {
            lines.Add(node.Name);
            lines.Add(node.TypeName);
            if (node.Attributes.Count > 0)
            {
                lines.Add(string.Join(", ", node.Attributes.Select(kv => $"{kv.Key}={kv.Value}")));
            }

            lines.Add($"in: {ShapeFormatter.Format(node.InputShape)}");
            lines.Add($"out: {ShapeFormatter.Format(node.OutputShape)}");
        }

        // Escape each line individually, then join with a literal DOT newline token so the
        // escaping of backslashes doesn't corrupt the line-break markers themselves.
        return string.Join("\\n", lines.Select(Escape));
    }

    private static string Escape(string value) => value.Replace("\\", "\\\\").Replace("\"", "\\\"");

    private static void RunDot(string dotSource, string outputPath, string format, string dotExecutable)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = dotExecutable,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        startInfo.ArgumentList.Add($"-T{format}");
        startInfo.ArgumentList.Add("-o");
        startInfo.ArgumentList.Add(outputPath);

        Process process;
        try
        {
            process = Process.Start(startInfo)
                       ?? throw new InvalidOperationException("Failed to start the Graphviz 'dot' process.");
        }
        catch (Win32Exception ex)
        {
            throw new InvalidOperationException(
                $"Could not run the Graphviz '{dotExecutable}' executable. Make sure Graphviz is installed " +
                "and available on PATH (e.g. 'apt-get install graphviz', 'brew install graphviz', or 'choco install graphviz').",
                ex);
        }

        using (process)
        {
            process.StandardInput.Write(dotSource);
            process.StandardInput.Close();

            var stderr = process.StandardError.ReadToEnd();
            process.WaitForExit();

            if (process.ExitCode != 0)
            {
                throw new InvalidOperationException($"Graphviz 'dot' exited with code {process.ExitCode}: {stderr}");
            }
        }
    }
}
