# TorchSharpVisual

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![NuGet](https://img.shields.io/badge/nuget-TorchSharpVisual-blue)](https://www.nuget.org/packages/TorchSharpVisual)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](#license)
[![.NET 8.0](https://img.shields.io/badge/.NET-8.0-512BD4)](#)

**TorchSharpVisual** draws clean, color-coded architecture diagrams for [TorchSharp](https://github.com/dotnet/TorchSharp) models — the same idea as Python's [`torchview`](https://github.com/mert-kurttutan/torchview) and [`visualkeras`](https://github.com/paulgavrikov/visualkeras), but for .NET.

Give it a model and an input shape, and it hands back a PNG, SVG, or raw Graphviz `.dot` file showing every layer, how data flows between them, and the tensor shape at each step.

<p align="center">
  <img src="docs/example-architecture.png" alt="Example TorchSharpVisual architecture diagram" width="420">
</p>

## Features

- 🔍 **Automatic inspection** — recursively walks a model's sub-modules, no manual annotation needed.
- 📐 **Real shapes, not guesses** — runs one dummy forward pass and records the actual tensor shape flowing into and out of every layer.
- 🎨 **Color-coded by role** so a diagram reads at a glance:
  - 🟡 **Amber** — the input/output tensors
  - 🟢 **Green** — layers with learnable parameters (`Linear`, `Conv2d`, `BatchNorm2d`, …)
  - 🔵 **Blue** — stateless operations (`ReLU`, `Softmax`, `MaxPool2d`, `Flatten`, …)
- 📦 **Groups nested containers** — a `Sequential` block nested inside a bigger model is drawn as a labeled cluster, so structure stays visible.
- 🖼️ **PNG, SVG, or raw DOT** — render an image via your local Graphviz install, or grab the `.dot` source directly with zero external dependencies.
- 🧩 **One-line API** — `model.DrawGraph(...)` and you're done.

## Installation

```bash
dotnet add package TorchSharpVisual
```

You'll also need **Graphviz** installed if you want PNG/SVG output (raw `.dot` output has no external dependency):

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y graphviz

# macOS
brew install graphviz

# Windows
choco install graphviz
```

Verify it's on `PATH`:

```bash
dot -V
```

## Quick start

```csharp
using TorchSharp;
using TorchSharpVisual.Extensions;
using static TorchSharp.torch;

// Any TorchSharp nn.Module<Tensor, Tensor> — Sequential or a custom class.
var model = nn.Sequential(
    ("conv1", nn.Conv2d(3, 16, kernel_size: 3, padding: 1)),
    ("relu1", nn.ReLU()),
    ("pool1", nn.MaxPool2d(kernel_size: 2)),
    ("flatten", nn.Flatten()),
    ("fc", nn.Linear(16 * 16 * 16, 10)));

model.DrawGraph(inputShape: new long[] { 1, 3, 32, 32 }, fileName: "architecture.png");
```

That's it — `architecture.png` now shows the full model, labeled with layer names, types, and tensor shapes at every step.

Other useful calls:

```csharp
// SVG instead of PNG — same call, different extension.
model.DrawGraph(inputShape: new long[] { 1, 3, 32, 32 }, fileName: "architecture.svg");

// Raw Graphviz DOT source, no Graphviz install required.
model.DrawGraph(inputShape: new long[] { 1, 3, 32, 32 }, fileName: "architecture.dot");
string dot = model.ToDotGraph(inputShape: new long[] { 1, 3, 32, 32 });

// Give the diagram a friendlier title than the C# class name.
model.DrawGraph(inputShape: new long[] { 1, 3, 32, 32 }, fileName: "architecture.png", modelName: "MyConvNet");
```

## How it works

1. **`TorchSharpVisual.Extractors.TorchSharpExtractor`** recursively walks the model's sub-modules (`named_modules()`), finds the "leaf" layers (the ones that actually do work), and attaches a forward hook to each. It then runs a single dummy forward pass with a zero-filled tensor of the shape you gave it. Every hook fires in the exact order the model executes its layers, recording the input/output tensor shape at that point.
2. That execution trace is turned into a plain **`TorchSharpVisual.Core.Graph`** — a list of `Node`s (tensors and layers) and `Edge`s (the tensor flow between them), with no Graphviz or TorchSharp types leaking through.
3. **`TorchSharpVisual.Renderers.DotRenderer`** turns the graph into Graphviz DOT syntax, colors each node by its `NodeKind`, groups nested containers into clusters, and — for `.png`/`.svg` — shells out to your local `dot` executable to render the image.
4. **`TorchSharpVisual.Extensions.ModuleExtensions`** wraps steps 1–3 into the one-line `model.DrawGraph(...)` / `model.ToDotGraph(...)` calls shown above.

**A note on scope:** because the diagram is built from real execution order rather than parsing your `forward()` method, it currently supports models whose forward pass takes a single `Tensor` and returns a single `Tensor` (which covers the overwhelming majority of `Sequential` stacks and custom feed-forward/convolutional networks). Branching architectures with skip connections will still extract and diagram correctly as long as every individual layer is single-tensor-in/single-tensor-out — the layers are simply drawn in the order they executed.

## Project layout

```
src/TorchSharpVisual/
  Core/          Graph, Node, Edge — the plain data model for a diagram
  Extractors/    TorchSharpExtractor — inspects a model and builds a Graph
  Renderers/     DotRenderer — turns a Graph into Graphviz DOT / PNG / SVG
  Extensions/    ModuleExtensions — the model.DrawGraph(...) convenience API
tests/TorchSharpVisual.Tests/
  Unit tests covering extraction (Sequential, custom multi-layer, Conv2d, nested containers)
  and rendering (DOT syntax, color scheme, PNG/SVG generation via Graphviz).
```

## Running the tests

```bash
dotnet test
```

The rendering tests shell out to a real `dot` executable, so Graphviz needs to be installed to run the full suite (see [Installation](#installation)).

## Contributing

Issues and pull requests are welcome — this is a young project and there's plenty of room to grow (richer attribute extraction for more layer types, true multi-input/multi-output tracing, additional layout options).

## License

[MIT](LICENSE)
