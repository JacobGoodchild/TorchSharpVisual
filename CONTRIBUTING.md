# Contributing to TorchSharpVisual

Thanks for considering a contribution! This project is young, so there's plenty of room to help,
from small fixes to new features.

## Getting set up

You'll need:

- [.NET 8 SDK](https://dotnet.microsoft.com/download)
- [Graphviz](https://graphviz.org/download/) (`dot` on `PATH`) — required to run the full test suite, since some tests render real PNG/SVG output.

```bash
git clone https://github.com/JacobGoodchild/TorchSharpVisual.git
cd TorchSharpVisual
dotnet build
dotnet test
```

Try the sample app to see the library in action end to end:

```bash
dotnet run --project samples/TorchSharpVisual.Sample
```

## Project layout

```
src/TorchSharpVisual/
  Core/          Graph, Node, Edge — the plain data model for a diagram
  Extractors/    TorchSharpExtractor — inspects a model and builds a Graph
  Renderers/     DotRenderer, RenderOptions — turns a Graph into Graphviz DOT / PNG / SVG
  Extensions/    ModuleExtensions — the model.DrawGraph(...) convenience API
tests/TorchSharpVisual.Tests/
  Unit tests for extraction and rendering.
samples/TorchSharpVisual.Sample/
  A runnable console app demonstrating real usage on a few different model shapes.
```

## Before opening a pull request

1. `dotnet build` — the CI build treats warnings as errors, so please fix any before submitting.
2. `dotnet test` — all tests should pass. Add tests for new behavior; a bug fix should come with a
   regression test that fails before your fix and passes after.
3. If you change the diagram output (colors, layout, label format), regenerate the sample images and
   check them into `docs/` so the README stays accurate.
4. Keep the public API's XML doc comments up to date — they ship in the NuGet package.

## Reporting bugs

Please include:
- The model definition (or a minimal reproduction) that triggers the issue.
- The input shape you passed to `DrawGraph`/`Extract`.
- The exception message and stack trace, if any.

## Ideas for contributions

- Support for models with multiple inputs/outputs (`Module<Tensor, Tensor, Tensor>` and friends).
- Richer attribute extraction for more layer types (recurrent layers, transformer blocks, embeddings).
- Additional layout/theme options (dark mode palette, compact mode that hides shapes).
- Folding repeated blocks (e.g. a ResNet with 50 identical residual blocks) instead of drawing each one.

If you're not sure whether an idea fits, open an issue first to discuss it.
