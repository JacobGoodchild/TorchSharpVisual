# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Nothing yet.

## [0.2.0]

### Added
- Real dataflow-based edge construction in `TorchSharpExtractor`: edges are now derived from actual
  tensor identity (which layer produced the tensor another layer consumes) instead of assuming every
  model executes as one straight chain. This correctly diagrams models with a shared trunk feeding
  multiple branches, and falls back to a fan-in heuristic for untracked operations (e.g. a residual
  `+`) between modules.
- `Node.ParameterCount` and `Graph.TotalParameterCount` / `Graph.LayerCount` — every layer now reports
  its learnable parameter count, and diagrams show a summary header with the model's total layer and
  parameter counts.
- `RenderOptions` (layout direction, legend visibility, parameter-count visibility, summary header
  visibility, DPI, font) passed to `DotRenderer.ToDot`/`Render` and `ModuleExtensions.DrawGraph`/`ToDotGraph`.
- A color legend drawn on every diagram by default, explaining the amber/green/blue color scheme.
- `samples/TorchSharpVisual.Sample` — a runnable console app demonstrating a Sequential conv net, a
  custom multi-layer classifier, and a branching (fork/join) model.
- GitHub Actions CI workflow building and testing on every push/PR.
- `CONTRIBUTING.md`.

### Changed
- `DotRenderer.ToDot`/`Render` gained `RenderOptions` overloads; the previous signatures remain
  available and use `RenderOptions.Default`.

## [0.1.0] - Initial release

### Added
- `TorchSharpVisual.Core`: `Graph`, `Node`, `Edge`, `NodeKind`, `ShapeFormatter`.
- `TorchSharpVisual.Extractors.TorchSharpExtractor`: recursively inspects a `TorchSharp.torch.nn.Module`,
  runs a dummy forward pass, and records tensor shapes at each layer via forward hooks.
- `TorchSharpVisual.Renderers.DotRenderer`: renders a `Graph` as Graphviz DOT syntax, with PNG/SVG
  output via the local `dot` executable.
- `TorchSharpVisual.Extensions.ModuleExtensions`: `model.DrawGraph(...)` / `model.ToDotGraph(...)`.
- Initial test suite covering `Sequential`, custom multi-layer, and convolutional models.
