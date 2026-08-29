# Lunar Image Registration RAG Pipeline

A **Retrieval-Augmented Generation (RAG)** augmented algorithmic pipeline for registering multi-temporal, multi-sensor lunar imagery (OHRC, TMC-2, IIRS) across **20×–300× scale gaps** and differing Sun angles.

## Problem Statement

| Input | Process | Output |
|-------|---------|--------|
| OHRC strip (0.25 m), TMC-2 ortho (5 m), IIRS spectral slice (80 m) | Illumination-invariant features → cross-scale matching → robust geometric filtering | Sub-pixel tie points + co-registered pixel-aligned layers |

## Quick Start

```bash
# Install
pip install -e ".[dev,rag,deep]"

# Run synthetic demo
python -m src.main demo

# Register real image pair
python -m src.main register \
  --ref data/sample/ohrc.tif \
  --mov data/sample/tmc2.tif \
  --ref-sensor OHRC --mov-sensor TMC2 \
  --sun-ref 30 --sun-mov 55

# Query RAG knowledge base
python -m src.main rag-query --query "OHRC to IIRS 300x scale registration"
```

## Pipeline Stages

```
┌─────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────┐   ┌─────────────────┐   ┌──────────────┐
│ RAG Params  │ → │ Preprocess   │ → │ Feature Extract  │ → │ Matching │ → │ Geo. Filtering  │ → │ Registration │
│ (retrieval) │   │ (illumination)│   │ PC/Contour/Deep  │   │ semi-dense│   │ MAGSAC/Graph/SC │   │ warp + MPP   │
└─────────────┘   └──────────────┘   └──────────────────┘   └──────────┘   └─────────────────┘   └──────────────┘
```

## Measurable Part of the Project (MPP)

| Metric | Target | Description |
|--------|--------|-------------|
| `registration_rmse_px` | < 0.5 px | Sub-pixel reprojection error |
| `inlier_ratio` | ≥ 60% | Fraction of geometric inliers |
| `tie_point_uniformity_entropy` | ≥ 3.5 bits | Spatial distribution quality |
| `grid_occupancy_pct` | ≥ 70% | Coverage across 16×16 grid |
| `rag_retrieval_relevance` | ≥ 0.7 | RAG parameter selection quality |
| `stage_latency` | < 30 s/tile | Per-stage timing |

Reports saved to `outputs/mpp_report.json`.

## Design Patterns

See [Architecture.md](Architecture.md) and [Explanation.md](Explanation.md) for pattern-level code walkthroughs.

| Pattern | Usage |
|---------|-------|
| **Pipeline** | Sequential registration stages |
| **Strategy** | Swappable features, matchers, estimators |
| **Factory** | Component creation by sensor/config |
| **Observer** | Stage timing and event hooks |
| **Chain of Responsibility** | RAG context enrichment |
| **Template Method** | Base stage validate → process → annotate |

## Project Structure

See [Codebase.md](Codebase.md).

## Documentation

- [Architecture.md](Architecture.md) — system design and data flow
- [Explanation.md](Explanation.md) — algorithms, RAG role, design patterns with code
- [Codebase.md](Codebase.md) — file structure reference

## License

MIT
