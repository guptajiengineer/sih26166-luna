# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Lunar Registration RAG System                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────────┐  │
│  │  CLI / API  │───▶│              LunarRegistrationPipeline              │  │
│  │  main.py    │    │                   (Orchestrator)                    │  │
│  └─────────────┘    └────────────────────────┬────────────────────────────┘  │
│                                              │                               │
│                    ┌─────────────────────────▼─────────────────────────┐   │
│                    │              Pipeline (Design Pattern)              │   │
│                    │  Stage₁ → Stage₂ → Stage₃ → Stage₄ → Stage₅ → Stage₆│   │
│                    └─────────────────────────┬─────────────────────────┘   │
│                                              │                               │
│     ┌────────────┬────────────┬─────────────┼─────────────┬──────────────┐ │
│     ▼            ▼            ▼             ▼             ▼              ▼   │ │
│  ┌──────┐   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌───────┐ │ │
│  │ RAG  │   │Preproc  │  │Features │  │Matching │  │ Geometry │  │Register│ │ │
│  │Layer │   │ Stage   │  │ Stage   │  │ Stage   │  │  Stage   │  │ Stage │ │ │
│  └──┬───┘   └─────────┘  └────┬────┘  └────┬────┘  └────┬─────┘  └───┬───┘ │ │
│     │                         │            │            │            │     │ │
│     ▼                         ▼            ▼            ▼            ▼     │ │
│  ChromaDB              Illumination   Strategy      Strategy     MPP       │ │
│  + Embeddings          Normalize      Factory       Factory      Metrics   │ │
│  Knowledge Base        (sun angle)    (PC/Contour/  (MAGSAC/      Report    │ │
│                                        Deep)        Graph/SC)               │ │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
LunarImage (ref) ──┐
                   ├──▶ PipelineContext ──▶ [Stages] ──▶ RegistrationResult
LunarImage (mov) ──┘                                      + MPPReport
```

### PipelineContext (shared state)

| Field | Stage Populated | Description |
|-------|-----------------|-------------|
| `reference`, `moving` | Input | Multi-sensor lunar images |
| `rag_params` | RAG | Retrieved parameters |
| `ref_keypoints`, `mov_keypoints` | Features | Illumination-invariant keypoints |
| `tie_points` | Matching | Raw correspondences |
| `inlier_tie_points`, `transform_matrix` | Geometry | Filtered matches + transform |
| `registered_image`, `result` | Registration | Warped output |
| `stage_timings` | All | MPP latency metrics |

## RAG Layer Architecture

```
User Query (sensor pair, sun angles)
        │
        ▼
┌───────────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│ SensorContext     │ ──▶ │ SunAngleContext    │ ──▶ │ RAGRetrievalHandler │
│ Handler           │     │ Handler            │     │ (ChromaDB + ST)     │
└───────────────────┘     └────────────────────┘     └──────────┬──────────┘
                                                                  │
                                                                  ▼
                                                    suggest_parameters()
                                                    → feature_extractors
                                                    → matcher
                                                    → geometric_estimator
                                                    → thresholds
```

The RAG layer does **not** perform registration directly. It retrieves domain knowledge (sensor pair strategies, sun-angle handling, scale-gap advice) and **configures** the algorithmic pipeline adaptively.

## Component Layer

### Feature Extraction (Strategy Pattern)

```
FeatureExtractor (ABC)
├── PhaseCongruencyExtractor   — log-Gabor / multi-scale gradient PC proxy
├── ContourFeatureExtractor    — adaptive threshold contours
└── DeepEmbeddingExtractor     — MobileNet embeddings / ORB fallback
```

### Matching (Strategy Pattern)

```
Matcher (ABC)
├── SemiDenseMatcher  — descriptor NN + scale rescaling + sub-pixel refine
└── DenseMatcher        — Farneback optical flow grid sampling
```

### Geometric Filtering (Strategy Pattern)

```
GeometricEstimator (ABC)
├── MAGSACEstimator          — USAC_MAGSAC homography/affine
├── GraphMatchingFilter      — clique consistency graph
└── SpatialConsistencyFilter — local neighborhood deviation
```

## Scale Gap Handling

| Pair | Scale Ratio | RAG-Recommended Strategy |
|------|-------------|--------------------------|
| OHRC ↔ TMC-2 | ~20× | Phase congruency + semi-dense, MAGSAC |
| TMC-2 ↔ IIRS | ~16× | Deep embeddings + graph matching |
| OHRC ↔ IIRS | ~300× | Hierarchical chain via TMC-2 anchor |

Scale bridging is implemented in `SemiDenseMatcher` via coordinate rescaling:

```
mov_x_scaled = mov_x × (resolution_ref / resolution_mov)
```

## MPP Integration

Every pipeline run produces an `MPPReport`:

```
RegistrationResult
       │
       ▼
MPPMetrics.build_report()
       │
       ├── registration_rmse_px
       ├── sub_pixel_accuracy (RMSE < 0.5)
       ├── tie_point_uniformity_entropy
       ├── grid_occupancy_pct
       ├── inlier_ratio
       ├── stage_timings[]
       └── rag_retrieval_relevance
```

## Deployment Topology

```
config/default.yaml  ──▶  LunarRegistrationPipeline.from_yaml()
data/knowledge_base/ ──▶  LunarRegistrationRetriever (ChromaDB persist)
outputs/             ◀──  registered.tif + mpp_report.json
```

## Extension Points

1. **New sensor** — add to `SensorType` enum + `config/default.yaml` + RAG documents
2. **New feature extractor** — implement `FeatureExtractor`, register in `RegistrationComponentFactory`
3. **New estimator** — implement `GeometricEstimator`, register in factory
4. **New pipeline stage** — extend `BaseRegistrationStage`, add to orchestrator
