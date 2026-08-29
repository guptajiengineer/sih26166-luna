# Codebase — File Structure

```
lunar-registration-rag/
│
├── README.md                          # Project overview, quick start, MPP summary
├── Architecture.md                    # System design, data flow, component diagrams
├── Explanation.md                     # Algorithms, design patterns, MPP, build order
├── Codebase.md                        # This file — directory reference
├── pyproject.toml                     # Package metadata, dependencies, CLI entry point
├── requirements.txt                   # Full dependency list (incl. optional RAG/deep)
│
├── config/
│   └── default.yaml                   # Pipeline, sensor, RAG, and MPP configuration
│
├── data/
│   └── knowledge_base/
│       ├── documents.py               # Curated RAG knowledge documents (6 entries)
│       └── chroma/                    # (generated) ChromaDB persistence directory
│
├── src/
│   ├── __init__.py                    # Package version
│   ├── main.py                        # CLI: register, rag-query, demo commands
│   │
│   ├── models/
│   │   └── domain.py                  # LunarImage, Keypoint, TiePoint, RegistrationResult
│   │
│   ├── patterns/
│   │   ├── __init__.py                # Pipeline, Strategy, Factory, Observer, Chain, Template
│   │   └── factory.py                 # RegistrationComponentFactory
│   │
│   ├── features/
│   │   ├── base.py                    # FeatureExtractor ABC + gray conversion helper
│   │   ├── phase_congruency.py        # Multi-scale PC proxy extractor
│   │   ├── contour.py                 # Adaptive-threshold contour keypoints
│   │   └── deep_embeddings.py         # MobileNet / ORB embedding extractor
│   │
│   ├── matching/
│   │   ├── base.py                    # Matcher ABC + ratio test helper
│   │   ├── semi_dense_matcher.py      # Cross-scale semi-dense matching + sub-pixel refine
│   │   └── dense_matcher.py           # Farneback optical-flow dense matcher
│   │
│   ├── geometry/
│   │   ├── base.py                    # GeometricEstimator ABC + point array helpers
│   │   ├── magsac.py                  # USAC_MAGSAC homography/affine estimation
│   │   ├── graph_matching.py          # Clique-based consistency graph filter
│   │   └── spatial_consistency.py     # Local neighborhood deviation filter
│   │
│   ├── rag/
│   │   ├── retriever.py               # ChromaDB + sentence-transformers retriever
│   │   └── context_builder.py         # Chain-of-responsibility RAG context handlers
│   │
│   ├── mpp/
│   │   └── metrics.py                 # MPPMetrics, MPPReport, StageTimer
│   │
│   └── pipeline/
│       ├── context.py                 # PipelineContext shared state dataclass
│       ├── stages.py                  # 6 pipeline stages (RAG → Register)
│       └── orchestrator.py            # LunarRegistrationPipeline main class
│
├── tests/
│   └── test_pipeline.py               # RAG, MPP, and end-to-end pipeline tests
│
└── outputs/                           # (generated) registered images + MPP reports
    └── demo/
        └── mpp_report.json
```

## Module Dependency Graph

```
main.py
  └── pipeline/orchestrator.py
        ├── pipeline/stages.py
        │     ├── rag/context_builder.py → rag/retriever.py → data/knowledge_base/documents.py
        │     ├── patterns/factory.py
        │     │     ├── features/*.py
        │     │     ├── matching/*.py
        │     │     └── geometry/*.py
        │     └── mpp/metrics.py
        ├── patterns/__init__.py
        └── models/domain.py
```

## Key Entry Points

| File | Purpose | Run As |
|------|---------|--------|
| `src/main.py` | CLI interface | `python -m src.main demo` |
| `src/pipeline/orchestrator.py` | Programmatic API | `LunarRegistrationPipeline().register(ref, mov)` |
| `src/rag/retriever.py` | Standalone RAG queries | `LunarRegistrationRetriever().retrieve(query)` |
| `src/mpp/metrics.py` | Standalone metrics | `MPPMetrics.build_report(...)` |

## Configuration Files

| File | Controls |
|------|----------|
| `config/default.yaml` | Sensor resolutions, default extractors/matcher/estimator, RAG model, MPP thresholds |
| `pyproject.toml` | Package name, optional dependency groups (`rag`, `deep`, `dev`) |

## Generated / Runtime Artifacts

| Path | Created By | Contents |
|------|-----------|----------|
| `data/knowledge_base/chroma/` | First RAG query | ChromaDB vector index |
| `outputs/registered.tif` | `register` command | Co-registered moving image |
| `outputs/mpp_report.json` | Every pipeline run | MPP metrics + pass/fail |

## Adding New Files

| To add… | Create… | Register in… |
|---------|---------|-------------|
| New feature extractor | `src/features/my_extractor.py` | `patterns/factory.py` `_FEATURES` |
| New matcher | `src/matching/my_matcher.py` | `patterns/factory.py` `_MATCHERS` |
| New estimator | `src/geometry/my_estimator.py` | `patterns/factory.py` `_ESTIMATORS` |
| New pipeline stage | `src/pipeline/stages.py` | `pipeline/orchestrator.py` `_build_pipeline()` |
| New RAG document | `data/knowledge_base/documents.py` | Auto-indexed on first run |
| New sensor type | `models/domain.py` `SensorType` | `config/default.yaml` sensors section |
