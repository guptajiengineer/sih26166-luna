# Explanation

## Problem Decomposition

The lunar registration problem has three hard constraints that this pipeline addresses:

1. **Illumination variance** — different Sun angles ($\alpha_{\text{sun1}} \neq \alpha_{\text{sun2}}$) change shadows and albedo appearance.
2. **Scale gap** — OHRC (0.25 m), TMC-2 (5 m), and IIRS (80 m) differ by 20×–300× in ground resolution.
3. **Outlier-heavy matching** — repetitive crater terrain produces many false correspondences.

**RAG's role:** Before running any CV algorithm, the system queries a knowledge base of registration strategies (sensor-pair recipes, threshold values, feature recommendations) and configures the pipeline dynamically.

---

## Algorithm Walkthrough

### Step 1 — RAG Parameter Selection

Given `(ref_sensor=OHRC, mov_sensor=TMC2, sun_diff=25°)`, the retriever embeds a natural-language query and returns the top-k relevant documents. These are parsed into concrete parameters:

```python
# src/rag/retriever.py
params = retriever.suggest_parameters("OHRC", "TMC2", sun_diff=25.0)
# → feature_extractors: [phase_congruency, contour]
# → matcher: semi_dense
# → geometric_estimator: magsac
# → reproj_threshold: 2.0
```

### Step 2 — Illumination-Invariant Feature Extraction

Three complementary extractors run in parallel (Strategy pattern):

| Extractor | Invariance Mechanism |
|-----------|---------------------|
| **Phase Congruency** | Multi-scale gradient magnitude — responds to structure, not absolute intensity |
| **Contour** | Adaptive threshold on gradient-normalized image — edges survive shadow shifts |
| **Deep Embedding** | CNN features from local patches — semantic structure beyond raw pixels |

```python
# src/features/phase_congruency.py (simplified)
for scale in [1, 2, 4, 8]:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=scale)
    gx, gy = cv2.Sobel(blurred, ...), cv2.Sobel(blurred, ...)
    pc_map += sqrt(gx² + gy²)
```

### Step 3 — Cross-Scale Semi-Dense Matching

Key insight: rescale moving-image coordinates to the reference frame before descriptor matching.

```python
# src/matching/semi_dense_matcher.py
scaled_mov_x = mov_x * (resolution_ref / resolution_mov)
# OHRC 0.25m vs TMC-2 5.0m → scale factor = 0.05
```

Lowe's ratio test filters ambiguous matches. Sub-pixel refinement uses `cv2.matchTemplate` on local patches.

### Step 4 — Robust Geometric Filtering

Three estimators available (selected by RAG or config):

- **MAGSAC++** (`cv2.USAC_MAGSAC`) — adaptive threshold RANSAC, no prior outlier rate needed
- **Graph Matching** — builds consistency graph, keeps largest clique of mutually consistent matches
- **Spatial Consistency** — rejects matches whose local neighborhood affine deviates

```python
# src/geometry/graph_matching.py
for i, j in pairs:
    if abs(dist_src(i,j) - dist_dst(i,j)) < tolerance:
        G.add_edge(i, j)
best_clique = max(nx.find_cliques(G), key=len)
```

### Step 5 — Registration Output

Homography $H$ warps the moving image to the reference grid:

$$p_{\text{ref}} = H \cdot p_{\text{mov}}$$

Output artifacts:
- `registered.tif` — pixel-aligned co-registered layer
- List of `TiePoint` objects with sub-pixel coordinates
- `mpp_report.json` — measurable quality metrics

---

## Design Patterns — Explained with Code

### 1. Pipeline Pattern

**Why:** Registration is inherently sequential; each stage depends on the previous output.

```python
# src/patterns/__init__.py
class Pipeline(Generic[T]):
    def run(self, context: T) -> T:
        for stage in self._stages:
            context = stage.run(context)
        return context

# src/pipeline/orchestrator.py
pipe.add_stage(RAGParameterStage())
pipe.add_stage(PreprocessingStage())
pipe.add_stage(FeatureExtractionStage())
# ...
ctx = pipe.run(ctx)
```

### 2. Strategy Pattern

**Why:** Feature extractors, matchers, and estimators are interchangeable algorithms with the same interface.

```python
# src/patterns/__init__.py
class Strategy(ABC, Generic[T, R]):
    @abstractmethod
    def execute(self, context: T) -> R: ...

# src/features/base.py
class FeatureExtractor(Strategy[tuple[LunarImage, dict], list[Keypoint]]):
    @abstractmethod
    def extract(self, image, params) -> list[Keypoint]: ...

# Usage — swap without changing pipeline code:
extractors = factory.create_feature_extractors(["phase_congruency", "contour"])
for ext in extractors:
    keypoints.extend(ext.execute((image, params)))
```

### 3. Factory Pattern

**Why:** Components are created by name from config/RAG output without `if/elif` chains in the pipeline.

```python
# src/patterns/factory.py
class RegistrationComponentFactory(ComponentFactory):
    _FEATURES = {"contour": ContourFeatureExtractor, ...}
    _MATCHERS = {"semi_dense": SemiDenseMatcher, ...}
    _ESTIMATORS = {"magsac": MAGSACEstimator, ...}

    def create(self, component_type, **kwargs):
        return self._FEATURES[name](**kwargs)
```

### 4. Template Method Pattern

**Why:** All stages share validate → process → annotate skeleton; subclasses only implement `process`.

```python
# src/patterns/__init__.py
class BaseRegistrationStage(PipelineStage[T], ABC):
    def run(self, context: T) -> T:
        self.validate(context)
        result = self.process(context)   # ← subclass implements
        return self.annotate(result)
```

### 5. Chain of Responsibility (RAG)

**Why:** Context enrichment (sensor info → sun angle → retrieval) is modular and extensible.

```python
# src/rag/context_builder.py
sensor = SensorContextHandler()
sun = SunAngleContextHandler()
rag = RAGRetrievalHandler(retriever)
sensor.set_next(sun).set_next(rag)

enriched = sensor.handle({"ref_sensor": "OHRC", "mov_sensor": "IIRS", ...})
```

### 6. Observer Pattern

**Why:** Decouple metrics/logging from stage logic.

```python
# src/patterns/__init__.py
class MetricsObserver(PipelineObserver):
    def on_event(self, event: str, payload: Any) -> None:
        self.events.append((event, payload))

pipe.attach(lambda event, ctx: observer.on_event(event, ctx))
```

---

## Measurable Part of the Project (MPP)

The MPP defines **quantifiable success criteria** — not just "it works" but measurable proof:

| # | Metric | Formula / Method | Pass Threshold |
|---|--------|-----------------|----------------|
| 1 | Registration RMSE | Mean $\|H \cdot p_{mov} - p_{ref}\|$ over inliers | < 0.5 px |
| 2 | Sub-pixel accuracy | Boolean: RMSE < 0.5 | `true` |
| 3 | Inlier ratio | `#inliers / #raw_matches` | ≥ 60% |
| 4 | Spatial entropy | $-\sum p_i \log_2 p_i$ over 16×16 grid | ≥ 3.5 bits |
| 5 | Grid occupancy | `%` of 256 grid cells with ≥1 tie point | ≥ 70% |
| 6 | RAG relevance | Mean cosine similarity of retrieved chunks | ≥ 0.7 |
| 7 | Stage latency | `time.perf_counter()` per stage | < 30 s/tile |

```python
# src/mpp/metrics.py
report = MPPMetrics.build_report(tie_points, H, image_shape, timings, rag_relevance)
report.passes_mpp()  # True if ALL thresholds met
report.save("outputs/mpp_report.json")
```

---

## Build Piece by Piece

Recommended build order (each piece is independently testable):

| Piece | Module | Verify With |
|-------|--------|-------------|
| **1. Domain models** | `src/models/domain.py` | Unit tests for dataclasses |
| **2. Design patterns** | `src/patterns/` | Import + instantiate Pipeline |
| **3. Feature extractors** | `src/features/` | Extract keypoints from synthetic image |
| **4. Matchers** | `src/matching/` | Match two synthetic keypoint sets |
| **5. Geometry filters** | `src/geometry/` | Filter synthetic outliers |
| **6. RAG layer** | `src/rag/` | `rag-query` CLI command |
| **7. Pipeline stages** | `src/pipeline/stages.py` | Run single stage on context |
| **8. Orchestrator** | `src/pipeline/orchestrator.py` | Full end-to-end |
| **9. MPP metrics** | `src/mpp/metrics.py` | Check report thresholds |
| **10. CLI + docs** | `src/main.py`, docs/ | `demo` command |

```bash
# After each piece:
pytest tests/test_pipeline.py -k "test_name" -v
```

---

## Why RAG + Algorithmic Pipeline?

Pure deep-learning registration struggles with 300× scale gaps and limited labeled lunar pairs. Pure classical CV lacks adaptive parameter selection. This hybrid approach:

1. **RAG** selects the right classical algorithm recipe for each sensor pair
2. **Classical CV** provides interpretable, sub-pixel geometric accuracy
3. **MPP** provides auditable, measurable quality gates

This is the recommended architecture for planetary science pipelines where interpretability and measurable accuracy matter as much as automation.
