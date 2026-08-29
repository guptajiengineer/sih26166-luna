"""RAG knowledge base documents for lunar registration."""

KNOWLEDGE_DOCUMENTS = [
    {
        "id": "ohrc_tmc2_registration",
        "title": "OHRC to TMC-2 Cross-Scale Registration",
        "content": (
            "Register OHRC (0.25 m) to TMC-2 (5 m) ortho-images using phase congruency "
            "and semi-dense matching. Scale ratio ~20x. Use homography model when terrain "
            "relief is low; affine partial for moderate slopes. Sun angle differences up to "
            "30 degrees require illumination-invariant features. Target RMSE < 0.5 px at "
            "reference resolution. Recommended: phase_congruency + contour features, "
            "semi_dense matcher, MAGSAC++ estimator, reproj_threshold=2.0."
        ),
        "tags": ["OHRC", "TMC2", "scale_20x", "phase_congruency"],
    },
    {
        "id": "tmc2_iirs_registration",
        "title": "TMC-2 to IIRS Spectral Cube Registration",
        "content": (
            "TMC-2 (5 m) to IIRS (80 m) slice registration spans ~16x scale gap. "
            "IIRS spectral bands vary in SNR; use mean reflectance or PC1 slice for "
            "matching. Deep embeddings help when albedo contrast is weak. Graph matching "
            "filter recommended due to repetitive crater patterns. Minimum 80 tie points "
            "for stable homography. Spatial consistency check with neighbors=8."
        ),
        "tags": ["TMC2", "IIRS", "scale_16x", "deep_embedding", "graph_matching"],
    },
    {
        "id": "ohrc_iirs_direct",
        "title": "Direct OHRC to IIRS Registration (300x gap)",
        "content": (
            "Direct OHRC to IIRS registration (~300x scale) requires hierarchical "
            "coarse-to-fine: OHRC→TMC-2→IIRS chain preferred. If direct: downsample OHRC "
            "to 5 m, match to TMC-2 first as anchor. Use contour features at coarse scale, "
            "phase congruency at fine scale. Semi-dense matcher with ratio_threshold=0.85. "
            "Expect lower inlier ratio (~40-60%); increase max_iters to 20000."
        ),
        "tags": ["OHRC", "IIRS", "scale_300x", "hierarchical"],
    },
    {
        "id": "sun_angle_invariance",
        "title": "Illumination Invariance for Different Sun Angles",
        "content": (
            "When alpha_sun1 != alpha_sun2, avoid raw intensity matching. Phase congruency "
            "is robust to shadows up to ~45 degree sun difference. Contour features on "
            "gradient-normalized images reduce albedo bias. Deep embeddings (trained on "
            "terrestrial domain) transfer moderately to lunar; combine with PC features. "
            "Preprocessing: subtract low-frequency illumination via large Gaussian (sigma=50)."
        ),
        "tags": ["illumination", "sun_angle", "phase_congruency", "contour"],
    },
    {
        "id": "mpp_targets",
        "title": "Measurable Performance Targets",
        "content": (
            "MPP targets for lunar registration: (1) RMSE < 0.5 px sub-pixel at reference, "
            "(2) inlier ratio > 60%, (3) tie-point spatial entropy > 3.5 bits (uniform coverage), "
            "(4) grid cell occupancy > 70% for 16x16 grid, (5) stage latency ref<30s per MP tile, "
            "(6) RAG retrieval relevance cosine > 0.7 for parameter selection."
        ),
        "tags": ["MPP", "metrics", "quality"],
    },
    {
        "id": "outlier_rejection",
        "title": "Robust Outlier Rejection Strategies",
        "content": (
            "MAGSAC++ (USAC_MAGSAC) preferred for homography with unknown outlier rate. "
            "Graph matching effective when >30% false matches from repetitive terrain. "
            "Spatial consistency enforces local affine coherence. Pipeline order: "
            "match → MAGSAC → spatial consistency (optional second pass). "
            "reproj_threshold: 1.5 px for OHRC, 2.5 px for TMC-2, 4.0 px for IIRS."
        ),
        "tags": ["MAGSAC", "graph_matching", "spatial_consistency", "outliers"],
    },
]
