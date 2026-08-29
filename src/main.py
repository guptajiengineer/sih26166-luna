"""CLI entry point for lunar registration pipeline."""

from __future__ import annotations

from pathlib import Path

import click
import cv2
import numpy as np

from src.models.domain import SensorType
from src.pipeline.orchestrator import LunarRegistrationPipeline


@click.group()
def cli() -> None:
    """Lunar Image Registration RAG Pipeline."""


@cli.command()
@click.option("--ref", required=True, type=click.Path(exists=True), help="Reference image path")
@click.option("--mov", required=True, type=click.Path(exists=True), help="Moving image path")
@click.option("--ref-sensor", default="OHRC", type=click.Choice(["OHRC", "TMC2", "IIRS"]))
@click.option("--mov-sensor", default="TMC2", type=click.Choice(["OHRC", "TMC2", "IIRS"]))
@click.option("--sun-ref", default=30.0, type=float, help="Reference sun angle (deg)")
@click.option("--sun-mov", default=55.0, type=float, help="Moving sun angle (deg)")
@click.option("--config", default="config/default.yaml", type=click.Path(exists=True))
@click.option("--output", default="outputs", type=click.Path())
def register(
    ref: str,
    mov: str,
    ref_sensor: str,
    mov_sensor: str,
    sun_ref: float,
    sun_mov: float,
    config: str,
    output: str,
) -> None:
    """Run full RAG-augmented registration pipeline."""
    RESOLUTIONS = {"OHRC": 0.25, "TMC2": 5.0, "IIRS": 80.0}

    pipeline = LunarRegistrationPipeline.from_yaml(config)
    reference = pipeline.load_image(ref, SensorType(ref_sensor), sun_ref, RESOLUTIONS[ref_sensor])
    moving = pipeline.load_image(mov, SensorType(mov_sensor), sun_mov, RESOLUTIONS[mov_sensor])

    ctx, mpp = pipeline.register(reference, moving)

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ctx.registered_image is not None:
        reg_path = out_dir / "registered.tif"
        cv2.imwrite(str(reg_path), ctx.registered_image.astype(np.float32))

    mpp_path = out_dir / "mpp_report.json"
    mpp.save(mpp_path)

    click.echo(f"Registration complete.")
    click.echo(f"  Tie points (inliers): {mpp.tie_point_count}")
    click.echo(f"  RMSE: {mpp.registration_rmse_px:.4f} px")
    click.echo(f"  Inlier ratio: {mpp.inlier_ratio:.2%}")
    click.echo(f"  MPP pass: {mpp.passes_mpp()}")
    click.echo(f"  Report: {mpp_path}")


@cli.command()
@click.option("--query", required=True, help="RAG query for parameter advice")
@click.option("--top-k", default=5, type=int)
def rag_query(query: str, top_k: int) -> None:
    """Query the RAG knowledge base."""
    from src.rag.retriever import LunarRegistrationRetriever

    retriever = LunarRegistrationRetriever()
    results = retriever.retrieve(query, top_k=top_k)
    for i, r in enumerate(results, 1):
        click.echo(f"\n--- Result {i} (relevance={r.get('relevance', 0):.3f}) ---")
        click.echo(r.get("content", r.get("title", "")))


@cli.command()
def demo() -> None:
    """Run demo with synthetic lunar-like image pair."""
    from src.models.domain import LunarImage, SensorType

    click.echo("Generating synthetic lunar image pair...")
    h, w = 512, 512
    rng = np.random.default_rng(42)

    # Synthetic crater-like terrain
    y, x = np.ogrid[:h, :w]
    craters = np.zeros((h, w), dtype=np.float32)
    for _ in range(15):
        cx, cy = rng.integers(50, w - 50), rng.integers(50, h - 50)
        r = rng.integers(10, 40)
        crater = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (r / 2) ** 2))
        craters += crater * rng.uniform(0.3, 1.0)

    ref_data = craters + rng.normal(0, 0.05, (h, w))
    # Simulate different sun angle (directional shading)
    shading = 0.3 * (x / w) + 0.2 * (y / h)
    mov_data = craters * (1 + shading) + rng.normal(0, 0.08, (h, w))

    # Downsample moving to simulate TMC-2 scale (~20x)
    mov_small = cv2.resize(mov_data, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    mov_up = cv2.resize(mov_small, (w, h), interpolation=cv2.INTER_LINEAR)

    ref = LunarImage(data=ref_data, sensor=SensorType.OHRC, sun_angle_deg=30, resolution_m=0.25)
    mov = LunarImage(data=mov_up, sensor=SensorType.TMC2, sun_angle_deg=55, resolution_m=5.0)

    pipeline = LunarRegistrationPipeline.from_yaml("config/default.yaml")
    ctx, mpp = pipeline.register(ref, mov)

    out_dir = Path("outputs/demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    mpp.save(out_dir / "mpp_report.json")

    click.echo(f"\nDemo complete — RMSE: {mpp.registration_rmse_px:.4f} px, MPP pass: {mpp.passes_mpp()}")


if __name__ == "__main__":
    cli()
