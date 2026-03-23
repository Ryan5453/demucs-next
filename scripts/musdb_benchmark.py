from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import statistics
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import typer

from demucs.api import Separator
from demucs.cli.models import ensure_model_available

REFERENCE_STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")
DEFAULT_MODELS = ["htdemucs", "htdemucs_ft"]
DEFAULT_PRECISIONS = ["fp32", "fp16"]
DEFAULT_COMPILE_MODES = [False, True]
DEFAULT_SPLIT_MODES = [False, True]
DEFAULT_SHIFTS = [1, 2, 4]
DEFAULT_SPLIT_SIZES = [10, 20, 30]
DEFAULT_SPLIT_OVERLAPS = [0.1, 0.25, 0.5]

app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass(frozen=True)
class BenchmarkTrack:
    name: str
    directory: Path
    mixture_path: Path
    reference_stems: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkConfig:
    config_id: str
    model: str
    precision: str
    compile: bool
    split: bool
    shifts: int
    split_size: int | None
    split_overlap: float | None


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6f}"


def _build_track_seed(seed: int, track_name: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{track_name}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % (2**31)


def _compute_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    estimate = estimate.to(dtype=torch.float64, device="cpu")
    reference = reference.to(dtype=torch.float64, device="cpu")

    length = min(estimate.shape[-1], reference.shape[-1])
    estimate = estimate[..., :length]
    reference = reference[..., :length]

    noise = estimate - reference
    reference_energy = torch.sum(reference * reference).item()
    noise_energy = torch.sum(noise * noise).item()

    if reference_energy <= 0.0:
        return float("nan")
    if noise_energy <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(reference_energy / noise_energy)


def _is_unsupported_full_track_error(error: Exception) -> bool:
    return "longer than training length" in str(error).lower()


def _discover_tracks(musdb_root: Path) -> list[BenchmarkTrack]:
    if not musdb_root.is_dir():
        raise typer.BadParameter(f"MUSDB root does not exist: {musdb_root}")

    tracks: list[BenchmarkTrack] = []
    for track_dir in sorted(d for d in musdb_root.iterdir() if d.is_dir()):
        mixture_path = track_dir / "mixture.wav"
        reference_stems = tuple(
            stem for stem in REFERENCE_STEMS if (track_dir / f"{stem}.wav").exists()
        )

        if not mixture_path.exists() or not reference_stems:
            continue

        tracks.append(
            BenchmarkTrack(
                name=track_dir.name,
                directory=track_dir,
                mixture_path=mixture_path,
                reference_stems=reference_stems,
            )
        )

    if not tracks:
        raise typer.BadParameter(
            f"No MUSDB tracks with mixture.wav and reference stems found under {musdb_root}"
        )
    return tracks


def _build_configs() -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    config_index = 1

    for model in DEFAULT_MODELS:
        for precision in DEFAULT_PRECISIONS:
            for compile_mode in DEFAULT_COMPILE_MODES:
                for split_mode in DEFAULT_SPLIT_MODES:
                    for shifts in DEFAULT_SHIFTS:
                        if split_mode:
                            for split_size in DEFAULT_SPLIT_SIZES:
                                for split_overlap in DEFAULT_SPLIT_OVERLAPS:
                                    configs.append(
                                        BenchmarkConfig(
                                            config_id=f"cfg_{config_index:04d}",
                                            model=model,
                                            precision=precision,
                                            compile=compile_mode,
                                            split=True,
                                            shifts=shifts,
                                            split_size=split_size,
                                            split_overlap=split_overlap,
                                        )
                                    )
                                    config_index += 1
                        else:
                            configs.append(
                                BenchmarkConfig(
                                    config_id=f"cfg_{config_index:04d}",
                                    model=model,
                                    precision=precision,
                                    compile=compile_mode,
                                    split=False,
                                    shifts=shifts,
                                    split_size=None,
                                    split_overlap=None,
                                )
                            )
                            config_index += 1
    return configs


def _precision_to_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported precision: {precision}")


def _detail_row_base(
    config: BenchmarkConfig,
    chunk_batch_size: int | None,
    base_seed: int | None,
) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "model": config.model,
        "precision": config.precision,
        "device": "cuda",
        "compile": config.compile,
        "split": config.split,
        "shifts": config.shifts,
        "split_size": config.split_size,
        "split_overlap": config.split_overlap,
        "chunk_batch_size": chunk_batch_size,
        "base_seed": base_seed,
    }


def _summary_row_base(
    config: BenchmarkConfig,
    chunk_batch_size: int | None,
    base_seed: int | None,
) -> dict[str, Any]:
    return {
        "config_id": config.config_id,
        "model": config.model,
        "precision": config.precision,
        "device": "cuda",
        "compile": config.compile,
        "split": config.split,
        "shifts": config.shifts,
        "split_size": config.split_size,
        "split_overlap": config.split_overlap,
        "chunk_batch_size": chunk_batch_size,
        "base_seed": base_seed,
    }


@app.command()
def main(
    musdb_root: Path = typer.Option(
        ...,
        "--musdb-root",
        help="Path to the MUSDB18-HQ split directory containing per-track folders",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory to write benchmark CSV/JSON results into",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Limit benchmarking to the first N tracks",
    ),
    seed: int | None = typer.Option(
        1234,
        "--seed",
        help="Base random seed used to derive a deterministic seed per track",
    ),
    chunk_batch_size: int | None = typer.Option(
        None,
        "--chunk-batch-size",
        min=1,
        help="Override how many split chunks are processed per batch",
    ),
) -> None:
    """
    Benchmark the built-in MUSDB CUDA matrix and record SDR plus timing.
    """
    if not torch.cuda.is_available():
        raise typer.BadParameter("CUDA benchmarking was requested but CUDA is not available.")

    tracks = _discover_tracks(musdb_root)
    if limit is not None:
        tracks = tracks[:limit]

    configs = _build_configs()

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("benchmarks") / "musdb" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Benchmarking {len(configs)} configs across {len(tracks)} MUSDB tracks")
    typer.echo(f"Results directory: {output_dir}")

    details_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for config_index, config in enumerate(configs, start=1):
        label = (
            f"[{config_index}/{len(configs)}] {config.config_id} "
            f"model={config.model} precision={config.precision} "
            f"compile={config.compile} split={config.split} "
            f"shifts={config.shifts}"
        )
        if config.split:
            label += (
                f" split_size={config.split_size} split_overlap={config.split_overlap}"
            )
        typer.echo(label)

        if not ensure_model_available(config.model):
            summary_rows.append(
                {
                    **_summary_row_base(config, chunk_batch_size, seed),
                    "status": "model_unavailable",
                    "error_type": "ModelUnavailable",
                    "error_message": f"Could not download or load model '{config.model}'",
                    "num_tracks": len(tracks),
                }
            )
            continue

        init_started_at = perf_counter()
        try:
            separator = Separator(
                model=config.model,
                device="cuda",
                dtype=_precision_to_dtype(config.precision),
                compile=config.compile,
            )
        except Exception as error:
            summary_rows.append(
                {
                    **_summary_row_base(config, chunk_batch_size, seed),
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "num_tracks": len(tracks),
                    "model_init_sec": perf_counter() - init_started_at,
                }
            )
            typer.echo(f"Failed to initialize {config.config_id}: {error}")
            continue

        model_init_sec = perf_counter() - init_started_at
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak_vram_bytes = 0

        successful_track_rows: list[dict[str, Any]] = []
        error_count = 0
        oom_count = 0
        unsupported_length_count = 0
        unsupported_config = False
        config_error_type = ""
        config_error_message = ""

        for track_index, track in enumerate(tracks, start=1):
            detail_row = {
                **_detail_row_base(config, chunk_batch_size, seed),
                "track_index": track_index,
                "track_name": track.name,
                "track_seed": None if seed is None else _build_track_seed(seed, track.name),
            }

            started_at = perf_counter()
            try:
                separated = separator.separate(
                    audio=track.mixture_path,
                    shifts=config.shifts,
                    split=config.split,
                    split_size=config.split_size,
                    split_overlap=0.25 if config.split_overlap is None else config.split_overlap,
                    seed=detail_row["track_seed"],
                    chunk_batch_size=chunk_batch_size,
                )
                elapsed_sec = perf_counter() - started_at
                detail_row["elapsed_sec"] = elapsed_sec

                stem_scores: dict[str, float] = {}
                for stem_name in track.reference_stems:
                    if stem_name not in separated.sources:
                        continue
                    reference = separator._to_tensor(track.directory / f"{stem_name}.wav")
                    stem_scores[stem_name] = _compute_sdr(
                        separated.sources[stem_name],
                        reference,
                    )

                detail_row["status"] = "ok"
                detail_row["error_type"] = ""
                detail_row["error_message"] = ""
                detail_row["num_scored_stems"] = len(stem_scores)
                detail_row["mean_sdr"] = (
                    statistics.fmean(stem_scores.values()) if stem_scores else float("nan")
                )
                for stem_name in REFERENCE_STEMS:
                    detail_row[f"{stem_name}_sdr"] = stem_scores.get(stem_name)

                successful_track_rows.append(detail_row)
                details_rows.append(detail_row)
            except Exception as error:
                elapsed_sec = perf_counter() - started_at
                error_type = type(error).__name__
                is_oom = isinstance(error, torch.OutOfMemoryError) or (
                    "out of memory" in str(error).lower()
                )
                is_unsupported_length = (
                    not config.split and _is_unsupported_full_track_error(error)
                )

                detail_row["elapsed_sec"] = elapsed_sec
                if is_unsupported_length:
                    detail_row["status"] = "unsupported_length"
                else:
                    detail_row["status"] = "oom" if is_oom else "error"
                detail_row["error_type"] = error_type
                detail_row["error_message"] = str(error)
                detail_row["num_scored_stems"] = 0
                detail_row["mean_sdr"] = float("nan")
                for stem_name in REFERENCE_STEMS:
                    detail_row[f"{stem_name}_sdr"] = None

                details_rows.append(detail_row)
                if is_oom:
                    oom_count += 1
                elif is_unsupported_length:
                    unsupported_length_count += 1
                    unsupported_config = True
                    config_error_type = error_type
                    config_error_message = str(error)
                else:
                    error_count += 1

                typer.echo(
                    f"{config.config_id} track {track.name} failed with "
                    f"{detail_row['status']}: {error_type}: {error}"
                )
                traceback.print_exc()
                torch.cuda.empty_cache()

                if unsupported_config:
                    for skipped_track_index, skipped_track in enumerate(
                        tracks[track_index:],
                        start=track_index + 1,
                    ):
                        details_rows.append(
                            {
                                **_detail_row_base(config, chunk_batch_size, seed),
                                "track_index": skipped_track_index,
                                "track_name": skipped_track.name,
                                "track_seed": None
                                if seed is None
                                else _build_track_seed(seed, skipped_track.name),
                                "status": "skipped_unsupported_length",
                                "error_type": error_type,
                                "error_message": str(error),
                                "elapsed_sec": None,
                                "num_scored_stems": 0,
                                "mean_sdr": float("nan"),
                                **{
                                    f"{stem_name}_sdr": None
                                    for stem_name in REFERENCE_STEMS
                                },
                            }
                        )
                    break

            peak_vram_bytes = max(
                peak_vram_bytes,
                int(torch.cuda.max_memory_allocated()),
            )

        config_detail_rows = [
            row for row in details_rows if row["config_id"] == config.config_id
        ]
        ok_rows = [row for row in successful_track_rows if row["status"] == "ok"]
        ok_elapsed = [float(row["elapsed_sec"]) for row in ok_rows]
        remaining_elapsed = ok_elapsed[1:]
        attempted_elapsed = [
            float(row["elapsed_sec"])
            for row in config_detail_rows
            if row.get("elapsed_sec") is not None
        ]
        mean_sdr_values = [
            float(row["mean_sdr"])
            for row in ok_rows
            if not math.isnan(float(row["mean_sdr"]))
        ]

        per_stem_summary: dict[str, float] = {}
        for stem_name in REFERENCE_STEMS:
            stem_values = [
                float(row[f"{stem_name}_sdr"])
                for row in ok_rows
                if row.get(f"{stem_name}_sdr") is not None
                and not math.isnan(float(row[f"{stem_name}_sdr"]))
            ]
            per_stem_summary[stem_name] = (
                statistics.fmean(stem_values) if stem_values else float("nan")
            )

        summary_rows.append(
            {
                **_summary_row_base(config, chunk_batch_size, seed),
                "status": (
                    "unsupported_length"
                    if unsupported_config
                    else "ok" if len(ok_rows) == len(tracks) else "partial"
                ),
                "error_type": config_error_type,
                "error_message": config_error_message,
                "num_tracks": len(tracks),
                "ok_tracks": len(ok_rows),
                "error_tracks": error_count,
                "oom_tracks": oom_count,
                "unsupported_length_tracks": unsupported_length_count,
                "model_init_sec": model_init_sec,
                "first_attempt_sec": (
                    float(config_detail_rows[0]["elapsed_sec"]) if config_detail_rows else None
                ),
                "first_attempt_status": (
                    str(config_detail_rows[0]["status"]) if config_detail_rows else ""
                ),
                "first_track_sec": ok_elapsed[0] if ok_elapsed else None,
                "remaining_total_sec": sum(remaining_elapsed) if remaining_elapsed else 0.0,
                "steady_state_mean_sec": (
                    statistics.fmean(remaining_elapsed) if remaining_elapsed else None
                ),
                "steady_state_median_sec": (
                    statistics.median(remaining_elapsed) if remaining_elapsed else None
                ),
                "attempted_track_sec": sum(attempted_elapsed),
                "track_total_sec": sum(ok_elapsed),
                "mean_sdr": (
                    statistics.fmean(mean_sdr_values) if mean_sdr_values else float("nan")
                ),
                "median_sdr": (
                    statistics.median(mean_sdr_values) if mean_sdr_values else float("nan")
                ),
                "peak_vram_mb": peak_vram_bytes / (1024 * 1024) if peak_vram_bytes else None,
                **{f"{stem}_mean_sdr": value for stem, value in per_stem_summary.items()},
            }
        )

        del separator
        gc.collect()
        torch.cuda.empty_cache()

    detail_csv = output_dir / "benchmark_details.csv"
    summary_csv = output_dir / "benchmark_summary.csv"
    metadata_json = output_dir / "benchmark_metadata.json"

    detail_fieldnames = [
        "config_id",
        "model",
        "precision",
        "device",
        "compile",
        "split",
        "shifts",
        "split_size",
        "split_overlap",
        "chunk_batch_size",
        "base_seed",
        "track_index",
        "track_name",
        "track_seed",
        "status",
        "error_type",
        "error_message",
        "elapsed_sec",
        "num_scored_stems",
        "mean_sdr",
        *[f"{stem}_sdr" for stem in REFERENCE_STEMS],
    ]
    with open(detail_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        for row in details_rows:
            writer.writerow(
                {
                    key: _format_float(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    summary_fieldnames = [
        "config_id",
        "model",
        "precision",
        "device",
        "compile",
        "split",
        "shifts",
        "split_size",
        "split_overlap",
        "chunk_batch_size",
        "base_seed",
        "status",
        "error_type",
        "error_message",
        "num_tracks",
        "ok_tracks",
        "error_tracks",
        "oom_tracks",
        "unsupported_length_tracks",
        "model_init_sec",
        "first_attempt_sec",
        "first_attempt_status",
        "first_track_sec",
        "remaining_total_sec",
        "steady_state_mean_sec",
        "steady_state_median_sec",
        "attempted_track_sec",
        "track_total_sec",
        "peak_vram_mb",
        "mean_sdr",
        "median_sdr",
        *[f"{stem}_mean_sdr" for stem in REFERENCE_STEMS],
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    key: _format_float(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    metadata = {
        "musdb_root": str(musdb_root),
        "output_dir": str(output_dir),
        "device": "cuda",
        "models": DEFAULT_MODELS,
        "precisions": DEFAULT_PRECISIONS,
        "compile_modes": DEFAULT_COMPILE_MODES,
        "split_modes": DEFAULT_SPLIT_MODES,
        "shifts": DEFAULT_SHIFTS,
        "split_sizes": DEFAULT_SPLIT_SIZES,
        "split_overlaps": DEFAULT_SPLIT_OVERLAPS,
        "seed": seed,
        "limit": limit,
        "num_tracks": len(tracks),
        "num_configs": len(configs),
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))

    typer.echo(f"Wrote details: {detail_csv}")
    typer.echo(f"Wrote summary: {summary_csv}")
    typer.echo(f"Wrote metadata: {metadata_json}")


if __name__ == "__main__":
    app()
