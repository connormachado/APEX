"""End-to-end APEX pipeline demo on synthetic data — no hardware required.

Generates a mock SD-card capture (.BIN + YAML sidecar), runs it through the
real ingestion pipeline (parse -> validate metadata -> per-attempt HDF5
windowing -> QC), verifies the data round-trips bit-exactly, and prints a
summary.

    python run_pipeline.py --n-attempts 5 --seed 42
    python run_pipeline.py --n-attempts 5 --seed 42 --strict --keep-artifacts

Exit codes: 0 = success, 1 = QC threshold violation under --strict.

This is a pipeline correctness proof, not a data source. The .BIN it writes is
byte-format-identical to a real capture, but the signal inside is synthetic and
is labeled as such in the metadata — never mix its output into a real dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

import synthetic_imu
from ingest import (
    ACCEL_SCALE_G,
    GYRO_SCALE_DPS,
    MAX_CRC_FAILURE_RATE,
    MAX_DROPPED_FRAME_RATE,
    NUM_IMUS,
    build_qc_report,
    load_session_metadata,
    parse_bin_file,
    write_hdf5,
)

# QC report keys that legitimately differ between two identical runs. Used by
# the determinism test to compare reports without wall-clock noise.
VOLATILE_QC_KEYS: tuple[str, ...] = ("ingest_timestamp_utc",)

BAR = "═" * 76


def verify_round_trip(
    hdf5_path: Path, synth: synthetic_imu.SyntheticSession
) -> tuple[bool, float]:
    """Check that every generated LSB survived to the HDF5 unchanged.

    Compares against the *quantized* values actually encoded into the frames,
    so an exact match is the correct expectation: the conversion in ingest is
    `raw.astype(float32) * float32(scale)`, and this recomputes it identically.
    Any nonzero error means a real defect (bad demux, wrong window, unit slip),
    not floating-point noise.
    """
    max_error = 0.0
    ok = True

    with h5py.File(hdf5_path, "r") as h5:
        for attempt_id, per_imu in synth.truth.items():
            for imu, expected in per_imu.items():
                grp = h5[f"attempts/{attempt_id}/imu_{imu}"]

                got_ts = grp["timestamps_us"][:]
                if not np.array_equal(got_ts, expected["timestamps_us"]):
                    ok = False
                    continue

                want_gyro = expected["gyro_raw"].astype(np.float32) * np.float32(
                    GYRO_SCALE_DPS
                )
                want_accel = expected["accel_raw"].astype(np.float32) * np.float32(
                    ACCEL_SCALE_G
                )
                for name, want in (("gyro_dps", want_gyro), ("accel_g", want_accel)):
                    got = grp[name][:]
                    if got.shape != want.shape:
                        ok = False
                        continue
                    err = float(np.max(np.abs(got - want))) if got.size else 0.0
                    max_error = max(max_error, err)
                    if err != 0.0:
                        ok = False

    return ok, max_error


def _format_summary(result: dict[str, Any]) -> str:
    """Build the screenshot-able summary block."""
    report = result["qc_report"]
    raw = report["raw_file"]
    lines: list[str] = []

    lines.append(BAR)
    lines.append(" APEX PIPELINE — SYNTHETIC END-TO-END RUN")
    lines.append(BAR)
    lines.append(f" seed                  : {result['seed']}")
    lines.append(f" session id            : {report['session_id']}")
    lines.append(
        f" synthetic capture     : {result['bin_bytes']:,} bytes "
        f"({result['frames_generated']:,} frames, 32-byte v2)"
    )
    lines.append(f" capture duration      : {result['duration_s']:.2f} s")
    lines.append(f" sha256                : {raw['sha256'][:16]}…")
    # KNOWN: SYNTHETIC SCALE factors are placeholders — surfaced on every run so
    # nobody reads these magnitudes as physically meaningful.
    lines.append(
        f" scale factors         : {synthetic_imu.SCALE_FACTORS_VERSION} "
        f"(accel {synthetic_imu.ACCEL_G_PER_UNIT} g/unit, "
        f"gyro {synthetic_imu.GYRO_DPS_PER_UNIT:g} dps/unit) — KNOWN placeholder"
    )
    if result["n_clipped"]:
        lines.append(f" [WARN] clipped samples: {result['n_clipped']:,} hit the sensor FS")

    lines.append("")
    lines.append(" PARSE")
    lines.append(
        f"   frames parsed       : {raw['frames_parsed']:,} of "
        f"{result['frames_generated']:,} generated "
        f"({100.0 * raw['frames_parsed'] / max(result['frames_generated'], 1):.2f}%)"
    )
    lines.append(f"   crc failures        : {raw['crc_failures']:,}")
    lines.append(f"   sync hit rate       : {raw['sync_hit_rate']:.2%}")
    lines.append(f"   bad imu index       : {raw['bad_index_discards']:,}")

    lines.append("")
    lines.append(f" ATTEMPT WINDOWS  ({len(report['per_attempt'])} attempts split out)")
    lines.append(
        f"   {'attempt_id':<20} {'class':<8} {'grade':<6} {'dur[s]':>8}   frames per imu 0-4"
    )
    lines.append("   " + "-" * 70)
    by_id = {a["attempt_id"]: a for a in result["attempts"]}
    for entry in report["per_attempt"]:
        meta = by_id[entry["attempt_id"]]
        counts = " ".join(
            f"{entry['per_imu'][f'imu_{i}']['frame_count']:>5}" for i in range(NUM_IMUS)
        )
        lines.append(
            f"   {entry['attempt_id']:<20} {meta['class_name']:<8} "
            f"{meta['posted_grade']:<6} {entry['duration_s']:>8.2f}   {counts}"
        )

    lines.append("")
    lines.append(" ROUND-TRIP")
    verdict = "PASS" if result["round_trip_ok"] else "FAIL"
    lines.append(
        f"   generated LSBs vs HDF5 physical units : {verdict} "
        f"(max abs error {result['round_trip_max_abs_error']:g})"
    )

    lines.append("")
    lines.append(" QC THRESHOLDS")
    worst_drop = max(
        (qc["dropped_frame_rate"] for qc in report["per_imu"].values()), default=0.0
    )
    non_monotonic = [
        name for name, qc in report["per_imu"].items() if not qc["timestamp_monotonic"]
    ]
    checks = [
        (
            "crc_failure_rate",
            f"{raw['crc_failure_rate']:.4%}",
            f"max {MAX_CRC_FAILURE_RATE:.4%}",
            raw["crc_failure_rate"] <= MAX_CRC_FAILURE_RATE,
        ),
        (
            "dropped_frame_rate",
            f"{worst_drop:.4%}",
            f"max {MAX_DROPPED_FRAME_RATE:.4%}",
            worst_drop <= MAX_DROPPED_FRAME_RATE,
        ),
        (
            "timestamp_monotonic",
            "OK" if not non_monotonic else "FAILED " + ",".join(non_monotonic),
            "no violations",
            not non_monotonic,
        ),
    ]
    for name, value, limit, passed in checks:
        lines.append(
            f"   {name:<20}: {value:<24} ({limit:<18}) "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if report["thresholds_violated"]:
        lines.append("")
        lines.append(" VIOLATIONS")
        for violation in report["thresholds_violated"]:
            lines.append(f"   [{violation['metric']}] {violation['message']}")

    lines.append("")
    lines.append(" OUTPUT")
    lines.append(f"   hdf5   : {result['hdf5_path']}")
    lines.append(f"   qc json: {result['qc_path']}")
    if not result["artifacts_kept"]:
        lines.append("   (artifacts removed — pass --keep-artifacts to inspect them)")

    lines.append("")
    overall = "PASSED" if report["passed"] and result["round_trip_ok"] else "FAILED"
    lines.append(f" RESULT: {overall}   (elapsed {result['elapsed_s']:.2f} s)")
    lines.append(BAR)
    return "\n".join(lines)


def run_pipeline(
    n_attempts: int = 5,
    seed: int = 42,
    rest_seconds: float = 20.0,
    keep_artifacts: bool = False,
    out_dir: Path | str | None = None,
    quiet: bool = True,
) -> dict[str, Any]:
    """Run the full synthetic pipeline and return everything it produced.

    Returns a dict with the QC report, output paths, round-trip verdict, and
    the generated attempt table. Artifacts are deleted afterwards unless
    `keep_artifacts` is set — the round-trip check runs before cleanup.
    """
    started = time.perf_counter()

    workdir = Path(out_dir) if out_dir is not None else Path(
        tempfile.mkdtemp(prefix="apex_pipeline_")
    )
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Generate the synthetic capture.
    synth = synthetic_imu.synthesize_session(
        n_attempts=n_attempts, seed=seed, rest_seconds=rest_seconds
    )

    # 2. Write it out exactly as an SD card would present it.
    bin_path = workdir / synth.bin_filename
    bin_path.write_bytes(synth.bin_bytes)
    yaml_path = workdir / "session.yaml"
    yaml_path.write_text(yaml.safe_dump(synth.metadata, sort_keys=False))

    # 3-5. Feed it through the real pipeline.
    session = load_session_metadata(yaml_path)
    per_imu, stats = parse_bin_file(bin_path)

    out_subdir = workdir / "out"
    out_subdir.mkdir(parents=True, exist_ok=True)
    session_id = session.session.session_id
    hdf5_path = out_subdir / f"{session_id}.hdf5"
    attempts = session.attempts_for_file(bin_path.name)
    write_hdf5(hdf5_path, session, per_imu, bin_path, attempts)

    report = build_qc_report(bin_path, session, per_imu, stats, attempts)
    qc_path = out_subdir / f"{session_id}_qc.json"
    qc_path.write_text(json.dumps(report, indent=2, default=str))

    # 6. Verify nothing was lost or mangled on the way through.
    round_trip_ok, max_error = verify_round_trip(hdf5_path, synth)

    result: dict[str, Any] = {
        "seed": seed,
        "n_attempts": n_attempts,
        "workdir": str(workdir),
        "bin_path": str(bin_path),
        "yaml_path": str(yaml_path),
        "hdf5_path": str(hdf5_path),
        "qc_path": str(qc_path),
        "qc_report": report,
        "attempts": synth.attempts,
        "frames_generated": synth.n_frames,
        "frames_parsed": int(stats["total_frames"]),
        "bin_bytes": len(synth.bin_bytes),
        "duration_s": synth.duration_us / 1e6,
        "n_clipped": synth.n_clipped,
        "round_trip_ok": round_trip_ok,
        "round_trip_max_abs_error": max_error,
        "artifacts_kept": keep_artifacts,
        "elapsed_s": time.perf_counter() - started,
    }
    result["summary"] = _format_summary(result)

    if not quiet:
        print(result["summary"])

    if not keep_artifacts:
        shutil.rmtree(workdir, ignore_errors=True)

    return result


def main() -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Run the APEX ingestion pipeline end to end on synthetic data."
    )
    parser.add_argument(
        "--n-attempts", type=int, default=5, help="Mock attempts to generate (default 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed; same seed = identical output"
    )
    parser.add_argument(
        "--rest-seconds",
        type=float,
        default=20.0,
        help="Rest period between attempts (default 20)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Exit 1 if any QC threshold is violated"
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the generated .BIN/YAML/HDF5 instead of cleaning up",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Working directory (default: a temp dir)",
    )
    args = parser.parse_args()

    try:
        result = run_pipeline(
            n_attempts=args.n_attempts,
            seed=args.seed,
            rest_seconds=args.rest_seconds,
            keep_artifacts=args.keep_artifacts,
            out_dir=args.out_dir,
            quiet=False,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not result["round_trip_ok"]:
        print(
            "[ERROR] Round-trip verification failed: data changed between the "
            ".BIN and the HDF5.",
            file=sys.stderr,
        )
        return 1
    if args.strict and result["qc_report"]["thresholds_violated"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
