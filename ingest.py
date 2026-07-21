"""APEX binary log ingestion: parse a raw .BIN file, validate frames, convert
to physical units, run QC, and write a structured HDF5 file plus a QC report.

Usage:
    python ingest.py --bin path/to/APEX0001.BIN \
                     --meta path/to/session.yaml \
                     --out path/to/output/ \
                     [--qc-report path/to/qc.json] [--strict]

Exit codes:
    0  success
    1  QC threshold violation while --strict was set
    2  unrecoverable error (missing file, metadata schema failure)

Outputs, written into --out:
    <session_id>.hdf5      root-level /imu_N/ streams + per-attempt windows
    <session_id>_qc.json   QC report (see build_qc_report)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import struct
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from metadata_models import Attempt, SessionDocument, load_session

ACCEL_SCALE_G: float = 0.000122
GYRO_SCALE_DPS: float = 0.035
CRC8_POLY: int = 0x07
CRC8_INIT: int = 0x00
CRC8_INPUT_BYTES: int = 18
FRAME_SIZE: int = 32
SYNC_BYTE: int = 0xA5
EXPECTED_ODR_HZ: int = 104
EXPECTED_DELTA_US: float = 1_000_000.0 / EXPECTED_ODR_HZ
DROP_THRESHOLD_US: float = 2.0 * EXPECTED_DELTA_US
NUM_IMUS: int = 5

# Frame format history. v1 captures (Data/firstDataset.BIN, Data/APEX0004.BIN)
# are 34 bytes; v2 is the current 32. This parser reads v2, and survives v1
# only because it resyncs byte-by-byte — see the alignment note in the QC report.
FRAME_SIZE_BY_VERSION: dict[int, int] = {1: 34, 2: 32}

INGEST_TOOL_VERSION: str = "apex_ingest 2.0"

# QC thresholds. A run with --strict exits 1 if any of these is violated.
MAX_CRC_FAILURE_RATE: float = 0.001  # 0.1% of candidate frames, file-scoped
MAX_DROPPED_FRAME_RATE: float = 0.01  # 1% of expected samples, per IMU
# Monotonicity has no tolerance: a single backwards timestamp means either a
# parser bug or a TIM5 wraparound, and both invalidate every window boundary.

# Frame layout (little-endian): sync u8, ts u32, idx u8, gyro xyz i16, accel xyz i16, crc u8
# = 1 + 4 + 1 + 6 + 6 + 1 = 19 bytes. Remaining 13 bytes are reserved (ignored).
_FRAME_STRUCT = struct.Struct("<BIBhhhhhhB")


def crc8(data: bytes) -> int:
    """Compute CRC-8/CCITT (polynomial 0x07, initial value 0x00) over data."""
    crc = CRC8_INIT
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def encode_frame(
    timestamp_us: int,
    imu_index: int,
    gyro: tuple[int, int, int],
    accel: tuple[int, int, int],
) -> bytes:
    """Build one 32-byte frame — the exact inverse of the parser's struct.

    This exists so synthetic fixtures and the round-trip tests cannot drift
    away from the layout `parse_bin_file` expects: both sides read the frame
    definition from `_FRAME_STRUCT` and `CRC8_INPUT_BYTES` here.
    """
    body = struct.pack(
        "<BIBhhhhhh", SYNC_BYTE, timestamp_us & 0xFFFFFFFF, imu_index, *gyro, *accel
    )
    assert len(body) == CRC8_INPUT_BYTES
    return body + bytes([crc8(body)]) + bytes(FRAME_SIZE - CRC8_INPUT_BYTES - 1)


def parse_bin_file(bin_path: Path) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """Scan a raw APEX .BIN file and return per-IMU frame data and parse stats.

    Walks the file searching for the sync byte, validates each candidate frame
    by CRC, and routes valid frames to per-IMU buffers. On CRC mismatch advances
    by a single byte to avoid missing a real frame after a corrupted region.
    """
    data = bin_path.read_bytes()
    n = len(data)

    ts_buf: list[list[int]] = [[] for _ in range(NUM_IMUS)]
    gyro_buf: list[list[tuple[int, int, int]]] = [[] for _ in range(NUM_IMUS)]
    accel_buf: list[list[tuple[int, int, int]]] = [[] for _ in range(NUM_IMUS)]

    crc_fail_count = 0
    bad_index_count = 0
    total_frames = 0

    offset = 0
    while offset + FRAME_SIZE <= n:
        if data[offset] != SYNC_BYTE:
            offset += 1
            continue

        computed = crc8(data[offset : offset + CRC8_INPUT_BYTES])
        stored = data[offset + CRC8_INPUT_BYTES]
        if computed != stored:
            crc_fail_count += 1
            print(
                f"[CRC fail] offset=0x{offset:08X} "
                f"computed=0x{computed:02X} stored=0x{stored:02X}"
            )
            offset += 1
            continue

        sync, ts, idx, gx, gy, gz, ax, ay, az, _crc = _FRAME_STRUCT.unpack_from(
            data, offset
        )

        if idx >= NUM_IMUS:
            bad_index_count += 1
            print(f"[Bad IMU index] offset=0x{offset:08X} idx={idx}")
            offset += FRAME_SIZE
            continue

        ts_buf[idx].append(ts)
        gyro_buf[idx].append((gx, gy, gz))
        accel_buf[idx].append((ax, ay, az))
        total_frames += 1
        offset += FRAME_SIZE

    per_imu: list[dict[str, np.ndarray]] = []
    for i in range(NUM_IMUS):
        per_imu.append(
            {
                "timestamps_us": np.asarray(ts_buf[i], dtype=np.uint64),
                "gyro_raw": np.asarray(gyro_buf[i], dtype=np.int16).reshape(-1, 3),
                "accel_raw": np.asarray(accel_buf[i], dtype=np.int16).reshape(-1, 3),
            }
        )

    stats: dict[str, Any] = {
        "total_frames": total_frames,
        "crc_fail_count": crc_fail_count,
        "bad_index_count": bad_index_count,
        "per_imu_counts": [len(ts_buf[i]) for i in range(NUM_IMUS)],
    }
    assert stats["total_frames"] == sum(stats["per_imu_counts"])
    return per_imu, stats


def compute_sync_stats(raw: bytes) -> dict[str, Any]:
    """Sync-byte hit rate over frame-aligned offsets.

    Computed independently of the parser so it stays a pure measurement of the
    file rather than a side effect of the recovery walk. A healthy capture has
    a 0xA5 at every multiple of FRAME_SIZE; anything less means the writer
    lost frame alignment (partial sector write, truncated file, SD hiccup).
    Counting only aligned offsets is deliberate — a 0xA5 elsewhere in the file
    is almost always ordinary payload data, not a frame boundary.
    """
    frame_slots = len(raw) // FRAME_SIZE
    if frame_slots == 0:
        return {
            "byte_count": len(raw),
            "frame_slots": 0,
            "sync_hits": 0,
            "sync_misses": 0,
            "sync_hit_rate": 0.0,
            "trailing_bytes": len(raw),
        }

    aligned = np.frombuffer(raw[: frame_slots * FRAME_SIZE], dtype=np.uint8).reshape(
        frame_slots, FRAME_SIZE
    )[:, 0]
    hits = int(np.count_nonzero(aligned == SYNC_BYTE))
    return {
        "byte_count": len(raw),
        "frame_slots": frame_slots,
        "sync_hits": hits,
        "sync_misses": frame_slots - hits,
        "sync_hit_rate": hits / frame_slots,
        "trailing_bytes": len(raw) - frame_slots * FRAME_SIZE,
    }


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 of a raw capture, so an HDF5 can be traced back to its .BIN."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_monotonic(timestamps_us: np.ndarray, imu_idx: int) -> None:
    """Warn if a per-IMU timestamp stream contains non-monotonic samples."""
    if timestamps_us.size < 2:
        return
    deltas = np.diff(timestamps_us.astype(np.int64))
    bad = int(np.sum(deltas < 0))
    if bad:
        print(f"[WARN] imu_{imu_idx}: {bad} non-monotonic timestamp(s)")


def compute_qc(timestamps_us: np.ndarray) -> dict[str, Any]:
    """Compute QC metrics for a single IMU's timestamp stream."""
    n = int(timestamps_us.size)
    if n < 2:
        return {
            "total_frames": n,
            "duration_sec": 0.0,
            "mean_delta_us": 0.0,
            "max_gap_us": 0.0,
            "dropped_frames": 0,
            "drop_rate_pct": 0.0,
            "estimated_dropped_frames": 0,
            "dropped_frame_rate": 0.0,
            "observed_rate_hz": 0.0,
            "timestamp_monotonic": True,
            "monotonicity_violations": 0,
            "first_non_monotonic_index": None,
        }
    deltas = np.diff(timestamps_us.astype(np.int64))
    duration_sec = float(int(timestamps_us[-1]) - int(timestamps_us[0])) / 1e6
    mean_delta = float(np.mean(deltas))
    max_gap = float(np.max(deltas))
    dropped = int(np.sum(deltas > DROP_THRESHOLD_US))
    drop_rate = (dropped / n) * 100.0

    # Monotonicity. A negative delta is never legitimate in a single stream.
    negative = np.flatnonzero(deltas < 0)
    violations = int(negative.size)
    first_violation = int(negative[0]) if violations else None

    # Estimated dropped frames: each forward gap of k nominal periods means
    # k-1 samples never made it to the card. Sum over gaps rather than just
    # counting them, so one 500 ms stall is not scored the same as one missed
    # sample. Negative deltas are excluded — they are a monotonicity problem,
    # not a drop, and counting them here would double-report the same defect.
    forward = deltas[deltas > 0]
    per_gap_drops = np.maximum(np.round(forward / EXPECTED_DELTA_US) - 1.0, 0.0)
    estimated_dropped = int(per_gap_drops.sum())
    expected_total = n + estimated_dropped
    dropped_frame_rate = estimated_dropped / expected_total if expected_total else 0.0
    observed_rate_hz = (n - 1) / duration_sec if duration_sec > 0 else 0.0

    return {
        "total_frames": n,
        "duration_sec": duration_sec,
        "mean_delta_us": mean_delta,
        "max_gap_us": max_gap,
        "dropped_frames": dropped,
        "drop_rate_pct": drop_rate,
        "estimated_dropped_frames": estimated_dropped,
        "dropped_frame_rate": dropped_frame_rate,
        "observed_rate_hz": observed_rate_hz,
        "timestamp_monotonic": violations == 0,
        "monotonicity_violations": violations,
        "first_non_monotonic_index": first_violation,
    }


def window_indices(
    timestamps_us: np.ndarray, start_us: int, end_us: int
) -> tuple[int, int]:
    """Half-open [start_us, end_us) slice bounds for a sorted timestamp array.

    Both bounds use np.searchsorted's default side='left', so a sample landing
    exactly on start_us is included and one landing exactly on end_us is not.

    WRAPAROUND-SENSITIVE — DO NOT "FIX" BY SORTING. searchsorted requires the
    array to be sorted, and TIM5 is a 32-bit microsecond counter that wraps
    every ~71.6 minutes. A session long enough to wrap produces a timestamp
    stream that steps backwards at the wrap point, which silently returns
    garbage bounds here. That is why the QC report treats any non-monotonic
    timestamp as a hard, no-tolerance failure: it is the wraparound alarm.
    Sorting the array to make this "work" would hide the wrap and quietly
    interleave two different hours of data into one attempt window. If you
    ever need to support sessions past 71.6 minutes, unwrap the counter in
    the parser first and widen the field to uint64 — do not touch this line.
    """
    if timestamps_us.size == 0:
        return 0, 0
    lo, hi = np.searchsorted(timestamps_us, np.asarray([start_us, end_us]))
    return int(lo), int(hi)


# ─────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────


def load_session_metadata(meta_path: Path) -> SessionDocument:
    """Load and validate the session metadata YAML sidecar.

    This is the live metadata path. Schema and all cross-reference checks live
    in metadata_models.SessionDocument — the single source of truth.
    """
    return load_session(meta_path)


# ---- LEGACY: flat-JSON metadata path -------------------------------------
# Superseded by load_session_metadata() above, which validates a YAML sidecar
# against the Pydantic schema in metadata_models.py. Retained, not deleted,
# because sessions captured before the YAML schema existed still have flat
# .json sidecars on disk; if one of those ever needs re-ingesting, this is the
# reader for it. Nothing calls these three functions today. They intentionally
# do NOT understand attempts windowing, per-attempt labels, or sensor layout
# overrides — the flat JSON format has no way to express any of that.


def load_metadata(meta_path: Path) -> dict[str, Any]:
    """Load and minimally validate the session metadata JSON sidecar."""
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata JSON not found at: {meta_path}. "
            f"Expected sidecar file alongside the .BIN."
        )
    with meta_path.open("r") as f:
        meta = json.load(f)

    required = ["session_id", "date", "climber_id", "location", "imu_placement", "notes"]
    for key in required:
        if key not in meta:
            raise ValueError(f"Metadata missing required key: '{key}' in {meta_path}")

    placement = meta["imu_placement"]
    if not isinstance(placement, list) or len(placement) != NUM_IMUS:
        raise ValueError(
            f"'imu_placement' must be a list of {NUM_IMUS} strings, got {placement!r}"
        )

    meta.setdefault("attempts", [])
    return meta


def attempts_dtype() -> np.dtype:
    """Structured numpy dtype for the attempts/labels dataset."""
    return np.dtype(
        [
            ("start_us", "u8"),
            ("end_us", "u8"),
            ("setter_grade", h5py.string_dtype()),
            ("perceived_grade", h5py.string_dtype()),
            ("subjective_difficulty", "f4"),
            ("outcome", h5py.string_dtype()),
        ]
    )


def build_attempts_array(attempts: list[dict[str, Any]]) -> np.ndarray:
    """Convert a list of attempt dicts into a structured numpy array."""
    arr = np.zeros(len(attempts), dtype=attempts_dtype())
    for j, a in enumerate(attempts):
        arr[j] = (
            int(a.get("start_us", 0)),
            int(a.get("end_us", 0)),
            str(a.get("setter_grade", "")),
            str(a.get("perceived_grade", "")),
            float(a.get("subjective_difficulty", 0.0)),
            str(a.get("outcome", "")),
        )
    return arr


# ---- END LEGACY -----------------------------------------------------------


# ─────────────────────────────────────────────────────────────────────────
# QC report
# ─────────────────────────────────────────────────────────────────────────


def build_qc_report(
    bin_path: Path,
    session: SessionDocument,
    per_imu: list[dict[str, np.ndarray]],
    stats: dict[str, Any],
    attempts: list[Attempt] | None = None,
) -> dict[str, Any]:
    """Build the QC report dict for one .BIN file.

    Pure: reads the raw file for its hash and sync stats, but writes nothing
    and does not depend on the HDF5 having been produced.

    CRC failure rate is file-scoped, not per-IMU, on purpose: a frame that
    fails CRC has an untrustworthy imu_index byte, so attributing it to a
    specific sensor would be inventing information.
    """
    raw = bin_path.read_bytes()
    sync = compute_sync_stats(raw)

    frames_parsed = int(stats["total_frames"])
    crc_failures = int(stats["crc_fail_count"])
    candidates = frames_parsed + crc_failures
    crc_failure_rate = (crc_failures / candidates) if candidates else 0.0

    # Frame-size sanity. The parser recovers 34-byte (v1) frames fine — it
    # resyncs byte-by-byte after each missed sync — so a v1 file reads as
    # "0 CRC failures, 6% sync hit rate", which looks like catastrophic
    # corruption but is really just a format-version mismatch. Say so.
    observed_bytes_per_frame = (
        sync["byte_count"] / frames_parsed if frames_parsed else None
    )
    declared_version = session.session.firmware.frame_format_version
    alignment_note: str | None = None
    if (
        observed_bytes_per_frame is not None
        and abs(observed_bytes_per_frame - FRAME_SIZE) > 0.01
    ):
        alignment_note = (
            f"{observed_bytes_per_frame:.2f} bytes per recovered frame, but this "
            f"parser is built for {FRAME_SIZE}-byte frames. The metadata declares "
            f"frame_format_version={declared_version} "
            f"({FRAME_SIZE_BY_VERSION.get(declared_version, '?')}-byte). "
            f"Frames still parsed via byte-wise resync, so the data is intact, "
            f"but sync_hit_rate is meaningless for this file."
        )

    layout = session.sensor_layout.as_index_map()

    per_imu_qc: dict[str, Any] = {}
    for i in range(NUM_IMUS):
        qc = compute_qc(per_imu[i]["timestamps_us"])
        position = layout.get(i)
        per_imu_qc[f"imu_{i}"] = {
            "body_location": position.value if position else None,
            "frame_count": qc["total_frames"],
            "duration_s": qc["duration_sec"],
            "mean_delta_us": qc["mean_delta_us"],
            "max_gap_us": qc["max_gap_us"],
            "observed_rate_hz": qc["observed_rate_hz"],
            "estimated_dropped_frames": qc["estimated_dropped_frames"],
            "dropped_frame_rate": qc["dropped_frame_rate"],
            "timestamp_monotonic": qc["timestamp_monotonic"],
            "monotonicity_violations": qc["monotonicity_violations"],
            "first_non_monotonic_index": qc["first_non_monotonic_index"],
        }

    if attempts is None:
        attempts = session.attempts_for_file(bin_path.name)

    nominal_hz = session.session.firmware.sample_rate_hz
    per_attempt: list[dict[str, Any]] = []
    for attempt in attempts:
        attempt_imu: dict[str, Any] = {}
        resolved = session.resolved_layout(attempt)
        for i in range(NUM_IMUS):
            ts = per_imu[i]["timestamps_us"]
            lo, hi = window_indices(
                ts, attempt.timestamp_us_start, attempt.timestamp_us_end
            )
            window_qc = compute_qc(ts[lo:hi])
            position = resolved.get(i)
            attempt_imu[f"imu_{i}"] = {
                "body_location": position.value if position else None,
                "frame_count": int(hi - lo),
                "expected_frame_count": int(round(attempt.duration_s * nominal_hz)),
                "estimated_dropped_frames": window_qc["estimated_dropped_frames"],
                "dropped_frame_rate": window_qc["dropped_frame_rate"],
                "max_gap_us": window_qc["max_gap_us"],
                "observed_rate_hz": window_qc["observed_rate_hz"],
                "timestamp_monotonic": window_qc["timestamp_monotonic"],
            }
        per_attempt.append(
            {
                "attempt_id": attempt.attempt_id,
                "kind": attempt.kind.value,
                "climber_id": attempt.climber_id,
                "route_id": attempt.route_id,
                "timestamp_us_start": attempt.timestamp_us_start,
                "timestamp_us_end": attempt.timestamp_us_end,
                "duration_s": attempt.duration_s,
                "per_imu": attempt_imu,
            }
        )

    report: dict[str, Any] = {
        "session_id": session.session.session_id,
        "ingest_timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "ingest_tool_version": INGEST_TOOL_VERSION,
        "raw_file": {
            "filename": bin_path.name,
            "sha256": sha256_file(bin_path),
            "byte_count": sync["byte_count"],
            "frame_slots": sync["frame_slots"],
            "sync_hits": sync["sync_hits"],
            "sync_misses": sync["sync_misses"],
            "sync_hit_rate": sync["sync_hit_rate"],
            "trailing_bytes": sync["trailing_bytes"],
            "frames_parsed": frames_parsed,
            "crc_failures": crc_failures,
            "crc_failure_rate": crc_failure_rate,
            "bad_index_discards": int(stats["bad_index_count"]),
            "bytes_per_frame_observed": observed_bytes_per_frame,
            "declared_frame_format_version": declared_version,
            "declared_frame_size_bytes": FRAME_SIZE_BY_VERSION.get(declared_version),
            "frame_alignment_note": alignment_note,
        },
        "per_imu": per_imu_qc,
        "per_attempt": per_attempt,
        "thresholds": {
            "max_crc_failure_rate": MAX_CRC_FAILURE_RATE,
            "max_dropped_frame_rate": MAX_DROPPED_FRAME_RATE,
            "require_timestamp_monotonic": True,
        },
    }
    report["thresholds_violated"] = evaluate_thresholds(report)
    report["passed"] = not report["thresholds_violated"]
    return report


def evaluate_thresholds(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Check the QC report against the three --strict thresholds."""
    violations: list[dict[str, Any]] = []

    crc_rate = report["raw_file"]["crc_failure_rate"]
    if crc_rate > MAX_CRC_FAILURE_RATE:
        violations.append(
            {
                "metric": "crc_failure_rate",
                "scope": f"file:{report['raw_file']['filename']}",
                "value": crc_rate,
                "threshold": MAX_CRC_FAILURE_RATE,
                "message": (
                    f"CRC failure rate {crc_rate:.4%} exceeds "
                    f"{MAX_CRC_FAILURE_RATE:.4%}"
                ),
            }
        )

    for name, qc in report["per_imu"].items():
        if not qc["timestamp_monotonic"]:
            violations.append(
                {
                    "metric": "timestamp_monotonic",
                    "scope": name,
                    "value": qc["monotonicity_violations"],
                    "threshold": 0,
                    "message": (
                        f"{name}: {qc['monotonicity_violations']} non-monotonic "
                        f"timestamp(s), first at sample index "
                        f"{qc['first_non_monotonic_index']}"
                    ),
                }
            )
        drop_rate = qc["dropped_frame_rate"]
        if drop_rate > MAX_DROPPED_FRAME_RATE:
            violations.append(
                {
                    "metric": "dropped_frame_rate",
                    "scope": name,
                    "value": drop_rate,
                    "threshold": MAX_DROPPED_FRAME_RATE,
                    "message": (
                        f"{name}: dropped-frame rate {drop_rate:.4%} exceeds "
                        f"{MAX_DROPPED_FRAME_RATE:.4%} "
                        f"({qc['estimated_dropped_frames']} estimated drops)"
                    ),
                }
            )
    return violations


# ─────────────────────────────────────────────────────────────────────────
# HDF5 writing
# ─────────────────────────────────────────────────────────────────────────


def assert_h5_safe(name: str, what: str) -> str:
    """Guard any string about to become an HDF5 group or attribute name.

    metadata_models already rejects '/' in IDs at the schema boundary, so this
    should be unreachable — it is here because a silent group-nesting bug (an
    id like "cave/blue" quietly creating /attempts/cave/blue) is far more
    expensive to find later than a loud assertion is now.
    """
    if "/" in name or not name or name != name.strip():
        raise ValueError(
            f"{what} {name!r} is not a valid HDF5 name (no '/', no empty, "
            f"no leading/trailing whitespace)"
        )
    return name


def _h5_value(value: Any) -> Any:
    """Coerce a metadata value into something h5py can store as an attribute."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    if isinstance(value, bool):
        return np.bool_(value)
    return value


def _set_attrs(target: Any, values: dict[str, Any]) -> None:
    """Write attrs, skipping Nones (HDF5 has no null attribute type)."""
    for key, value in values.items():
        if value is None:
            continue
        target.attrs[key] = _h5_value(value)


def _create_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """Create a dataset, compressing only when there is something to compress.

    h5py cannot chunk (and therefore cannot compress) a zero-length dataset,
    and an IMU that produced no frames in an attempt window is a normal, if
    unwelcome, outcome.
    """
    if data.size:
        group.create_dataset(name, data=data, compression="gzip", compression_opts=4)
    else:
        group.create_dataset(name, data=data)


def write_hdf5(
    out_path: Path,
    session: SessionDocument,
    per_imu: list[dict[str, np.ndarray]],
    bin_path: Path,
    attempts: list[Attempt] | None = None,
) -> list[dict[str, Any]]:
    """Write session metadata, full per-IMU streams, and per-attempt windows.

    Layout:
        /                       session attrs
        /imu_N/                 the complete stream for sensor N (all attempts,
                                plus rest periods) — kept for debugging and for
                                re-windowing without re-parsing the .BIN
        /attempts/<id>/imu_N/   that attempt's slice of the same stream
    """
    if attempts is None:
        attempts = session.attempts_for_file(bin_path.name)

    info = session.session
    layout = session.sensor_layout.as_index_map()

    qc_per_imu: list[dict[str, Any]] = []
    with h5py.File(out_path, "w") as h5:
        _set_attrs(
            h5,
            {
                "schema_version": session.schema_version,
                "session_id": assert_h5_safe(info.session_id, "session_id"),
                "date": info.date,
                "start_time_local": info.start_time_local,
                "timezone": info.timezone,
                "collected_by": info.collected_by,
                "notes": info.notes,
                "gym_name": info.gym.name,
                "gym_code": info.gym.code,
                "firmware_commit": info.firmware.commit,
                "frame_format_version": info.firmware.frame_format_version,
                "sample_rate_hz": info.firmware.sample_rate_hz,
                "source_bin": bin_path.name,
                "source_bin_sha256": sha256_file(bin_path),
                "ingest_timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "ingest_tool_version": INGEST_TOOL_VERSION,
            },
        )
        h5.attrs.create(
            "imu_placement",
            [layout[i].value for i in sorted(layout)],
            dtype=h5py.string_dtype(),
        )

        # ── full per-IMU streams ─────────────────────────────────────────
        for i in range(NUM_IMUS):
            grp = h5.create_group(f"imu_{i}")
            ts = per_imu[i]["timestamps_us"]
            gyro_raw = per_imu[i]["gyro_raw"]
            accel_raw = per_imu[i]["accel_raw"]

            check_monotonic(ts, i)

            accel_g = accel_raw.astype(np.float32) * np.float32(ACCEL_SCALE_G)
            gyro_dps = gyro_raw.astype(np.float32) * np.float32(GYRO_SCALE_DPS)

            _create_dataset(grp, "timestamps_us", ts.astype(np.uint64))
            _create_dataset(grp, "accel_g", accel_g)
            _create_dataset(grp, "gyro_dps", gyro_dps)

            position = layout.get(i)
            if position is not None:
                grp.attrs["body_location"] = position.value

            qc = compute_qc(ts)
            for k, v in qc.items():
                if v is not None:
                    grp.attrs[k] = v
            qc_per_imu.append(qc)

        # ── per-attempt windows ──────────────────────────────────────────
        att_root = h5.create_group("attempts")
        for attempt in attempts:
            aid = assert_h5_safe(attempt.attempt_id, "attempt_id")
            att_grp = att_root.create_group(aid)
            _set_attrs(
                att_grp,
                {
                    "attempt_id": attempt.attempt_id,
                    "kind": attempt.kind,
                    "climber_id": attempt.climber_id,
                    "route_id": (
                        assert_h5_safe(attempt.route_id, "route_id")
                        if attempt.route_id
                        else None
                    ),
                    "attempt_number": attempt.attempt_number,
                    "timestamp_us_start": attempt.timestamp_us_start,
                    "timestamp_us_end": attempt.timestamp_us_end,
                    "duration_s": attempt.duration_s,
                    "posted_grade": attempt.posted_grade,
                    "perceived_grade": attempt.perceived_grade,
                    "posted_grade_int": attempt.posted_grade_int,
                    "perceived_grade_int": attempt.perceived_grade_int,
                    "subjective_difficulty": attempt.subjective_difficulty,
                    "difficulty_int": attempt.difficulty_int,
                    "outcome": attempt.outcome,
                    "fall_move_number": attempt.fall_move_number,
                    "fatigue_pre": attempt.fatigue_pre,
                    "fatigue_post": attempt.fatigue_post,
                    "rest_seconds_before": attempt.rest_seconds_before,
                    "notes": attempt.notes,
                },
            )
            if attempt.route_id is not None:
                route = session.route_by_id(attempt.route_id)
                _set_attrs(
                    att_grp,
                    {
                        "route_setter": route.setter,
                        "route_gym_section": route.gym_section,
                        "route_wall_angle_deg": route.wall_angle_deg,
                        "route_posted_grade": route.posted_grade,
                        "route_posted_grade_int": route.posted_grade_int,
                    },
                )

            resolved = session.resolved_layout(attempt)
            for i in range(NUM_IMUS):
                ts = per_imu[i]["timestamps_us"]
                lo, hi = window_indices(
                    ts, attempt.timestamp_us_start, attempt.timestamp_us_end
                )

                imu_grp = att_grp.create_group(f"imu_{i}")
                accel_g = per_imu[i]["accel_raw"][lo:hi].astype(np.float32) * np.float32(
                    ACCEL_SCALE_G
                )
                gyro_dps = per_imu[i]["gyro_raw"][lo:hi].astype(np.float32) * np.float32(
                    GYRO_SCALE_DPS
                )
                _create_dataset(imu_grp, "timestamps_us", ts[lo:hi].astype(np.uint64))
                _create_dataset(imu_grp, "gyro_dps", gyro_dps)
                _create_dataset(imu_grp, "accel_g", accel_g)

                position = resolved.get(i)
                _set_attrs(
                    imu_grp,
                    {
                        "body_location": position.value if position else None,
                        "window_start_index": lo,
                        "window_end_index": hi,
                        "frame_count": hi - lo,
                    },
                )

    return qc_per_imu


# ─────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────


def print_summary(
    bin_path: Path,
    out_path: Path,
    stats: dict[str, Any],
    qc_per_imu: list[dict[str, Any]],
) -> None:
    """Print a human-readable QC summary table to stdout."""
    print()
    print("=" * 78)
    print(f" APEX Ingest Summary    {bin_path.name}  ->  {out_path.name}")
    print("=" * 78)
    print(f" Total valid frames     : {stats['total_frames']}")
    print(f" CRC failures           : {stats['crc_fail_count']}")
    print(f" Bad IMU index discards : {stats['bad_index_count']}")
    print(
        f" Per-IMU sum            : {sum(stats['per_imu_counts'])}  "
        f"(matches total: {stats['total_frames'] == sum(stats['per_imu_counts'])})"
    )
    print()
    header = (
        f" {'IMU':<4} {'Frames':>8} {'Dur[s]':>10} "
        f"{'MeanDt[us]':>12} {'MaxGap[us]':>12} {'Drops':>7} {'Rate[%]':>9}"
    )
    print(header)
    print(" " + "-" * (len(header) - 1))
    for i, qc in enumerate(qc_per_imu):
        print(
            f" {i:<4} {qc['total_frames']:>8} {qc['duration_sec']:>10.3f} "
            f"{qc['mean_delta_us']:>12.2f} {qc['max_gap_us']:>12.0f} "
            f"{qc['estimated_dropped_frames']:>7} "
            f"{qc['dropped_frame_rate'] * 100.0:>9.4f}"
        )
    print("=" * 78)

    for i, qc in enumerate(qc_per_imu):
        if qc["total_frames"] == 0:
            print(f" [QC] imu_{i}: empty stream (no valid frames received)")


def print_qc_verdict(report: dict[str, Any], strict: bool) -> None:
    """Print the threshold verdict, naming every threshold that failed."""
    print()
    print(" QC thresholds")
    print(
        f"   crc_failure_rate      : {report['raw_file']['crc_failure_rate']:.4%} "
        f"(max {MAX_CRC_FAILURE_RATE:.4%})"
    )
    worst_drop = max(
        (qc["dropped_frame_rate"] for qc in report["per_imu"].values()), default=0.0
    )
    print(
        f"   dropped_frame_rate    : {worst_drop:.4%} worst IMU "
        f"(max {MAX_DROPPED_FRAME_RATE:.4%})"
    )
    non_monotonic = [
        name
        for name, qc in report["per_imu"].items()
        if not qc["timestamp_monotonic"]
    ]
    print(
        f"   timestamp_monotonic   : "
        f"{'OK' if not non_monotonic else 'FAILED on ' + ', '.join(non_monotonic)}"
    )

    if report["thresholds_violated"]:
        print()
        print(" QC FAILED:")
        for violation in report["thresholds_violated"]:
            print(f"   [{violation['metric']}] {violation['message']}")
        if not strict:
            print("   (not --strict, so exiting 0 anyway)")
    else:
        print()
        print(" QC PASSED: all thresholds within limits.")


def main() -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="APEX binary log -> HDF5 ingestion.")
    parser.add_argument("--bin", required=True, type=Path, help="Path to APEX .BIN file")
    parser.add_argument(
        "--meta", required=True, type=Path, help="Path to session metadata YAML"
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output directory for the HDF5 file"
    )
    parser.add_argument(
        "--qc-report",
        type=Path,
        default=None,
        help="QC report path (default: <out>/<session_id>_qc.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any QC threshold is violated",
    )
    args = parser.parse_args()

    if not args.bin.exists():
        print(f"[ERROR] Binary file not found: {args.bin}", file=sys.stderr)
        return 2

    try:
        session = load_session_metadata(args.meta)
    except Exception as exc:  # schema failure, missing file, malformed YAML
        print(f"[ERROR] Session metadata failed validation:\n{exc}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    attempts = session.attempts_for_file(args.bin.name)
    if not attempts:
        declared = [rf.filename for rf in session.session.raw_files]
        print(
            f"[WARN] No attempts in {args.meta.name} reference "
            f"'{args.bin.name}'. Declared raw_files: {declared}. "
            f"Writing root-level /imu_N streams with no attempt windows."
        )

    print(f"Parsing {args.bin} ({args.bin.stat().st_size} bytes)...")
    per_imu, stats = parse_bin_file(args.bin)

    session_id = session.session.session_id
    out_file = args.out / f"{session_id}.hdf5"
    print(f"Writing {out_file}...")
    qc_per_imu = write_hdf5(out_file, session, per_imu, args.bin, attempts)

    report = build_qc_report(args.bin, session, per_imu, stats, attempts)
    qc_path = args.qc_report or (args.out / f"{session_id}_qc.json")
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    with qc_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Writing {qc_path}...")

    print_summary(args.bin, out_file, stats, qc_per_imu)
    print(f" Attempt windows        : {len(attempts)}")
    print_qc_verdict(report, args.strict)

    if args.strict and report["thresholds_violated"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
