"""APEX binary log ingestion: parse a raw .BIN file, validate frames, convert
to physical units, run QC, and write a structured HDF5 file.

Usage:
    python ingest.py --bin path/to/APEX0001.BIN \
                     --meta path/to/session.json \
                     --out path/to/output/
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import h5py
import numpy as np


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
        }
    deltas = np.diff(timestamps_us.astype(np.int64))
    duration_sec = float(int(timestamps_us[-1]) - int(timestamps_us[0])) / 1e6
    mean_delta = float(np.mean(deltas))
    max_gap = float(np.max(deltas))
    dropped = int(np.sum(deltas > DROP_THRESHOLD_US))
    drop_rate = (dropped / n) * 100.0
    return {
        "total_frames": n,
        "duration_sec": duration_sec,
        "mean_delta_us": mean_delta,
        "max_gap_us": max_gap,
        "dropped_frames": dropped,
        "drop_rate_pct": drop_rate,
    }


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


def write_hdf5(
    out_path: Path,
    meta: dict[str, Any],
    per_imu: list[dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Write metadata, per-IMU datasets/QC, and attempts to an HDF5 file."""
    qc_per_imu: list[dict[str, Any]] = []
    with h5py.File(out_path, "w") as h5:
        h5.attrs["session_id"] = meta["session_id"]
        h5.attrs["date"] = meta["date"]
        h5.attrs["climber_id"] = meta["climber_id"]
        h5.attrs["location"] = meta["location"]
        h5.attrs["notes"] = meta["notes"]
        h5.attrs.create(
            "imu_placement", meta["imu_placement"], dtype=h5py.string_dtype()
        )

        for i in range(NUM_IMUS):
            grp = h5.create_group(f"imu_{i}")
            ts = per_imu[i]["timestamps_us"]
            gyro_raw = per_imu[i]["gyro_raw"]
            accel_raw = per_imu[i]["accel_raw"]

            check_monotonic(ts, i)

            accel_g = accel_raw.astype(np.float32) * np.float32(ACCEL_SCALE_G)
            gyro_dps = gyro_raw.astype(np.float32) * np.float32(GYRO_SCALE_DPS)

            grp.create_dataset("timestamps_us", data=ts, dtype=np.uint64)
            grp.create_dataset("accel_g", data=accel_g, dtype=np.float32)
            grp.create_dataset("gyro_dps", data=gyro_dps, dtype=np.float32)

            qc = compute_qc(ts)
            for k, v in qc.items():
                grp.attrs[k] = v
            qc_per_imu.append(qc)

        att_grp = h5.create_group("attempts")
        adt = attempts_dtype()
        attempts = meta.get("attempts", [])
        if attempts:
            att_grp.create_dataset(
                "labels", data=build_attempts_array(attempts), dtype=adt
            )
        else:
            att_grp.create_dataset("labels", shape=(0,), dtype=adt)

    return qc_per_imu


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
            f"{qc['dropped_frames']:>7} {qc['drop_rate_pct']:>9.4f}"
        )
    print("=" * 78)

    for i, qc in enumerate(qc_per_imu):
        if qc["total_frames"] == 0:
            print(f" [QC] imu_{i}: empty stream (no valid frames received)")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="APEX binary log -> HDF5 ingestion."
    )
    parser.add_argument("--bin", required=True, type=Path, help="Path to APEX .BIN file")
    parser.add_argument(
        "--meta", required=True, type=Path, help="Path to session metadata JSON"
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output directory for the HDF5 file"
    )
    args = parser.parse_args()

    if not args.bin.exists():
        raise FileNotFoundError(f"Binary file not found: {args.bin}")

    meta = load_metadata(args.meta)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {args.bin} ({args.bin.stat().st_size} bytes)...")
    per_imu, stats = parse_bin_file(args.bin)

    out_file = args.out / f"{meta['session_id']}.hdf5"
    print(f"Writing {out_file}...")
    qc_per_imu = write_hdf5(out_file, meta, per_imu)

    print_summary(args.bin, out_file, stats, qc_per_imu)


if __name__ == "__main__":
    main()
