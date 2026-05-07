#!/usr/bin/env python3
"""
APEX SD log validator + visualizer.

Reads a binary APEXxxxx.BIN file written by the H7 firmware, validates frame
integrity (sync byte + CRC8 + monotonic timestamps), and plots gyro/accel time
series. Use after pulling the SD card off the NUCLEO to eyeball whether the
captured data looks right.

Usage:
    python apex_log_view.py [path/to/APEX0000.BIN]
    python apex_log_view.py                       # defaults to ./APEX0000.BIN

Frame layout (must match apex_frame_t in sd_log.h):
    offset   bytes   field
    0        1       sync (0xA5)
    1        4       timestamp_us (uint32 LE)
    5        1       imu_index
    6        6       gyro[3] (int16 LE)
    12       6       accel[3] (int16 LE)
    18       1       crc8 (CCITT poly 0x07, init 0x00, over bytes 0..17)
    19       15      reserved (zero-filled)
                   = 34 bytes total
"""

import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# < = little-endian, no padding. Mirrors __attribute__((packed)) on the C side.
FRAME_FMT = "<BIBhhhhhhB15s"
FRAME_SIZE = struct.calcsize(FRAME_FMT)
assert FRAME_SIZE == 34, f"format string size mismatch: got {FRAME_SIZE}, expected 34"

# IMU scale factors from your firmware config (LSM6DSO at ±4g / ±1000 dps).
ACCEL_SCALE_G  = 0.122 / 1000.0   # 0.122 mg/LSB  → g
GYRO_SCALE_DPS = 35.0  / 1000.0   # 35.0 mdps/LSB → dps


def crc8_ccitt(data: bytes) -> int:
    """CRC-8/CCITT, poly 0x07, init 0x00, no reflect, no xorout. Matches firmware."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def parse_file(path: Path):
    """Parse the binary file into an array of (ts, idx, gx, gy, gz, ax, ay, az) rows.
    Reports sync/CRC pass rates and any timestamp regressions."""
    raw = path.read_bytes()
    n_complete = len(raw) // FRAME_SIZE
    leftover = len(raw) % FRAME_SIZE
    if leftover:
        print(f"  [warn] {leftover} trailing bytes (partial frame, ignored)")

    frames = np.empty((n_complete, 8), dtype=np.int64)
    sync_ok = 0
    crc_ok = 0
    bad_idx = []

    for i in range(n_complete):
        chunk = raw[i*FRAME_SIZE:(i+1)*FRAME_SIZE]
        sync, ts, idx, gx, gy, gz, ax, ay, az, crc, _ = struct.unpack(FRAME_FMT, chunk)
        if sync == 0xA5:
            sync_ok += 1
        else:
            bad_idx.append(i)
        if crc8_ccitt(chunk[:18]) == crc:
            crc_ok += 1
        frames[i] = (ts, idx, gx, gy, gz, ax, ay, az)

    print(f"  parsed {n_complete} frames")
    print(f"  sync 0xA5: {sync_ok}/{n_complete}  ({100*sync_ok/n_complete:.1f}%)")
    print(f"  CRC pass:  {crc_ok}/{n_complete}  ({100*crc_ok/n_complete:.1f}%)")
    if bad_idx:
        print(f"  [warn] bad sync at frame indices: {bad_idx[:10]}"
              f"{'...' if len(bad_idx) > 10 else ''}")

    # Timestamp monotonicity check
    ts = frames[:, 0]
    dts = np.diff(ts)
    if (dts <= 0).any():
        regressions = int((dts <= 0).sum())
        print(f"  [warn] {regressions} non-monotonic timestamps")
    return frames


def summarize(frames: np.ndarray):
    ts = frames[:, 0] / 1e6  # μs → s
    duration = ts[-1] - ts[0]
    rate = (len(frames) - 1) / duration if duration > 0 else 0.0

    gyro_dps   = frames[:, 2:5] * GYRO_SCALE_DPS
    accel_g    = frames[:, 5:8] * ACCEL_SCALE_G

    # Heuristic: pick the first 5 seconds to estimate "at rest" baseline,
    # since the protocol asks you to keep it still on the desk at the start.
    rest_mask = ts - ts[0] < 5.0
    rest_n    = int(rest_mask.sum())

    print(f"  duration:       {duration:.2f} s")
    print(f"  effective rate: {rate:.2f} Hz  (expect ~16 Hz at HAL_Delay(60))")
    print(f"  rate stddev:    {(1e6 / np.diff(ts*1e6).std()):.2f} (frame-period jitter, μs)")
    print()
    print(f"  --- 'at rest' baseline (first {rest_n} frames) ---")
    print(f"  accel mean (g):   X={accel_g[rest_mask,0].mean():+.3f}  "
          f"Y={accel_g[rest_mask,1].mean():+.3f}  "
          f"Z={accel_g[rest_mask,2].mean():+.3f}   (expect ~ 0, 0, ±1)")
    print(f"  gyro  mean (dps): X={gyro_dps[rest_mask,0].mean():+.2f}  "
          f"Y={gyro_dps[rest_mask,1].mean():+.2f}  "
          f"Z={gyro_dps[rest_mask,2].mean():+.2f}   (expect ~ 0, 0, 0)")
    print(f"  accel std  (g):   X={accel_g[rest_mask,0].std():.4f}  "
          f"Y={accel_g[rest_mask,1].std():.4f}  "
          f"Z={accel_g[rest_mask,2].std():.4f}   (noise floor)")


def plot(frames: np.ndarray, save_path: Path = None):
    ts0 = frames[0, 0]
    ts = (frames[:, 0] - ts0) / 1e6  # seconds since first frame
    gyro_dps = frames[:, 2:5] * GYRO_SCALE_DPS
    accel_g  = frames[:, 5:8] * ACCEL_SCALE_G

    fig, (ax_g, ax_a) = plt.subplots(2, 1, sharex=True, figsize=(13, 7))

    for i, axis in enumerate(['X', 'Y', 'Z']):
        ax_g.plot(ts, gyro_dps[:, i], label=f'Gyro {axis}', linewidth=0.9)
        ax_a.plot(ts, accel_g[:, i],  label=f'Accel {axis}', linewidth=0.9)

    ax_g.set_ylabel('Angular velocity (dps)')
    ax_g.legend(loc='upper right')
    ax_g.grid(True, alpha=0.3)
    ax_g.axhline(0.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.6)
    ax_g.set_title(f'APEX IMU log  —  {len(frames)} frames, {ts[-1]:.1f} s')

    ax_a.set_ylabel('Acceleration (g)')
    ax_a.set_xlabel('Time (s)')
    ax_a.legend(loc='upper right')
    ax_a.grid(True, alpha=0.3)
    ax_a.axhline( 1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.6)
    ax_a.axhline( 0.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.6)
    ax_a.axhline(-1.0, color='gray', linestyle='--', alpha=0.4, linewidth=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"\n  saved figure → {save_path}")
    plt.show()


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else 'APEX0000.BIN')
    if not path.exists():
        sys.exit(f"file not found: {path}")

    print(f"reading {path} ({path.stat().st_size} bytes)")
    frames = parse_file(path)
    if len(frames) == 0:
        sys.exit("no complete frames parsed")

    print()
    summarize(frames)
    plot(frames, path.with_suffix('.png'))


if __name__ == '__main__':
    main()