"""Synthetic IMU data generation for APEX pipeline validation.

Two layers:

1. **Signal generation** — lifted verbatim from `data_generator.ipynb` so the
   pipeline demo and the CNN training notebook share one definition of what
   synthetic climbing data looks like. `tests/test_run_pipeline.py` executes
   the notebook's own cells and asserts bit-identical output, so if one copy
   is edited without the other, the suite goes red. If you change the signal
   design, change it in both places (or better: make the notebook import from
   here — `from synthetic_imu import generate_synthetic_imu`).

2. **Frame encoding** — new. Turns those signals into a byte-exact stand-in for
   a real SD-card capture: 32-byte v2 frames, round-robin across 5 IMUs, at the
   firmware's 104 Hz ODR, with rest periods between attempts so that per-attempt
   windowing has something to actually exclude. Frames are built with
   `ingest.encode_frame`, so the fixture format cannot drift from the parser.

The generator's float channels are unitless. Converting them to physical units
requires picking a scale (see ACCEL_G_PER_UNIT / GYRO_DPS_PER_UNIT below) —
those constants are a judgement call, documented at their definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ingest import ACCEL_SCALE_G, GYRO_SCALE_DPS, NUM_IMUS, encode_frame
from metadata_models import UINT32_MAX

# ═════════════════════════════════════════════════════════════════════════
# LAYER 1 — signal generation, extracted verbatim from data_generator.ipynb
# ═════════════════════════════════════════════════════════════════════════

# Sampling + tensor shape
FS = 100                 # Hz, the notebook's assumed rate for the CNN tensor
T_MAX = 4500             # samples per padded attempt (45 s at FS=100)
N_CHANNELS = 30          # 5 IMUs x (accel xyz + gyro xyz)
N_IMUS = 5
ACCEL_AXES = [0, 1, 2]   # relative to each IMU's 6-channel block
GYRO_AXES = [3, 4, 5]

# Per-class parameters. Index = class label.
CLASS_PARAMS = [
    {"accel_std": 0.3, "gyro_std": 0.2, "length": 1500, "freq_hz": 1.0},
    {"accel_std": 0.8, "gyro_std": 0.6, "length": 2500, "freq_hz": 2.0},
    {"accel_std": 1.5, "gyro_std": 1.2, "length": 3800, "freq_hz": 3.0},
]

NOISE_STD = 0.1          # shared Gaussian noise floor across all classes
SINE_AMPLITUDE = 0.5     # amplitude of the class-specific sinusoidal component

CLASS_NAMES = ("easy", "medium", "hard")


def _generate_one_attempt(params, rng):
    """Generate a single (T_MAX, 30) attempt for the given class params."""
    length = params["length"]
    accel_std = params["accel_std"]
    gyro_std = params["gyro_std"]
    freq_hz = params["freq_hz"]

    # Start with zeros so the pad region (length:T_MAX) stays exactly zero.
    x = np.zeros((T_MAX, N_CHANNELS), dtype=np.float32)

    # Time vector for the active portion of the attempt.
    t = np.arange(length) / FS  # seconds

    # Class-specific sinusoidal component shared across channels but with a
    # random phase per channel so the network sees coordinated-but-not-identical
    # oscillation across IMUs.
    phases = rng.uniform(0, 2 * np.pi, size=N_CHANNELS).astype(np.float32)
    sine = SINE_AMPLITUDE * np.sin(
        2 * np.pi * freq_hz * t[:, None] + phases[None, :]
    ).astype(np.float32)

    # Structured random walk in accel channels scaled to accel_std, plus the
    # same idea scaled to gyro_std for gyro channels. This gives the variance
    # signal its raw magnitude.
    for imu in range(N_IMUS):
        base = imu * 6
        for axis in ACCEL_AXES:
            ch = base + axis
            x[:length, ch] = rng.normal(0.0, accel_std, size=length)
        for axis in GYRO_AXES:
            ch = base + axis
            x[:length, ch] = rng.normal(0.0, gyro_std, size=length)

    # Overlay the class-specific sinusoid on the active region only.
    x[:length, :] += sine

    # Gaussian noise floor, same sigma for all classes, applied only to the
    # active region. The zero-padded tail remains identically zero so the
    # transition index is an unambiguous duration cue.
    x[:length, :] += rng.normal(0.0, NOISE_STD, size=(length, N_CHANNELS)).astype(np.float32)

    return x


def generate_synthetic_imu(n_per_class, seed=42):
    """
    Generate a synthetic 3-class IMU dataset for APEX pipeline validation.

    Parameters
    ----------
    n_per_class : int
        Number of attempts per class. Total samples = 3 * n_per_class.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (3*n_per_class, 4500, 30), dtype float32
    y : np.ndarray, shape (3*n_per_class,),           dtype int64
        Class labels: 0=easy, 1=medium, 2=hard.
    """
    rng = np.random.default_rng(seed)

    n_total = 3 * n_per_class
    X = np.zeros((n_total, T_MAX, N_CHANNELS), dtype=np.float32)
    y = np.zeros(n_total, dtype=np.int64)

    idx = 0
    for class_label in range(3):
        params = CLASS_PARAMS[class_label]
        for _ in range(n_per_class):
            X[idx] = _generate_one_attempt(params, rng)
            y[idx] = class_label
            idx += 1

    # Shuffle so class order is not an artifact in any downstream split.
    perm = rng.permutation(n_total)
    return X[perm], y[perm]


# ═════════════════════════════════════════════════════════════════════════
# LAYER 2 — frame encoding: signals -> a faithful stand-in for an SD capture
# ═════════════════════════════════════════════════════════════════════════

ODR_HZ = 104
DELTA_US = round(1_000_000 / ODR_HZ)  # 9615 us; integer so timestamps stay exact

# Per-IMU sampling skew. Real hardware polls the five sensors in sequence over
# SPI1, so their timestamps are staggered within a sample period rather than
# identical. Keeping this well under DELTA_US means every IMU still lands the
# same number of samples inside a given attempt window.
IMU_SKEW_US = 137

# ═══ KNOWN: SYNTHETIC SCALE FACTORS ARE PLACEHOLDERS ═════════════════════
# grep "KNOWN: SYNTHETIC SCALE" to find everywhere this matters.
#
# The generator's channels are unitless (per-class std 0.3-1.5). The two
# constants below are the ONLY thing converting them into g and dps, and they
# were chosen to exercise the sensor range without clipping — they are NOT
# measured from real climbing motion:
#   accel: 1.5 std x 0.4 = 0.6 g std   -> 4 sigma + 1 g gravity = 3.4 g < 4 g FS
#   gyro : 1.2 std x 200 = 240 dps std -> 4 sigma = 960 dps     < 1000 dps FS
#
# WHEN THESE ARE REPLACED WITH REAL-DATA-DERIVED VALUES, these symptoms point
# HERE — check this block before suspecting the parser, the windowing, or the
# model:
#   * run_pipeline reports "clipped samples" > 0 — the new scale exceeds the
#     configured full scale (+/-4 g, +/-1000 dps, set in the LSM6DSO CTRL1_XL
#     and CTRL2_G registers). Either rescale or widen the FS in the firmware.
#   * CNN class separation collapses — the three classes are separated by
#     VARIANCE, so changing the scale rescales the exact feature being learned.
#     A retrained model is expected; a broken pipeline is not.
#   * Magnitudes look wrong but QC is clean — scale factors touch neither
#     timing nor framing. If drop rate / CRC / sync are unchanged, the defect
#     is here, not in ingest.
#   * Old and new HDF5 files disagree in magnitude only — compare the
#     scale_factors_version recorded in each file's root `notes` attribute.
#
# Bump SCALE_FACTORS_VERSION whenever these change, so every generated
# artifact stays traceable to the constants that produced it.
SCALE_FACTORS_VERSION = "synthetic-scale-v1"
ACCEL_G_PER_UNIT = 0.4
GYRO_DPS_PER_UNIT = 200.0
GRAVITY_G = 1.0  # added to accel Z so a resting sensor reads ~1 g, like reality
# ═════════════════════════════════════════════════════════════════════════

# Rest periods between attempts: sensor is on the climber, climber is standing
# around. Near-zero rotation, gravity on Z, small motion noise.
REST_ACCEL_NOISE_G = 0.03
REST_GYRO_NOISE_DPS = 2.0

LEAD_IN_S = 2.0   # recording starts before the first attempt
TAIL_S = 2.0      # and stops after the last

ACCEL_FS_G = 32767 * ACCEL_SCALE_G
GYRO_FS_DPS = 32767 * GYRO_SCALE_DPS

SENSOR_LAYOUT = {
    "imu_0": "right_wrist",
    "imu_1": "left_wrist",
    "imu_2": "right_upper_arm",
    "imu_3": "left_upper_arm",
    "imu_4": "hip",
}

# One route per difficulty class, so route difficulty and the generated signal
# class tell the same story.
ROUTES = [
    {"route_id": "SYN-slab-green-01", "posted_grade": "V2", "gym_section": "slab"},
    {"route_id": "SYN-cave-blue-02", "posted_grade": "V4", "gym_section": "cave"},
    {"route_id": "SYN-roof-red-03", "posted_grade": "V6", "gym_section": "roof"},
]


@dataclass
class SyntheticSession:
    """A complete synthetic capture: the bytes, the sidecar, and the truth."""

    bin_filename: str
    bin_bytes: bytes
    metadata: dict[str, Any]
    attempts: list[dict[str, Any]]
    # attempt_id -> imu_index -> {"timestamps_us", "gyro_raw", "accel_raw"}
    truth: dict[str, dict[int, dict[str, np.ndarray]]] = field(repr=False, default_factory=dict)
    n_frames: int = 0
    n_clipped: int = 0
    duration_us: int = 0


def _quantize(values_g_or_dps: np.ndarray, lsb: float) -> tuple[np.ndarray, int]:
    """Convert physical units to raw int16 LSBs, counting any clipping."""
    raw = np.rint(values_g_or_dps / lsb)
    clipped = int(np.count_nonzero((raw < -32768) | (raw > 32767)))
    return np.clip(raw, -32768, 32767).astype(np.int16), clipped


def _attempt_block(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert a (L, 30) generator signal into per-IMU raw int16 arrays.

    Returns (gyro_raw, accel_raw) each shaped (L, NUM_IMUS, 3), plus a clip count.
    Channel layout follows the notebook: channels base+0..2 of each IMU's
    6-channel block are accelerometer, base+3..5 are gyro.
    """
    length = signal.shape[0]
    accel_g = np.zeros((length, NUM_IMUS, 3), dtype=np.float32)
    gyro_dps = np.zeros((length, NUM_IMUS, 3), dtype=np.float32)

    for imu in range(NUM_IMUS):
        base = imu * 6
        for j, axis in enumerate(ACCEL_AXES):
            accel_g[:, imu, j] = signal[:, base + axis] * ACCEL_G_PER_UNIT
        for j, axis in enumerate(GYRO_AXES):
            gyro_dps[:, imu, j] = signal[:, base + axis] * GYRO_DPS_PER_UNIT

    accel_g[:, :, 2] += GRAVITY_G  # gravity on Z

    accel_raw, clip_a = _quantize(accel_g, ACCEL_SCALE_G)
    gyro_raw, clip_g = _quantize(gyro_dps, GYRO_SCALE_DPS)
    return gyro_raw, accel_raw, clip_a + clip_g


def _rest_block(n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    """Low-motion filler between attempts: gravity on Z, near-zero rotation."""
    shape = (n_samples, NUM_IMUS, 3)
    accel_g = rng.normal(0.0, REST_ACCEL_NOISE_G, size=shape).astype(np.float32)
    accel_g[:, :, 2] += GRAVITY_G
    gyro_dps = rng.normal(0.0, REST_GYRO_NOISE_DPS, size=shape).astype(np.float32)

    accel_raw, clip_a = _quantize(accel_g, ACCEL_SCALE_G)
    gyro_raw, clip_g = _quantize(gyro_dps, GYRO_SCALE_DPS)
    return gyro_raw, accel_raw, clip_a + clip_g


def _encode_block(
    gyro_raw: np.ndarray,
    accel_raw: np.ndarray,
    block_start_us: int,
) -> tuple[list[bytes], np.ndarray]:
    """Encode one time block as round-robin frames across all IMUs.

    Returns the frame list (file order) and a (L, NUM_IMUS) timestamp array.
    """
    length = gyro_raw.shape[0]
    sample_us = block_start_us + np.arange(length, dtype=np.int64) * DELTA_US
    timestamps = sample_us[:, None] + np.arange(NUM_IMUS, dtype=np.int64) * IMU_SKEW_US

    frames: list[bytes] = []
    for s in range(length):
        for imu in range(NUM_IMUS):
            frames.append(
                encode_frame(
                    int(timestamps[s, imu]),
                    imu,
                    tuple(int(v) for v in gyro_raw[s, imu]),
                    tuple(int(v) for v in accel_raw[s, imu]),
                )
            )
    return frames, timestamps


def _build_metadata(
    session_id: str,
    bin_filename: str,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the YAML sidecar as a plain dict (validated by the caller)."""
    return {
        "schema_version": "1.0",
        "session": {
            "session_id": session_id,
            "date": "2026-05-19",
            "start_time_local": "18:30",
            "timezone": "America/New_York",
            "collected_by": "run_pipeline.py (synthetic)",
            # Provenance travels with the data: ingest.write_hdf5 copies this
            # into the HDF5 root attrs, so any artifact can be traced back to
            # the scale factors that produced it. See the KNOWN block above.
            "notes": (
                "SYNTHETIC SESSION — generated by synthetic_imu.synthesize_session. "
                "No human subject was recorded. Do not mix with real captures. "
                f"scale_factors_version={SCALE_FACTORS_VERSION} "
                f"(accel_g_per_unit={ACCEL_G_PER_UNIT}, "
                f"gyro_dps_per_unit={GYRO_DPS_PER_UNIT}, gravity_g={GRAVITY_G}). "
                "KNOWN: those scale factors are placeholders chosen to fill the "
                "sensor range, NOT measured from real climbing."
            ),
            "gym": {"name": "Synthetic Bouldering Lab", "city": "Hanover, NH", "code": "SYN"},
            "firmware": {
                "repo": "apex-firmware",
                "commit": "synth00",
                "frame_format_version": 2,
                "sample_rate_hz": ODR_HZ,
            },
            "raw_files": [{"filename": bin_filename, "sd_card_id": "SD-SYNTH"}],
        },
        "climbers": [
            {
                "climber_id": "C001",
                "pseudonym": "synthetic-climber",
                "height_cm": 178,
                "weight_kg": 72,
                "ape_index_cm": 2,
                "dominant_hand": "right",
                "years_climbing": 4,
                "age": 21,
                "typical_max_onsight": "V5",
                "typical_max_redpoint": "V7",
                # Synthetic data has no human subject, so there is nothing to
                # consent to — but the schema requires the flag, and ingestion
                # refuses to run without it.
                "consent_recorded": True,
            }
        ],
        "sensor_layout": {
            **SENSOR_LAYOUT,
            "orientation_convention": "+X forward, +Y up, +Z right (per-sensor body frame)",
            "mounting": "Synthetic — no physical mounting.",
            "calibration": {"static_baseline_recorded": False},
        },
        "routes": [
            {
                **route,
                "setter": "synthetic",
                "date_set": "2026-04-15",
                "style_tags": ["synthetic"],
                "height_m": 4.2,
            }
            for route in ROUTES
        ],
        "attempts": [
            {
                "attempt_id": a["attempt_id"],
                "kind": "climb",
                "climber_id": "C001",
                "route_id": a["route_id"],
                "attempt_number": a["attempt_number"],
                "raw_file": bin_filename,
                "timestamp_us_start": a["timestamp_us_start"],
                "timestamp_us_end": a["timestamp_us_end"],
                "posted_grade": a["posted_grade"],
                "perceived_grade": a["perceived_grade"],
                "subjective_difficulty": a["subjective_difficulty"],
                "outcome": a["outcome"],
                "fall_move_number": a["fall_move_number"],
                "fatigue_pre": a["fatigue_pre"],
                "fatigue_post": a["fatigue_post"],
                "rest_seconds_before": a["rest_seconds_before"],
                "sensor_layout_override": None,
                "notes": f"Synthetic class-{a['class_label']} attempt.",
            }
            for a in attempts
        ],
    }


def synthesize_session(
    n_attempts: int = 5,
    seed: int = 42,
    rest_seconds: float = 20.0,
    bin_filename: str = "APEX_SYNTH.BIN",
) -> SyntheticSession:
    """Generate a complete synthetic capture: .BIN bytes + validated metadata.

    Attempts cycle through the three difficulty classes, separated by rest
    periods, so that per-attempt windowing has to genuinely select a subset of
    the stream rather than trivially taking all of it.

    Deterministic: the same (n_attempts, seed, rest_seconds) always produces
    byte-identical output.
    """
    if n_attempts < 1:
        raise ValueError(f"n_attempts must be >= 1, got {n_attempts}")

    rng = np.random.default_rng(seed)
    session_id = f"S-SYNTH-{seed:04d}"

    class_labels = [i % 3 for i in range(n_attempts)]
    lengths = [CLASS_PARAMS[c]["length"] for c in class_labels]
    rest_samples = max(int(round(rest_seconds * ODR_HZ)), 1)

    # Plan the timeline BEFORE generating any signal, so a session that would
    # overflow TIM5 fails in milliseconds instead of after minutes of work.
    lead_in_samples = max(int(round(LEAD_IN_S * ODR_HZ)), 1)
    tail_samples = max(int(round(TAIL_S * ODR_HZ)), 1)
    total_samples = (
        lead_in_samples + sum(lengths) + rest_samples * (n_attempts - 1) + tail_samples
    )
    total_us = total_samples * DELTA_US
    if total_us > UINT32_MAX:
        raise ValueError(
            f"Synthetic session would span {total_us / 1e6:.0f} s, but TIM5 is a "
            f"32-bit microsecond counter that wraps at {UINT32_MAX / 1e6:.0f} s "
            f"(~71.6 min). Reduce --n-attempts or --rest-seconds. Generating a "
            f"session that wraps would produce non-monotonic timestamps and "
            f"silently corrupt every attempt window."
        )

    frames: list[bytes] = []
    attempts: list[dict[str, Any]] = []
    truth: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    n_clipped = 0
    cursor_us = 0
    route_counts: dict[str, int] = {}

    def emit_rest(n_samples: int) -> None:
        nonlocal cursor_us, n_clipped
        gyro_raw, accel_raw, clipped = _rest_block(n_samples, rng)
        block_frames, _ = _encode_block(gyro_raw, accel_raw, cursor_us)
        frames.extend(block_frames)
        n_clipped += clipped
        cursor_us += n_samples * DELTA_US

    emit_rest(lead_in_samples)

    for i, class_label in enumerate(class_labels):
        if i > 0:
            emit_rest(rest_samples)

        params = CLASS_PARAMS[class_label]
        length = params["length"]
        # Trim the notebook's zero padding: only the active region is real data.
        signal = _generate_one_attempt(params, rng)[:length]
        gyro_raw, accel_raw, clipped = _attempt_block(signal)
        n_clipped += clipped

        start_us = cursor_us
        # Half-open window: every sample of every IMU (including the +skew ones)
        # lands inside [start_us, end_us), and the next block starts exactly at
        # end_us so its first sample is excluded.
        end_us = start_us + length * DELTA_US

        block_frames, timestamps = _encode_block(gyro_raw, accel_raw, start_us)
        frames.extend(block_frames)
        cursor_us = end_us

        attempt_id = f"{session_id}-A{i + 1:03d}"
        route = ROUTES[class_label]
        route_counts[route["route_id"]] = route_counts.get(route["route_id"], 0) + 1

        posted = route["posted_grade"]
        posted_int = int(posted[1:])
        perceived_int = int(np.clip(posted_int + int(rng.integers(-1, 2)), 0, 17))
        outcome = "send" if rng.random() < 0.6 else "fall"

        attempts.append(
            {
                "attempt_id": attempt_id,
                "class_label": class_label,
                "class_name": CLASS_NAMES[class_label],
                "route_id": route["route_id"],
                "attempt_number": route_counts[route["route_id"]],
                "n_samples": length,
                "timestamp_us_start": start_us,
                "timestamp_us_end": end_us,
                "duration_s": (end_us - start_us) / 1e6,
                "posted_grade": posted,
                "perceived_grade": f"V{perceived_int}",
                "subjective_difficulty": CLASS_NAMES[class_label],
                "outcome": outcome,
                "fall_move_number": int(rng.integers(1, 9)) if outcome == "fall" else None,
                "fatigue_pre": min(5, 1 + i // 2),
                "fatigue_post": min(5, 2 + i // 2),
                "rest_seconds_before": rest_seconds if i > 0 else None,
            }
        )

        truth[attempt_id] = {
            imu: {
                "timestamps_us": timestamps[:, imu].astype(np.uint64),
                "gyro_raw": gyro_raw[:, imu, :].copy(),
                "accel_raw": accel_raw[:, imu, :].copy(),
            }
            for imu in range(NUM_IMUS)
        }

    emit_rest(tail_samples)

    metadata = _build_metadata(session_id, bin_filename, attempts)

    # Validate here rather than letting the pipeline discover it later: a
    # generator that emits invalid metadata is a bug in this file, not in ingest.
    from metadata_models import SessionDocument

    SessionDocument.model_validate(metadata)

    return SyntheticSession(
        bin_filename=bin_filename,
        bin_bytes=b"".join(frames),
        metadata=metadata,
        attempts=attempts,
        truth=truth,
        n_frames=len(frames),
        n_clipped=n_clipped,
        duration_us=cursor_us,
    )
