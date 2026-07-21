"""Tests for the extended ingest pipeline: YAML metadata, per-attempt HDF5
windowing, the QC report, and --strict threshold enforcement.

Everything here runs on a hand-built synthetic .BIN — no hardware, no SD card.
The frame layout used to build those fixtures comes from `ingest.encode_frame`,
which is the documented inverse of the parser's struct, so the fixtures cannot
drift away from the format the parser expects.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

import ingest
from ingest import (
    FRAME_SIZE,
    NUM_IMUS,
    SYNC_BYTE,
    build_qc_report,
    compute_sync_stats,
    crc8,
    encode_frame,
    load_session_metadata,
    parse_bin_file,
    window_indices,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ODR_HZ = 104
DELTA_US = 1_000_000 // ODR_HZ  # 9615 us


# ─────────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────────


def build_bin(
    path: Path,
    *,
    n_samples: int = 1040,
    start_us: int = 0,
    n_imus: int = NUM_IMUS,
    corrupt_every: int | None = None,
    drop_every: int | None = None,
) -> int:
    """Write a synthetic .BIN of round-robin IMU frames at 104 Hz.

    Returns the number of frames actually written. `corrupt_every` flips a bit
    in the payload (breaking CRC) every Nth frame; `drop_every` omits every Nth
    frame entirely, which shows up downstream as a timestamp gap.
    """
    frames: list[bytes] = []
    written = 0
    for s in range(n_samples):
        ts = start_us + s * DELTA_US
        for imu in range(n_imus):
            seq = s * n_imus + imu
            if drop_every and seq % drop_every == 0:
                continue
            gyro = (s % 100, imu * 10, -(s % 50))
            accel = (imu, s % 30, 8192)
            frame = bytearray(encode_frame(ts, imu, gyro, accel))
            if corrupt_every and seq % corrupt_every == 0:
                frame[7] ^= 0xFF  # payload bit flip -> CRC mismatch
            frames.append(bytes(frame))
            written += 1
    path.write_bytes(b"".join(frames))
    return written


def build_yaml(path: Path, bin_name: str, attempts: list[dict]) -> dict:
    """Write a minimal but schema-valid session YAML referencing `bin_name`."""
    doc = {
        "schema_version": "1.0",
        "session": {
            "session_id": "S-TEST-01",
            "date": "2026-05-19",
            "start_time_local": "18:30",
            "timezone": "America/New_York",
            "collected_by": "pytest",
            "notes": "synthetic",
            "gym": {"name": "Test Gym", "city": "Lebanon, NH", "code": "TST"},
            "firmware": {
                "repo": "apex-firmware",
                "commit": "deadbee",
                "frame_format_version": 2,
                "sample_rate_hz": ODR_HZ,
            },
            "raw_files": [{"filename": bin_name, "sd_card_id": "SD-TEST"}],
        },
        "climbers": [
            {
                "climber_id": "C001",
                "pseudonym": "Tester",
                "dominant_hand": "right",
                "consent_recorded": True,
            }
        ],
        "sensor_layout": {
            "imu_0": "right_wrist",
            "imu_1": "left_wrist",
            "imu_2": "right_upper_arm",
            "imu_3": "left_upper_arm",
            "imu_4": "hip",
        },
        "routes": [
            {
                "route_id": "TST-route-01",
                "gym_section": "cave",
                "posted_grade": "V4",
                "setter": "K. Patel",
            }
        ],
        "attempts": attempts,
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return doc


def climb_attempt(attempt_id: str, start_us: int, end_us: int, **kw) -> dict:
    base = {
        "attempt_id": attempt_id,
        "kind": "climb",
        "climber_id": "C001",
        "route_id": "TST-route-01",
        "raw_file": "TEST0001.BIN",
        "attempt_number": 1,
        "timestamp_us_start": start_us,
        "timestamp_us_end": end_us,
        "posted_grade": "V4",
        "perceived_grade": "V5",
        "subjective_difficulty": "medium",
        "outcome": "send",
    }
    base.update(kw)
    return base


@pytest.fixture
def clean_session(tmp_path: Path) -> dict:
    """A 10 s, 5-IMU synthetic capture with two declared attempts."""
    bin_path = tmp_path / "TEST0001.BIN"
    n_frames = build_bin(bin_path, n_samples=1040)
    yaml_path = tmp_path / "session.yaml"
    build_yaml(
        yaml_path,
        "TEST0001.BIN",
        [
            climb_attempt("A001", 1_000_000, 4_000_000),
            climb_attempt("A002", 5_000_000, 9_000_000, attempt_number=2),
        ],
    )
    return {
        "dir": tmp_path,
        "bin": bin_path,
        "yaml": yaml_path,
        "n_frames": n_frames,
        "out": tmp_path / "out",
    }


def run_cli(session: dict, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "ingest.py"),
            "--bin", str(session["bin"]),
            "--meta", str(session["yaml"]),
            "--out", str(session["out"]),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


# ─────────────────────────────────────────────────────────────────────────
# Guardrail tests for the frame logic that must not change
# ─────────────────────────────────────────────────────────────────────────


def test_crc8_known_vector():
    assert crc8(b"123456789") == 0xF4


def test_encode_frame_roundtrips_through_parser(tmp_path: Path):
    frame = encode_frame(123_456, 3, (-1, 2, -3), (4, -5, 6))
    assert len(frame) == FRAME_SIZE
    assert frame[0] == SYNC_BYTE
    assert crc8(frame[:18]) == frame[18]
    assert frame[19:] == bytes(13)  # reserved region zero-filled

    p = tmp_path / "one.BIN"
    p.write_bytes(frame)
    per_imu, stats = parse_bin_file(p)
    assert stats["total_frames"] == 1
    assert per_imu[3]["timestamps_us"].tolist() == [123_456]
    assert per_imu[3]["gyro_raw"].tolist() == [[-1, 2, -3]]
    assert per_imu[3]["accel_raw"].tolist() == [[4, -5, 6]]


# ─────────────────────────────────────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────────────────────────────────────


def test_window_indices_matches_searchsorted_hand_checked():
    """Hand-checked example with known timestamps.

    ts    = [0, 100, 200, 300, 400, 500]
    start = 150 -> insertion point 2 (first element >= 150 is ts[2]=200)
    end   = 400 -> insertion point 4 (first element >= 400 is ts[4]=400)
    So the window is ts[2:4] == [200, 300] — end-exclusive.
    """
    ts = np.array([0, 100, 200, 300, 400, 500], dtype=np.uint64)
    lo, hi = window_indices(ts, 150, 400)
    assert (lo, hi) == (2, 4)
    assert np.array_equal(
        np.asarray([lo, hi]), np.searchsorted(ts, np.asarray([150, 400]))
    )
    assert ts[lo:hi].tolist() == [200, 300]


def test_window_indices_exact_boundary_is_inclusive_at_start():
    ts = np.array([0, 100, 200, 300], dtype=np.uint64)
    lo, hi = window_indices(ts, 100, 300)
    assert ts[lo:hi].tolist() == [100, 200]


def test_window_indices_empty_stream():
    ts = np.array([], dtype=np.uint64)
    assert window_indices(ts, 0, 1000) == (0, 0)


def test_window_indices_window_outside_data():
    ts = np.array([0, 100, 200], dtype=np.uint64)
    lo, hi = window_indices(ts, 5000, 9000)
    assert lo == hi == 3


# ─────────────────────────────────────────────────────────────────────────
# Metadata loading
# ─────────────────────────────────────────────────────────────────────────


def test_load_session_metadata_reads_yaml(clean_session: dict):
    session = load_session_metadata(clean_session["yaml"])
    assert session.session.session_id == "S-TEST-01"
    assert len(session.attempts) == 2


def test_load_session_metadata_rejects_bad_yaml(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    build_yaml(bad, "TEST0001.BIN", [climb_attempt("A001", 100, 50)])
    with pytest.raises(Exception):
        load_session_metadata(bad)


def test_legacy_json_loader_still_importable():
    """The flat-JSON path is retained but unused; keep it callable."""
    assert callable(ingest.load_metadata)


# ─────────────────────────────────────────────────────────────────────────
# Sync stats
# ─────────────────────────────────────────────────────────────────────────


def test_sync_stats_on_clean_file(clean_session: dict):
    raw = clean_session["bin"].read_bytes()
    stats = compute_sync_stats(raw)
    assert stats["frame_slots"] == len(raw) // FRAME_SIZE
    assert stats["sync_hit_rate"] == 1.0


def test_sync_stats_detects_missing_sync(tmp_path: Path):
    p = tmp_path / "x.BIN"
    build_bin(p, n_samples=20)
    raw = bytearray(p.read_bytes())
    raw[0] = 0x42  # clobber one frame's sync byte
    stats = compute_sync_stats(bytes(raw))
    assert stats["sync_hit_rate"] < 1.0
    assert stats["sync_misses"] == 1


# ─────────────────────────────────────────────────────────────────────────
# End-to-end HDF5 structure
# ─────────────────────────────────────────────────────────────────────────


def test_ingest_writes_root_and_attempt_groups(clean_session: dict):
    result = run_cli(clean_session)
    assert result.returncode == 0, result.stderr

    h5_path = clean_session["out"] / "S-TEST-01.hdf5"
    assert h5_path.exists()

    with h5py.File(h5_path, "r") as h5:
        # Root-level per-IMU groups survive (debugging aid).
        for i in range(NUM_IMUS):
            grp = h5[f"imu_{i}"]
            assert grp["timestamps_us"].shape[0] == 1040
            assert grp["accel_g"].shape == (1040, 3)
            assert grp["gyro_dps"].shape == (1040, 3)

        # Per-attempt windowed groups.
        assert "attempts" in h5
        for aid in ("A001", "A002"):
            att = h5[f"attempts/{aid}"]
            assert att.attrs["climber_id"] == "C001"
            assert att.attrs["route_id"] == "TST-route-01"
            assert att.attrs["posted_grade_int"] == 4
            assert att.attrs["perceived_grade_int"] == 5
            assert att.attrs["difficulty_int"] == 1
            for i in range(NUM_IMUS):
                imu = att[f"imu_{i}"]
                assert set(imu.keys()) == {"timestamps_us", "gyro_dps", "accel_g"}
                assert imu.attrs["body_location"] in {
                    "right_wrist", "left_wrist",
                    "right_upper_arm", "left_upper_arm", "hip",
                }


def test_attempt_window_contents_match_searchsorted(clean_session: dict):
    run_cli(clean_session)
    h5_path = clean_session["out"] / "S-TEST-01.hdf5"

    with h5py.File(h5_path, "r") as h5:
        for aid, (start, end) in (
            ("A001", (1_000_000, 4_000_000)),
            ("A002", (5_000_000, 9_000_000)),
        ):
            for i in range(NUM_IMUS):
                full = h5[f"imu_{i}/timestamps_us"][:]
                lo, hi = np.searchsorted(full, np.asarray([start, end]))
                windowed = h5[f"attempts/{aid}/imu_{i}/timestamps_us"][:]
                assert np.array_equal(windowed, full[lo:hi])
                assert windowed.size > 0
                assert windowed[0] >= start
                assert windowed[-1] < end


def test_attempt_window_units_are_physical(clean_session: dict):
    run_cli(clean_session)
    with h5py.File(clean_session["out"] / "S-TEST-01.hdf5", "r") as h5:
        accel = h5["attempts/A001/imu_0/accel_g"][:]
        # Synthetic accel z is a constant 8192 LSB -> 8192 * 0.000122 g.
        assert np.allclose(accel[:, 2], 8192 * ingest.ACCEL_SCALE_G, rtol=1e-5)


def test_root_attrs_carry_session_metadata(clean_session: dict):
    run_cli(clean_session)
    with h5py.File(clean_session["out"] / "S-TEST-01.hdf5", "r") as h5:
        assert h5.attrs["session_id"] == "S-TEST-01"
        assert h5.attrs["schema_version"] == "1.0"
        assert h5.attrs["date"] == "2026-05-19"
        assert list(h5.attrs["imu_placement"]) == [
            "right_wrist", "left_wrist",
            "right_upper_arm", "left_upper_arm", "hip",
        ]


# ─────────────────────────────────────────────────────────────────────────
# QC report
# ─────────────────────────────────────────────────────────────────────────


def test_qc_report_has_all_six_required_fields(clean_session: dict):
    run_cli(clean_session)
    qc_path = clean_session["out"] / "S-TEST-01_qc.json"
    assert qc_path.exists()
    qc = json.loads(qc_path.read_text())

    # 1. per-IMU frame count
    for i in range(NUM_IMUS):
        assert qc["per_imu"][f"imu_{i}"]["frame_count"] == 1040
    # 2. CRC failure rate
    assert qc["raw_file"]["crc_failure_rate"] == 0.0
    # 3. estimated dropped-frame count
    for i in range(NUM_IMUS):
        assert qc["per_imu"][f"imu_{i}"]["estimated_dropped_frames"] == 0
    # 4. sync-byte hit rate
    assert qc["raw_file"]["sync_hit_rate"] == 1.0
    # 5. timestamp monotonicity
    for i in range(NUM_IMUS):
        assert qc["per_imu"][f"imu_{i}"]["timestamp_monotonic"] is True
        assert qc["per_imu"][f"imu_{i}"]["monotonicity_violations"] == 0
    # 6. raw-file SHA256
    expected = hashlib.sha256(clean_session["bin"].read_bytes()).hexdigest()
    assert qc["raw_file"]["sha256"] == expected


def test_qc_flags_frame_size_mismatch(tmp_path: Path):
    """A v1 (34-byte) capture read by the v2 (32-byte) parser.

    Every frame still recovers, because the parser resyncs byte-by-byte after
    a missed sync — that is how Data/firstDataset.BIN gets 864/864 despite
    being a 34-byte file. But frame-aligned sync hits collapse to ~6%, which
    reads as catastrophic corruption unless the report says otherwise.
    """
    frames = [
        encode_frame(i * DELTA_US, i % NUM_IMUS, (0, 0, 0), (0, 0, 0)) + b"\x00\x00"
        for i in range(200)
    ]
    bin_path = tmp_path / "TEST0001.BIN"
    bin_path.write_bytes(b"".join(frames))
    yaml_path = tmp_path / "session.yaml"
    build_yaml(yaml_path, "TEST0001.BIN", [climb_attempt("A001", 0, 400_000)])
    session = {"bin": bin_path, "yaml": yaml_path, "out": tmp_path / "out"}

    run_cli(session)
    raw_qc = json.loads((session["out"] / "S-TEST-01_qc.json").read_text())["raw_file"]

    assert raw_qc["frames_parsed"] == 200
    assert raw_qc["crc_failure_rate"] == 0.0  # nothing is actually corrupt
    assert raw_qc["sync_hit_rate"] < 0.2  # but alignment looks terrible
    assert raw_qc["bytes_per_frame_observed"] == pytest.approx(34.0)
    assert raw_qc["frame_alignment_note"] is not None
    assert "34" in raw_qc["frame_alignment_note"]


def test_qc_no_alignment_note_on_clean_v2_file(clean_session: dict):
    run_cli(clean_session)
    raw_qc = json.loads(
        (clean_session["out"] / "S-TEST-01_qc.json").read_text()
    )["raw_file"]
    assert raw_qc["bytes_per_frame_observed"] == pytest.approx(32.0)
    assert raw_qc["frame_alignment_note"] is None


def test_qc_report_per_attempt_section(clean_session: dict):
    run_cli(clean_session)
    qc = json.loads((clean_session["out"] / "S-TEST-01_qc.json").read_text())
    by_id = {a["attempt_id"]: a for a in qc["per_attempt"]}
    assert set(by_id) == {"A001", "A002"}
    assert by_id["A001"]["duration_s"] == pytest.approx(3.0)
    assert by_id["A001"]["per_imu"]["imu_0"]["frame_count"] > 0


def test_qc_detects_injected_crc_failures(tmp_path: Path):
    bin_path = tmp_path / "TEST0001.BIN"
    build_bin(bin_path, n_samples=200, corrupt_every=10)
    yaml_path = tmp_path / "session.yaml"
    build_yaml(yaml_path, "TEST0001.BIN", [climb_attempt("A001", 0, 1_900_000)])
    session = {"bin": bin_path, "yaml": yaml_path, "out": tmp_path / "out"}

    run_cli(session)
    qc = json.loads((session["out"] / "S-TEST-01_qc.json").read_text())
    # 1 in 10 of the 1000 frames corrupted. The parser's byte-wise resync can
    # re-flag the same corrupted region more than once (a stray 0xA5 inside the
    # payload looks like a new candidate frame), so this is a floor, not an
    # equality — the point is that ~10% corruption is loudly visible.
    assert qc["raw_file"]["crc_failures"] >= 100
    assert qc["raw_file"]["crc_failure_rate"] > 0.05


def test_qc_detects_dropped_frames(tmp_path: Path):
    bin_path = tmp_path / "TEST0001.BIN"
    # 53 is coprime with NUM_IMUS, so the drops walk across all five streams
    # instead of landing on imu_0 every time.
    build_bin(bin_path, n_samples=500, drop_every=53)
    yaml_path = tmp_path / "session.yaml"
    build_yaml(yaml_path, "TEST0001.BIN", [climb_attempt("A001", 0, 4_800_000)])
    session = {"bin": bin_path, "yaml": yaml_path, "out": tmp_path / "out"}

    run_cli(session)
    qc = json.loads((session["out"] / "S-TEST-01_qc.json").read_text())
    total_dropped = sum(
        qc["per_imu"][f"imu_{i}"]["estimated_dropped_frames"] for i in range(NUM_IMUS)
    )
    assert total_dropped > 0


def test_qc_flags_non_monotonic_timestamps(tmp_path: Path):
    """A backwards timestamp must be reported, not silently sorted away."""
    frames = [
        encode_frame(1_000_000, 0, (0, 0, 0), (0, 0, 0)),
        encode_frame(2_000_000, 0, (0, 0, 0), (0, 0, 0)),
        encode_frame(1_500_000, 0, (0, 0, 0), (0, 0, 0)),  # goes backwards
        encode_frame(3_000_000, 0, (0, 0, 0), (0, 0, 0)),
    ]
    bin_path = tmp_path / "TEST0001.BIN"
    bin_path.write_bytes(b"".join(frames))
    yaml_path = tmp_path / "session.yaml"
    build_yaml(yaml_path, "TEST0001.BIN", [climb_attempt("A001", 0, 4_000_000)])
    session = {"bin": bin_path, "yaml": yaml_path, "out": tmp_path / "out"}

    result = run_cli(session)
    qc = json.loads((session["out"] / "S-TEST-01_qc.json").read_text())
    assert qc["per_imu"]["imu_0"]["timestamp_monotonic"] is False
    assert qc["per_imu"]["imu_0"]["monotonicity_violations"] == 1
    assert result.returncode == 0  # non-strict: reported, not fatal


# ─────────────────────────────────────────────────────────────────────────
# --strict
# ─────────────────────────────────────────────────────────────────────────


def test_strict_passes_on_clean_file(clean_session: dict):
    result = run_cli(clean_session, "--strict")
    assert result.returncode == 0, result.stdout + result.stderr
    qc = json.loads((clean_session["out"] / "S-TEST-01_qc.json").read_text())
    assert qc["thresholds_violated"] == []
    assert qc["passed"] is True


def test_strict_fails_on_injected_crc_corruption(tmp_path: Path):
    bin_path = tmp_path / "TEST0001.BIN"
    build_bin(bin_path, n_samples=200, corrupt_every=10)  # 10% >> 0.1% threshold
    yaml_path = tmp_path / "session.yaml"
    build_yaml(yaml_path, "TEST0001.BIN", [climb_attempt("A001", 0, 1_900_000)])
    session = {"bin": bin_path, "yaml": yaml_path, "out": tmp_path / "out"}

    result = run_cli(session, "--strict")
    assert result.returncode != 0
    assert "crc_failure_rate" in result.stdout
    qc = json.loads((session["out"] / "S-TEST-01_qc.json").read_text())
    assert any(v["metric"] == "crc_failure_rate" for v in qc["thresholds_violated"])
    assert qc["passed"] is False


def test_strict_fails_on_non_monotonic(tmp_path: Path):
    frames = [
        encode_frame(1_000_000, 0, (0, 0, 0), (0, 0, 0)),
        encode_frame(2_000_000, 0, (0, 0, 0), (0, 0, 0)),
        encode_frame(1_500_000, 0, (0, 0, 0), (0, 0, 0)),
    ]
    bin_path = tmp_path / "TEST0001.BIN"
    bin_path.write_bytes(b"".join(frames))
    yaml_path = tmp_path / "session.yaml"
    build_yaml(yaml_path, "TEST0001.BIN", [climb_attempt("A001", 0, 3_000_000)])
    session = {"bin": bin_path, "yaml": yaml_path, "out": tmp_path / "out"}

    result = run_cli(session, "--strict")
    assert result.returncode != 0
    assert "monotonic" in result.stdout.lower()


def test_strict_fails_on_high_drop_rate(tmp_path: Path):
    bin_path = tmp_path / "TEST0001.BIN"
    # ~14% of frames dropped, spread across all IMUs (7 coprime with 5).
    build_bin(bin_path, n_samples=400, drop_every=7)
    yaml_path = tmp_path / "session.yaml"
    build_yaml(yaml_path, "TEST0001.BIN", [climb_attempt("A001", 0, 3_800_000)])
    session = {"bin": bin_path, "yaml": yaml_path, "out": tmp_path / "out"}

    result = run_cli(session, "--strict")
    assert result.returncode != 0
    assert "drop" in result.stdout.lower()


def test_build_qc_report_is_pure(clean_session: dict):
    """QC computation must not depend on having written the HDF5 first."""
    session = load_session_metadata(clean_session["yaml"])
    per_imu, stats = parse_bin_file(clean_session["bin"])
    qc = build_qc_report(clean_session["bin"], session, per_imu, stats)
    assert qc["per_imu"]["imu_0"]["frame_count"] == 1040
    assert set(qc) >= {
        "session_id", "raw_file", "per_imu", "per_attempt",
        "thresholds", "thresholds_violated", "passed",
    }


# ─────────────────────────────────────────────────────────────────────────
# HDF5 name safety (the '/' gotcha)
# ─────────────────────────────────────────────────────────────────────────


def test_route_id_with_slash_is_rejected_before_hdf5(tmp_path: Path):
    bin_path = tmp_path / "TEST0001.BIN"
    build_bin(bin_path, n_samples=50)
    yaml_path = tmp_path / "session.yaml"
    build_yaml(yaml_path, "TEST0001.BIN", [climb_attempt("A001", 0, 400_000)])
    doc = yaml.safe_load(yaml_path.read_text())
    doc["routes"][0]["route_id"] = "TST/route/01"
    doc["attempts"][0]["route_id"] = "TST/route/01"
    yaml_path.write_text(yaml.safe_dump(doc, sort_keys=False))

    with pytest.raises(Exception, match="(?i)'/'|slash"):
        load_session_metadata(yaml_path)
