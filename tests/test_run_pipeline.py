"""Tests for the synthetic generator and the end-to-end driver script.

The headline test here is `test_generator_matches_notebook`: synthetic_imu.py
lifts its signal-generation code out of data_generator.ipynb, and that test
executes the notebook's own cells and demands bit-identical output. If someone
edits one copy and not the other, this goes red.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import synthetic_imu
from ingest import ACCEL_SCALE_G, FRAME_SIZE, GYRO_SCALE_DPS, NUM_IMUS, parse_bin_file
from metadata_models import SessionDocument
from run_pipeline import VOLATILE_QC_KEYS, run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "data_generator.ipynb"


# ─────────────────────────────────────────────────────────────────────────
# Fidelity to the notebook it was extracted from
# ─────────────────────────────────────────────────────────────────────────


def _exec_notebook_generator() -> dict:
    """Execute the notebook cells that define the generator, return the ns."""
    nb = json.loads(NOTEBOOK.read_text())
    namespace: dict = {}
    for cell in nb["cells"][:4]:  # imports, constants, _generate_one_attempt, generator
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)  # noqa: S102
    return namespace


def test_generator_matches_notebook():
    """synthetic_imu must reproduce data_generator.ipynb exactly, seed for seed."""
    notebook = _exec_notebook_generator()

    for seed in (7, 42):
        x_nb, y_nb = notebook["generate_synthetic_imu"](n_per_class=2, seed=seed)
        x_mod, y_mod = synthetic_imu.generate_synthetic_imu(n_per_class=2, seed=seed)
        assert x_nb.shape == x_mod.shape
        assert np.array_equal(x_nb, x_mod), f"signal drift at seed={seed}"
        assert np.array_equal(y_nb, y_mod), f"label drift at seed={seed}"


def test_generator_constants_match_notebook():
    notebook = _exec_notebook_generator()
    for name in (
        "FS", "T_MAX", "N_CHANNELS", "N_IMUS",
        "ACCEL_AXES", "GYRO_AXES", "NOISE_STD", "SINE_AMPLITUDE",
    ):
        assert getattr(synthetic_imu, name) == notebook[name], f"{name} drifted"
    assert synthetic_imu.CLASS_PARAMS == notebook["CLASS_PARAMS"]


# ─────────────────────────────────────────────────────────────────────────
# Session synthesis
# ─────────────────────────────────────────────────────────────────────────


def test_synthesize_session_is_deterministic():
    a = synthetic_imu.synthesize_session(n_attempts=3, seed=42, rest_seconds=3.0)
    b = synthetic_imu.synthesize_session(n_attempts=3, seed=42, rest_seconds=3.0)
    assert a.bin_bytes == b.bin_bytes
    assert a.metadata == b.metadata


def test_synthesize_session_varies_with_seed():
    a = synthetic_imu.synthesize_session(n_attempts=3, seed=1, rest_seconds=3.0)
    b = synthetic_imu.synthesize_session(n_attempts=3, seed=2, rest_seconds=3.0)
    assert a.bin_bytes != b.bin_bytes


def test_synthesized_metadata_validates():
    synth = synthetic_imu.synthesize_session(n_attempts=4, seed=42, rest_seconds=3.0)
    session = SessionDocument.model_validate(synth.metadata)
    assert len(session.climb_attempts()) == 4
    assert all(a.raw_file == synth.bin_filename for a in session.attempts)


def test_synthesized_bin_is_frame_aligned_and_parses_clean(tmp_path: Path):
    synth = synthetic_imu.synthesize_session(n_attempts=3, seed=42, rest_seconds=3.0)
    assert len(synth.bin_bytes) % FRAME_SIZE == 0

    p = tmp_path / synth.bin_filename
    p.write_bytes(synth.bin_bytes)
    per_imu, stats = parse_bin_file(p)

    assert stats["crc_fail_count"] == 0
    assert stats["bad_index_count"] == 0
    assert stats["total_frames"] == len(synth.bin_bytes) // FRAME_SIZE
    for i in range(NUM_IMUS):
        ts = per_imu[i]["timestamps_us"]
        assert ts.size > 0
        assert np.all(np.diff(ts.astype(np.int64)) > 0), f"imu_{i} not monotonic"


def test_synthesize_rejects_tim5_overflow():
    """A session long enough to wrap TIM5 must fail loudly, not silently wrap."""
    with pytest.raises(ValueError, match="(?i)tim5|wrap|4294"):
        synthetic_imu.synthesize_session(n_attempts=400, seed=42, rest_seconds=60.0)


# ─────────────────────────────────────────────────────────────────────────
# End-to-end driver
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def kept_run(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("kept")
    return run_pipeline(
        n_attempts=3, seed=42, rest_seconds=3.0, keep_artifacts=True, out_dir=out
    )


def test_pipeline_runs_end_to_end(kept_run: dict):
    assert kept_run["qc_report"]["passed"] is True
    assert kept_run["frames_parsed"] > 0
    assert kept_run["frames_parsed"] == kept_run["frames_generated"]
    assert len(kept_run["qc_report"]["per_attempt"]) == 3


def test_pipeline_writes_both_hdf5_group_families(kept_run: dict):
    with h5py.File(kept_run["hdf5_path"], "r") as h5:
        for i in range(NUM_IMUS):
            assert h5[f"imu_{i}/timestamps_us"].size > 0
        assert len(h5["attempts"]) == 3
        for aid in h5["attempts"]:
            for i in range(NUM_IMUS):
                grp = h5[f"attempts/{aid}/imu_{i}"]
                assert grp["timestamps_us"].size > 0
                assert grp["gyro_dps"].shape[1] == 3
                assert grp["accel_g"].shape[1] == 3


def test_scale_factor_provenance_travels_with_the_data(kept_run: dict):
    """A generated artifact must record which scale factors produced it.

    These are placeholders (see the KNOWN block in synthetic_imu.py). When they
    are swapped for real-data-derived values, this attr is what tells you
    whether an old HDF5 predates the change.
    """
    with h5py.File(kept_run["hdf5_path"], "r") as h5:
        notes = h5.attrs["notes"]
    assert synthetic_imu.SCALE_FACTORS_VERSION in notes
    assert f"accel_g_per_unit={synthetic_imu.ACCEL_G_PER_UNIT}" in notes
    assert f"gyro_dps_per_unit={synthetic_imu.GYRO_DPS_PER_UNIT}" in notes
    assert "KNOWN" in notes


def test_pipeline_round_trip_is_exact(kept_run: dict):
    """Every LSB written into the .BIN must survive to the HDF5 unchanged."""
    assert kept_run["round_trip_ok"] is True
    assert kept_run["round_trip_max_abs_error"] == 0.0


def test_attempt_windows_exclude_rest_data(kept_run: dict):
    """Windowing must actually select — rest periods stay out of the windows."""
    with h5py.File(kept_run["hdf5_path"], "r") as h5:
        full = sum(h5[f"imu_{i}/timestamps_us"].size for i in range(NUM_IMUS))
        windowed = sum(
            h5[f"attempts/{aid}/imu_{i}/timestamps_us"].size
            for aid in h5["attempts"]
            for i in range(NUM_IMUS)
        )
    assert windowed < full, "windows captured everything; rest data was not excluded"
    assert windowed > 0


def test_attempt_window_sample_counts_match_generated_lengths(kept_run: dict):
    """Each window should hold the generated attempt's sample count (±1 for skew)."""
    expected = {a["attempt_id"]: a["n_samples"] for a in kept_run["attempts"]}
    with h5py.File(kept_run["hdf5_path"], "r") as h5:
        for aid, n in expected.items():
            for i in range(NUM_IMUS):
                got = h5[f"attempts/{aid}/imu_{i}/timestamps_us"].size
                assert abs(got - n) <= 1, f"{aid}/imu_{i}: {got} vs {n}"


def test_units_are_physical(kept_run: dict):
    with h5py.File(kept_run["hdf5_path"], "r") as h5:
        accel = h5["imu_0/accel_g"][:]
        gyro = h5["imu_0/gyro_dps"][:]
    # Every stored value must be an integer number of LSBs — i.e. value/scale
    # lands on a whole number. (Don't use np.remainder here: for a value just
    # below a multiple it returns ~the divisor, not ~0.)
    quanta = accel / np.float32(ACCEL_SCALE_G)
    assert np.allclose(quanta, np.rint(quanta), atol=1e-2)
    assert np.abs(accel).max() < 32768 * ACCEL_SCALE_G  # within +/-4 g full scale
    assert np.abs(gyro).max() < 1000.0  # within the +/-1000 dps full scale


def test_artifacts_cleaned_up_by_default(tmp_path: Path):
    result = run_pipeline(
        n_attempts=2, seed=42, rest_seconds=3.0, keep_artifacts=False, out_dir=tmp_path
    )
    assert result["artifacts_kept"] is False
    assert not Path(result["workdir"]).exists()


def test_artifacts_kept_when_requested(kept_run: dict):
    assert kept_run["artifacts_kept"] is True
    workdir = Path(kept_run["workdir"])
    assert workdir.exists()
    assert (workdir / "session.yaml").exists()
    assert Path(kept_run["bin_path"]).exists()
    assert Path(kept_run["hdf5_path"]).exists()
    assert Path(kept_run["qc_path"]).exists()


# ─────────────────────────────────────────────────────────────────────────
# Determinism (acceptance test 9)
# ─────────────────────────────────────────────────────────────────────────


def _strip_volatile(report: dict) -> dict:
    """Drop wall-clock/path fields that legitimately differ between runs."""
    pruned = json.loads(json.dumps(report))
    for key in VOLATILE_QC_KEYS:
        pruned.pop(key, None)
    return pruned


def test_same_seed_produces_identical_qc(tmp_path: Path):
    a = run_pipeline(n_attempts=3, seed=42, rest_seconds=3.0, out_dir=tmp_path / "a")
    b = run_pipeline(n_attempts=3, seed=42, rest_seconds=3.0, out_dir=tmp_path / "b")
    assert _strip_volatile(a["qc_report"]) == _strip_volatile(b["qc_report"])
    assert a["qc_report"]["raw_file"]["sha256"] == b["qc_report"]["raw_file"]["sha256"]


def test_different_seed_produces_different_qc(tmp_path: Path):
    a = run_pipeline(n_attempts=3, seed=42, rest_seconds=3.0, out_dir=tmp_path / "a")
    b = run_pipeline(n_attempts=3, seed=7, rest_seconds=3.0, out_dir=tmp_path / "b")
    assert a["qc_report"]["raw_file"]["sha256"] != b["qc_report"]["raw_file"]["sha256"]


# ─────────────────────────────────────────────────────────────────────────
# CLI (acceptance test 8)
# ─────────────────────────────────────────────────────────────────────────


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_pipeline.py"), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_cli_default_invocation():
    result = _run_cli("--n-attempts", "5", "--seed", "42", "--rest-seconds", "3")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "APEX PIPELINE" in out
    assert "frames parsed" in out
    assert "ROUND-TRIP" in out
    assert "QC THRESHOLDS" in out
    assert "attempts split out" in out.lower() or "ATTEMPT WINDOWS" in out


def test_cli_strict_passes_on_clean_synthetic_data():
    result = _run_cli("--n-attempts", "2", "--seed", "42", "--rest-seconds", "3", "--strict")
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_stdout_is_deterministic_for_a_seed():
    """Two identical invocations must print the same thing, modulo paths/clock."""

    def normalize(text: str) -> list[str]:
        return [
            line
            for line in text.splitlines()
            if not any(
                token in line
                for token in ("/", "sha256", "elapsed", "workdir", "\\")
            )
        ]

    a = _run_cli("--n-attempts", "3", "--seed", "42", "--rest-seconds", "3")
    b = _run_cli("--n-attempts", "3", "--seed", "42", "--rest-seconds", "3")
    assert normalize(a.stdout) == normalize(b.stdout)
