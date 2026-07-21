"""Tests for metadata_models: the Pydantic v2 schema for APEX session YAML.

Positive path is `session_template.yaml` itself — it is the ground truth for
what the schema must accept. Negative cases each mutate one field of a
deep-copied template dict, so a failure points at exactly one rule.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from metadata_models import (
    Difficulty,
    Outcome,
    SensorPosition,
    SessionDocument,
    load_session,
    parse_v_grade,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "session_template.yaml"


@pytest.fixture(scope="module")
def template_raw() -> dict:
    with TEMPLATE.open("r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def doc(template_raw: dict) -> dict:
    """A fresh mutable copy of the template for each negative test."""
    return copy.deepcopy(template_raw)


# ─────────────────────────────────────────────────────────────────────────
# parse_v_grade
# ─────────────────────────────────────────────────────────────────────────


def test_parse_v_grade_numeric():
    assert parse_v_grade("V0") == 0
    assert parse_v_grade("V4") == 4
    assert parse_v_grade("V17") == 17


def test_parse_v_grade_vb_sentinel():
    # Documented convention: VB ("V-basic", below V0) maps to -1 so that the
    # full grade scale stays a single monotonically ordered integer axis.
    assert parse_v_grade("VB") == -1


def test_parse_v_grade_is_case_insensitive_and_strips():
    assert parse_v_grade(" v4 ") == 4
    assert parse_v_grade("vb") == -1


@pytest.mark.parametrize("bad", ["V18", "V-1", "5.11a", "V", "", "VV4", "4", None, "V4+"])
def test_parse_v_grade_rejects_malformed(bad):
    with pytest.raises((ValueError, TypeError)):
        parse_v_grade(bad)


# ─────────────────────────────────────────────────────────────────────────
# Positive: the template must parse
# ─────────────────────────────────────────────────────────────────────────


def test_template_parses(template_raw: dict):
    session = SessionDocument.model_validate(template_raw)
    assert session.session.session_id == "S20260519-NWC-01"
    assert len(session.climbers) == 2
    assert len(session.routes) == 2
    assert len(session.attempts) == 3


def test_load_session_from_path():
    session = load_session(TEMPLATE)
    assert isinstance(session, SessionDocument)
    assert session.schema_version == "1.0"


def test_sensor_layout_maps_indices_to_positions():
    session = load_session(TEMPLATE)
    layout = session.sensor_layout.as_index_map()
    assert layout[0] == SensorPosition.RIGHT_WRIST
    assert layout[4] == SensorPosition.HIP
    assert set(layout) == {0, 1, 2, 3, 4}


def test_attempt_convenience_fields():
    session = load_session(TEMPLATE)
    a1 = session.attempt_by_id("S20260519-NWC-01-A001")
    assert a1.posted_grade_int == 4
    assert a1.perceived_grade_int == 5
    assert a1.subjective_difficulty is Difficulty.MEDIUM
    assert a1.difficulty_int == 1
    assert a1.outcome is Outcome.FALL
    assert a1.duration_us == 58_901_000 - 32_500_000


def test_climb_attempts_excludes_baseline():
    session = load_session(TEMPLATE)
    ids = [a.attempt_id for a in session.climb_attempts()]
    assert ids == ["S20260519-NWC-01-A001", "S20260519-NWC-01-A002"]


def test_per_attempt_sensor_layout_override():
    """A remount mid-session swaps one IMU's position for that attempt only."""
    raw = yaml.safe_load(TEMPLATE.read_text())
    raw["attempts"][2]["sensor_layout_override"] = {"imu_4": "left_upper_arm"}
    session = SessionDocument.model_validate(raw)

    a2 = session.attempt_by_id("S20260519-NWC-01-A002")
    a1 = session.attempt_by_id("S20260519-NWC-01-A001")

    # Override is a partial patch over the session layout, not a replacement.
    resolved = session.resolved_layout(a2)
    assert resolved[4] == SensorPosition.LEFT_UPPER_ARM
    assert resolved[0] == SensorPosition.RIGHT_WRIST
    # Untouched attempts still see the session default.
    assert session.resolved_layout(a1)[4] == SensorPosition.HIP


# ─────────────────────────────────────────────────────────────────────────
# Negative cases (>= 6 required)
# ─────────────────────────────────────────────────────────────────────────


def test_neg_malformed_grade_string(doc):
    doc["attempts"][1]["perceived_grade"] = "V23"
    with pytest.raises(ValidationError, match="(?i)grade"):
        SessionDocument.model_validate(doc)


def test_neg_missing_required_field(doc):
    del doc["session"]["session_id"]
    with pytest.raises(ValidationError, match="session_id"):
        SessionDocument.model_validate(doc)


def test_neg_difficulty_out_of_enum(doc):
    doc["attempts"][1]["subjective_difficulty"] = "brutal"
    with pytest.raises(ValidationError, match="(?i)subjective_difficulty"):
        SessionDocument.model_validate(doc)


def test_neg_attempt_end_before_start(doc):
    doc["attempts"][1]["timestamp_us_end"] = doc["attempts"][1]["timestamp_us_start"] - 1
    with pytest.raises(ValidationError, match="(?i)end"):
        SessionDocument.model_validate(doc)


def test_neg_duplicate_attempt_ids(doc):
    doc["attempts"][2]["attempt_id"] = doc["attempts"][1]["attempt_id"]
    with pytest.raises(ValidationError, match="(?i)duplicate"):
        SessionDocument.model_validate(doc)


def test_neg_invalid_sensor_position_value(doc):
    doc["sensor_layout"]["imu_2"] = "left_ankle"
    with pytest.raises(ValidationError):
        SessionDocument.model_validate(doc)


def test_neg_invalid_sensor_layout_key(doc):
    doc["sensor_layout"]["imu_9"] = "hip"
    with pytest.raises(ValidationError, match="(?i)imu_9"):
        SessionDocument.model_validate(doc)


def test_neg_route_id_with_slash_rejected(doc):
    """HDF5 group names cannot contain '/'; reject at the schema boundary."""
    doc["routes"][0]["route_id"] = "NWC/cave/blue-12"
    doc["attempts"][1]["route_id"] = "NWC/cave/blue-12"
    doc["attempts"][2]["route_id"] = "NWC/cave/blue-12"
    with pytest.raises(ValidationError, match="(?i)'/'|slash"):
        SessionDocument.model_validate(doc)


def test_neg_attempt_references_unknown_climber(doc):
    doc["attempts"][1]["climber_id"] = "C999"
    with pytest.raises(ValidationError, match="C999"):
        SessionDocument.model_validate(doc)


def test_neg_attempt_references_unknown_route(doc):
    doc["attempts"][1]["route_id"] = "NOT-A-ROUTE"
    with pytest.raises(ValidationError, match="NOT-A-ROUTE"):
        SessionDocument.model_validate(doc)


def test_neg_timestamp_exceeds_uint32(doc):
    # TIM5 is a 32-bit microsecond counter; a value above 2^32-1 cannot have
    # come from the firmware and means the YAML was authored wrong.
    doc["attempts"][1]["timestamp_us_end"] = 2**32
    with pytest.raises(ValidationError):
        SessionDocument.model_validate(doc)


def test_neg_climb_attempt_missing_consent(doc):
    doc["climbers"][0]["consent_recorded"] = False
    with pytest.raises(ValidationError, match="(?i)consent"):
        SessionDocument.model_validate(doc)


def test_neg_climb_attempt_missing_route_id(doc):
    del doc["attempts"][1]["route_id"]
    with pytest.raises(ValidationError, match="(?i)route_id"):
        SessionDocument.model_validate(doc)


def test_neg_unknown_raw_file_reference(doc):
    doc["attempts"][1]["raw_file"] = "APEX9999.BIN"
    with pytest.raises(ValidationError, match="APEX9999.BIN"):
        SessionDocument.model_validate(doc)
