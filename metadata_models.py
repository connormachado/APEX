"""Pydantic v2 schema for an APEX climbing data-collection session.

`session_template.yaml` is the ground truth for this schema — one YAML file per
session (one gym visit), living alongside the raw .BIN files from the SD card.
This module turns that YAML into validated Python objects for the ingestion
pipeline, and is the single source of truth for the schema. Do not redefine
these shapes elsewhere; import them.

Usage:
    from metadata_models import load_session
    session = load_session(Path("session_template.yaml"))

Design notes:
  - Climbers and routes are declared once and referenced by ID from `attempts`.
    Cross-references are validated at the document level, so a typo'd route_id
    fails at load time rather than producing a silently empty HDF5 group.
  - Grades are canonical strings ("VB", "V0".."V17") in the YAML and are
    exposed as ints via `parse_v_grade` for downstream training code.
  - Any ID that becomes an HDF5 group or attribute name is validated to be
    HDF5-safe here, at the schema boundary, rather than sanitized later.
"""

from __future__ import annotations

import datetime as _dt
import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

# TIM5 is a 32-bit microsecond counter, so any timestamp the firmware can emit
# fits in a uint32. A YAML timestamp above this did not come from the hardware.
UINT32_MAX: int = 0xFFFFFFFF

# `VB` ("V-basic", easier than V0) maps to -1 so that the whole V-scale stays a
# single monotonically ordered integer axis with no gaps: VB=-1, V0=0, ..., V17=17.
# Downstream ordinal-regression / class-index code depends on that ordering.
VB_SENTINEL: int = -1
MAX_V_GRADE: int = 17

_V_GRADE_RE = re.compile(r"^V(B|\d{1,2})$")


def parse_v_grade(s: str) -> int:
    """Parse a canonical V-grade string to an int.

    "V0".."V17" map to 0..17; "VB" maps to `VB_SENTINEL` (-1). Case-insensitive
    and whitespace-tolerant. Anything else (ranges, plus/minus, YDS, out-of-range)
    raises — put those in an attempt's `notes` field instead.
    """
    if not isinstance(s, str):
        raise TypeError(f"V-grade must be a string, got {type(s).__name__}: {s!r}")

    token = s.strip().upper()
    match = _V_GRADE_RE.match(token)
    if match is None:
        raise ValueError(
            f"malformed V-grade {s!r}: expected 'VB' or 'V0'..'V{MAX_V_GRADE}'"
        )

    body = match.group(1)
    if body == "B":
        return VB_SENTINEL

    value = int(body)
    if value > MAX_V_GRADE:
        raise ValueError(
            f"V-grade {s!r} out of range: max supported is 'V{MAX_V_GRADE}'"
        )
    return value


def format_v_grade(value: int) -> str:
    """Inverse of `parse_v_grade`."""
    if value == VB_SENTINEL:
        return "VB"
    if not 0 <= value <= MAX_V_GRADE:
        raise ValueError(f"V-grade int {value} out of range")
    return f"V{value}"


def _check_v_grade(v: str) -> str:
    """Field validator: canonicalize a grade string, rejecting malformed ones."""
    parse_v_grade(v)
    return v.strip().upper()


def _check_h5_safe(v: str) -> str:
    """Field validator for any ID used as an HDF5 group or attribute name."""
    if "/" in v:
        raise ValueError(
            f"{v!r} contains '/' — HDF5 group names cannot contain '/'. "
            f"Use '-' or '_' instead."
        )
    if "\x00" in v:
        raise ValueError(f"{v!r} contains a NUL byte; not a valid HDF5 name")
    if v != v.strip() or not v:
        raise ValueError(f"{v!r} must be a non-empty name with no leading/trailing space")
    return v


def _check_timezone(v: str) -> str:
    """Field validator: the tz must be resolvable, or UTC can't be reconstructed."""
    try:
        ZoneInfo(v)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown IANA timezone {v!r}: {exc}") from exc
    return v


VGradeStr = Annotated[str, AfterValidator(_check_v_grade)]
H5Name = Annotated[
    str, StringConstraints(min_length=1, max_length=255), AfterValidator(_check_h5_safe)
]
TimezoneStr = Annotated[str, AfterValidator(_check_timezone)]
TimestampUs = Annotated[int, Field(ge=0, le=UINT32_MAX)]


# ─────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────


class Difficulty(str, Enum):
    """Climber's subjective difficulty rating for one attempt."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Outcome(str, Enum):
    """How the attempt ended."""

    SEND = "send"
    FALL = "fall"
    GIVE_UP = "give_up"
    PARTIAL = "partial"


class AttemptKind(str, Enum):
    """`baseline` = static calibration segment; `climb` = a real attempt."""

    BASELINE = "baseline"
    CLIMB = "climb"


class Hand(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    AMBIDEXTROUS = "ambidextrous"


class SensorPosition(str, Enum):
    """Body location an IMU is mounted at. Five deployed positions."""

    RIGHT_WRIST = "right_wrist"
    LEFT_WRIST = "left_wrist"
    RIGHT_UPPER_ARM = "right_upper_arm"
    LEFT_UPPER_ARM = "left_upper_arm"
    HIP = "hip"


# Ordinal encoding for the CNN label axis: matches data_generator.ipynb's
# 0=easy / 1=medium / 2=hard class indices.
DIFFICULTY_INT: dict[Difficulty, int] = {
    Difficulty.EASY: 0,
    Difficulty.MEDIUM: 1,
    Difficulty.HARD: 2,
}


class _Base(BaseModel):
    """Shared config: unknown keys are an error, not a silent no-op.

    A typo'd YAML key that gets silently dropped is how you discover, three
    weeks into collection, that half your labels were never recorded.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


# ─────────────────────────────────────────────────────────────────────────
# Session-level blocks
# ─────────────────────────────────────────────────────────────────────────


class Gym(_Base):
    name: str
    city: str | None = None
    code: str


class Firmware(_Base):
    repo: str | None = None
    commit: str | None = None
    frame_format_version: int = Field(ge=1, description="1 = 34-byte, 2 = 32-byte")
    sample_rate_hz: int = Field(gt=0)


class RawFile(_Base):
    """One .BIN file produced this session."""

    filename: str
    sd_card_id: str | None = None
    byte_count: int | None = Field(default=None, ge=0)
    sha256: str | None = None


class SessionInfo(_Base):
    session_id: H5Name
    date: _dt.date
    start_time_local: Annotated[str, StringConstraints(pattern=r"^\d{1,2}:\d{2}$")]
    timezone: TimezoneStr
    collected_by: str
    notes: str = ""
    gym: Gym
    firmware: Firmware
    raw_files: list[RawFile] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_raw_filenames(self) -> SessionInfo:
        seen: set[str] = set()
        for rf in self.raw_files:
            if rf.filename in seen:
                raise ValueError(f"duplicate raw_file filename: {rf.filename!r}")
            seen.add(rf.filename)
        return self


class Climber(_Base):
    climber_id: H5Name
    pseudonym: str | None = None
    height_cm: float | None = Field(default=None, gt=0)
    weight_kg: float | None = Field(default=None, gt=0)
    ape_index_cm: float | None = None  # wingspan − height; may be negative
    dominant_hand: Hand | None = None
    years_climbing: float | None = Field(default=None, ge=0)
    age: int | None = Field(default=None, ge=0, le=120)
    typical_max_onsight: VGradeStr | None = None
    typical_max_redpoint: VGradeStr | None = None
    consent_recorded: bool  # IRB / informed consent on file — required, no default


class Calibration(_Base):
    static_baseline_recorded: bool = False
    static_baseline_attempt_id: str | None = None


class SensorLayout(_Base):
    """Maps `imu_index` (the byte in the binary frame) to a body location.

    imu_0..imu_4 are the deployed sensors; imu_5 is a spare and may be omitted.
    Extra `imu_N` keys are rejected — a typo like `imu_7` would otherwise
    silently produce an unmapped sensor.
    """

    imu_0: SensorPosition
    imu_1: SensorPosition
    imu_2: SensorPosition
    imu_3: SensorPosition
    imu_4: SensorPosition
    imu_5: SensorPosition | None = None

    orientation_convention: str = ""
    mounting: str = ""
    calibration: Calibration | None = None

    @model_validator(mode="after")
    def _positions_unique(self) -> SensorLayout:
        placed = [p for p in self.as_index_map().values()]
        if len(set(placed)) != len(placed):
            raise ValueError(
                f"two IMUs mapped to the same body position: {sorted(p.value for p in placed)}"
            )
        return self

    def as_index_map(self) -> dict[int, SensorPosition]:
        """Return {imu_index: position} for every deployed sensor."""
        out: dict[int, SensorPosition] = {}
        for i in range(6):
            pos = getattr(self, f"imu_{i}", None)
            if pos is not None:
                out[i] = pos
        return out


class SensorLayoutOverride(_Base):
    """Partial per-attempt patch over the session layout (e.g. a remount).

    Every field is optional: only the IMUs that actually moved are listed, and
    the rest fall through to the session-level layout.
    """

    imu_0: SensorPosition | None = None
    imu_1: SensorPosition | None = None
    imu_2: SensorPosition | None = None
    imu_3: SensorPosition | None = None
    imu_4: SensorPosition | None = None
    imu_5: SensorPosition | None = None
    notes: str = ""

    def as_index_map(self) -> dict[int, SensorPosition]:
        out: dict[int, SensorPosition] = {}
        for i in range(6):
            pos = getattr(self, f"imu_{i}", None)
            if pos is not None:
                out[i] = pos
        return out


class Route(_Base):
    route_id: H5Name
    gym_section: str | None = None
    wall_angle_deg: float | None = None  # 0 = vertical, + = overhang, − = slab
    posted_grade: VGradeStr
    setter: str | None = None  # major label-noise source; track it
    date_set: _dt.date | None = None
    style_tags: list[str] = Field(default_factory=list)
    estimated_moves: int | None = Field(default=None, ge=1)
    height_m: float | None = Field(default=None, gt=0)
    notes: str = ""

    @property
    def posted_grade_int(self) -> int:
        return parse_v_grade(self.posted_grade)


# ─────────────────────────────────────────────────────────────────────────
# Attempts
# ─────────────────────────────────────────────────────────────────────────


class Attempt(_Base):
    """One climb attempt (or one baseline segment), the core data record."""

    attempt_id: H5Name
    kind: AttemptKind = AttemptKind.CLIMB
    climber_id: str
    raw_file: str

    # Attempt → IMU data linkage is in the MCU's timestamp_us domain.
    timestamp_us_start: TimestampUs
    timestamp_us_end: TimestampUs

    route_id: str | None = None
    attempt_number: int | None = Field(default=None, ge=1)

    # Three labels, always (climb attempts only).
    posted_grade: VGradeStr | None = None
    perceived_grade: VGradeStr | None = None
    subjective_difficulty: Difficulty | None = None

    outcome: Outcome | None = None
    fall_move_number: int | None = Field(default=None, ge=1)

    fatigue_pre: int | None = Field(default=None, ge=1, le=5)  # 1=fresh, 5=cooked
    fatigue_post: int | None = Field(default=None, ge=1, le=5)
    rest_seconds_before: float | None = Field(default=None, ge=0)

    sensor_layout_override: SensorLayoutOverride | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _check_window_and_labels(self) -> Attempt:
        if self.timestamp_us_end <= self.timestamp_us_start:
            raise ValueError(
                f"attempt {self.attempt_id!r}: timestamp_us_end "
                f"({self.timestamp_us_end}) must be strictly greater than "
                f"timestamp_us_start ({self.timestamp_us_start})"
            )

        if self.kind is AttemptKind.CLIMB:
            required = (
                "route_id",
                "posted_grade",
                "perceived_grade",
                "subjective_difficulty",
                "outcome",
            )
            missing = [name for name in required if getattr(self, name) is None]
            if missing:
                raise ValueError(
                    f"climb attempt {self.attempt_id!r} is missing required "
                    f"field(s): {', '.join(missing)}"
                )

        if self.outcome is not Outcome.FALL and self.fall_move_number is not None:
            raise ValueError(
                f"attempt {self.attempt_id!r}: fall_move_number is set but "
                f"outcome is {self.outcome.value if self.outcome else None!r}"
            )
        return self

    @property
    def duration_us(self) -> int:
        return self.timestamp_us_end - self.timestamp_us_start

    @property
    def duration_s(self) -> float:
        return self.duration_us / 1e6

    @property
    def posted_grade_int(self) -> int | None:
        return None if self.posted_grade is None else parse_v_grade(self.posted_grade)

    @property
    def perceived_grade_int(self) -> int | None:
        return (
            None if self.perceived_grade is None else parse_v_grade(self.perceived_grade)
        )

    @property
    def difficulty_int(self) -> int | None:
        """0=easy, 1=medium, 2=hard — the CNN's class index."""
        return (
            None
            if self.subjective_difficulty is None
            else DIFFICULTY_INT[self.subjective_difficulty]
        )


# ─────────────────────────────────────────────────────────────────────────
# Root document
# ─────────────────────────────────────────────────────────────────────────


class SessionDocument(_Base):
    """The whole session YAML: one gym visit."""

    schema_version: str
    session: SessionInfo
    climbers: list[Climber] = Field(min_length=1)
    sensor_layout: SensorLayout
    routes: list[Route] = Field(default_factory=list)
    attempts: list[Attempt] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_cross_references(self) -> SessionDocument:
        for label, ids in (
            ("climber_id", [c.climber_id for c in self.climbers]),
            ("route_id", [r.route_id for r in self.routes]),
            ("attempt_id", [a.attempt_id for a in self.attempts]),
        ):
            seen: set[str] = set()
            for value in ids:
                if value in seen:
                    raise ValueError(f"duplicate {label}: {value!r}")
                seen.add(value)

        climbers = {c.climber_id: c for c in self.climbers}
        route_ids = {r.route_id for r in self.routes}
        raw_files = {rf.filename for rf in self.session.raw_files}

        for attempt in self.attempts:
            where = f"attempt {attempt.attempt_id!r}"
            if attempt.climber_id not in climbers:
                raise ValueError(
                    f"{where} references unknown climber_id {attempt.climber_id!r}"
                )
            if attempt.route_id is not None and attempt.route_id not in route_ids:
                raise ValueError(
                    f"{where} references unknown route_id {attempt.route_id!r}"
                )
            if attempt.raw_file not in raw_files:
                raise ValueError(
                    f"{where} references unknown raw_file {attempt.raw_file!r}; "
                    f"declared files are {sorted(raw_files)}"
                )
            # IRB gate: no consent on file means the data must not be ingested.
            if not climbers[attempt.climber_id].consent_recorded:
                raise ValueError(
                    f"{where}: climber {attempt.climber_id!r} has "
                    f"consent_recorded=false; IRB consent is required before "
                    f"their data may be ingested"
                )

        cal = self.sensor_layout.calibration
        if cal is not None and cal.static_baseline_attempt_id is not None:
            known = {a.attempt_id for a in self.attempts}
            if cal.static_baseline_attempt_id not in known:
                raise ValueError(
                    f"sensor_layout.calibration.static_baseline_attempt_id "
                    f"{cal.static_baseline_attempt_id!r} does not match any attempt"
                )
        return self

    # ── lookups ──────────────────────────────────────────────────────────

    def attempt_by_id(self, attempt_id: str) -> Attempt:
        for attempt in self.attempts:
            if attempt.attempt_id == attempt_id:
                return attempt
        raise KeyError(f"no attempt with id {attempt_id!r}")

    def climber_by_id(self, climber_id: str) -> Climber:
        for climber in self.climbers:
            if climber.climber_id == climber_id:
                return climber
        raise KeyError(f"no climber with id {climber_id!r}")

    def route_by_id(self, route_id: str) -> Route:
        for route in self.routes:
            if route.route_id == route_id:
                return route
        raise KeyError(f"no route with id {route_id!r}")

    def climb_attempts(self) -> list[Attempt]:
        """Real climb attempts, excluding calibration baselines."""
        return [a for a in self.attempts if a.kind is AttemptKind.CLIMB]

    def attempts_for_file(self, filename: str) -> list[Attempt]:
        """Attempts whose data lives in the given .BIN file."""
        return [a for a in self.attempts if a.raw_file == filename]

    def resolved_layout(self, attempt: Attempt) -> dict[int, SensorPosition]:
        """Session sensor layout with this attempt's overrides applied."""
        layout = self.sensor_layout.as_index_map()
        if attempt.sensor_layout_override is not None:
            layout.update(attempt.sensor_layout_override.as_index_map())
        return layout


def load_session(path: Path | str) -> SessionDocument:
    """Load and validate a session YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Session metadata YAML not found at: {path}")
    with path.open("r") as f:
        raw: Any = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return SessionDocument.model_validate(raw)
