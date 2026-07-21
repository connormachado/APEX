# APEX SD → HDF5 Ingestion Pipeline — Spec v1.0

## Goal

A Python CLI that consumes a validated session YAML and the binary `.BIN`
files it references, and produces:

1. A labeled HDF5 file with raw IMU streams demultiplexed per attempt
2. A QC report (JSON) with frame-drop / CRC / timing diagnostics

Companion to the metadata schema in `metadata_models.py`. Together these
two define everything between SD card and trainable tensor.

---

## CLI

```
python -m apex_ingest <session.yaml>
    [--raw-dir DIR]            # where .BIN files live; default: same dir as YAML
    [--output PATH]            # default: <session_id>.h5 next to YAML
    [--qc-report PATH]         # default: <session_id>_qc.json
    [--strict]                 # fail (exit 1) on QC threshold violations
    [--no-sha256]              # skip SHA256 of raw files (faster, less safe)
```

Exit codes: 0 = success, 1 = QC threshold violation in `--strict` mode,
2 = unrecoverable error (missing file, schema fail, etc.).

---

## Binary frame format

### Version 2 (current — 32 bytes)

```
Offset  Size  Field         Notes
─────────────────────────────────────────────────────────────────────
  0     1     sync          0xA5 = data frame, 0xA6 = boundary marker
  1     4     timestamp_us  uint32, little-endian, MCU TIM5 microsecond counter
  5     1     imu_index     uint8, 0..4 (or 5 spare); 0xFF for boundary marker
  6     6     gyro          3× int16 LE: x, y, z (raw LSBs, ±1000 dps FS)
 12     6     accel         3× int16 LE: x, y, z (raw LSBs, ±4g FS)
 18     1     crc8          CCITT poly 0x07, init 0x00, over bytes 0..17
 19    13     reserved      Zero-fill; ignored on read
```

**16 frames per 512-byte sector exactly.** No straddling.

### CRC8 spec

CCITT-8, polynomial `0x07`, initial value `0x00`, no input/output
reflection, no XOR-out. Computed over bytes 0..17 inclusive. A reference
implementation:

```python
def crc8_ccitt(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc
```

Verify against a known vector during unit tests: `crc8_ccitt(b"123456789") == 0xF4`.

### Attempt-boundary markers (optional)

If the firmware writes `0xA6` marker frames at attempt boundaries, the
parser should extract them into a sidecar list `[(timestamp_us, kind)]`
where `kind` is some payload byte in the gyro/accel area (TBD by
firmware spec). The pipeline does **not** use these for attempt
windowing in v1 — the YAML's `timestamp_us_start/end` is authoritative.
But it should surface them in the QC report so you can sanity-check
that markers line up with your declared windows (a discrepancy means
either the YAML is wrong or the operator pressed the button at the wrong
moment).

### Scale factors

```
accel_g     = raw_lsb * 0.000122         # 0.122 mg/LSB at ±4g
gyro_dps    = raw_lsb * 0.035            # 35.0 mdps/LSB at ±1000 dps
```

Store in HDF5 as `float32` in physical units (g and dps), not raw LSBs.
Raw LSBs are a deployment-time concern; downstream code shouldn't have
to know the scale factors.

---

## Parsing algorithm

```
for each raw_file in session.raw_files:
    if not no_sha256: compute SHA256 and stash for QC
    open file, stream-read in 512-byte sector chunks
    state: parser_offset = 0
    for each sector:
        for each candidate frame at offset = 0, frame_size, 2*frame_size, ...:
            if buffer[offset] != 0xA5 (and != 0xA6):
                record resync event; scan forward for next 0xA5 boundary
                continue
            if crc8(buffer[offset..offset+18]) != buffer[offset+18]:
                record crc_failure; advance one byte and scan
                continue
            parse fields; append to per-file frame list
```

**Resync strategy on bad sync byte**: scan forward byte-by-byte for the
next `0xA5`, validate CRC there; if found, resume. Track how many bytes
were skipped — this is a primary corruption metric.

**Performance**: at 5 IMUs × 104 Hz × 32 bytes = ~17 KB/s. A 1-hour
session is ~60 MB. Don't over-engineer for speed; a straightforward
`numpy.frombuffer` per sector is plenty fast and easier to debug than a
streaming state machine.

---

## HDF5 output layout

```
<session_id>.h5
├── /  (root attrs)
│     schema_version       = "1.0"
│     ingest_timestamp_utc = "2026-05-20T18:30:00Z"
│     ingest_tool_version  = "apex_ingest 1.0"
│     source_yaml_sha256   = "..."
│
├── /metadata/
│   ├── session/  (attrs: every field from SessionInfo as scalar attrs)
│   ├── climbers/
│   │   ├── C001/  (attrs: every field from Climber)
│   │   └── C002/  ...
│   ├── routes/
│   │   ├── NWC-cave-blue-12/  (attrs)
│   │   └── ...
│   └── sensor_layout/  (attrs: imu_0..4 location, orientation_convention str)
│
└── /attempts/
    ├── S20260519-NWC-01-A001/
    │     (group attrs: climber_id, route_id, posted_grade, perceived_grade,
    │      subjective_difficulty, outcome, fall_move_number, attempt_number,
    │      fatigue_pre, fatigue_post, rest_seconds_before,
    │      timestamp_us_start, timestamp_us_end, duration_s,
    │      posted_grade_int, perceived_grade_int)
    │   ├── imu_0/  (group attrs: body_location = "right_wrist")
    │   │   ├── timestamp_us   [N] uint32   monotonically increasing
    │   │   ├── gyro           [N, 3] float32   dps, axes XYZ in sensor frame
    │   │   └── accel          [N, 3] float32   g,   axes XYZ in sensor frame
    │   ├── imu_1/  ...
    │   ├── imu_2/  ...
    │   ├── imu_3/  ...
    │   └── imu_4/  ...
    └── S20260519-NWC-01-BASELINE/  ...   # baseline kind = same structure
```

**Notes:**

- Per-IMU `N` will differ slightly across IMUs of the same attempt
  (sub-frame timing). Don't try to force a common length here — let
  downstream resample/align as part of the preprocessing study.
- `posted_grade_int` / `perceived_grade_int` are convenience attrs
  computed via `parse_v_grade()` from `metadata_models`. Saves every
  training script from re-parsing strings.
- Use `h5py` with `compression="gzip"`, `compression_opts=4`. Sessions
  compress well (~3×) because gyro is near-zero in rest periods.
- Chunk timestamp / gyro / accel datasets on the first axis with
  `chunks=(1024,)` / `chunks=(1024, 3)` — reasonable for partial reads
  at training time.

---

## QC report

JSON. Structure:

```json
{
  "session_id": "S20260519-NWC-01",
  "ingest_timestamp_utc": "...",
  "per_file": [
    {
      "filename": "APEX0023.BIN",
      "sha256": "...",
      "byte_count": 49283072,
      "frames_parsed": 1538852,
      "crc_failures": 3,
      "resync_events": 1,
      "bytes_skipped_on_resync": 47,
      "boundary_markers_seen": 8
    }
  ],
  "per_attempt": [
    {
      "attempt_id": "S20260519-NWC-01-A001",
      "duration_s": 26.4,
      "per_imu": {
        "imu_0": {
          "frame_count": 2745,
          "expected_count": 2746,
          "drop_rate": 0.000364,
          "sample_rate_hz_observed": 103.98,
          "max_gap_us": 9712,
          "timestamp_monotonic": true
        },
        "imu_1": { ... },
        ...
      }
    }
  ],
  "thresholds_violated": [],
  "summary": {
    "total_climb_attempts": 2,
    "total_climb_duration_s": 54.0,
    "global_drop_rate": 0.0003,
    "global_crc_failure_rate": 0.000002
  }
}
```

**QC thresholds** (violations listed in `thresholds_violated`, and cause
exit 1 in `--strict` mode):

| Metric | Threshold | Rationale |
|---|---|---|
| `drop_rate` per IMU per attempt | < 0.001 (0.1%) | Plan target; higher means wiring or SPI issue |
| `crc_failure_rate` per file | < 1e-5 | Should be effectively zero with crimped harness |
| `sample_rate_hz_observed` per IMU | within ±2% of 104 Hz | Clock drift / dropped frames |
| `max_gap_us` per IMU per attempt | < 50000 (50 ms) | A 50 ms gap = 5 missed samples; degrades model |
| `timestamp_monotonic` | true | Hard fail; indicates parser bug or rollover |

---

## Acceptance tests

The pipeline should ship with these tests passing. They run without any
real hardware — that's the whole point.

### Unit tests (`tests/test_frame_parser.py`)

1. **CRC8 known vector**: `crc8_ccitt(b"123456789") == 0xF4`
2. **Round-trip one frame**: encode a known (ts, idx, gyro, accel) tuple
   into 32 bytes with valid CRC, parse it back, assert all fields match
3. **CRC failure detection**: flip one bit anywhere in bytes 0–17 of a
   valid frame, assert parser flags `crc_failure` and does not emit
4. **Bad sync byte**: replace sync `0xA5` with `0x42`, assert parser
   records `resync_event`
5. **Boundary marker**: frame with sync `0xA6` parses into the markers
   sidecar, not the main stream
6. **Frame format v1 (legacy)**: parse 34-byte frames identically; only
   reserved-area size differs

### Integration tests (`tests/test_ingest_synthetic.py`)

Build a synthetic `.BIN` + `session.yaml` pair entirely in test fixtures:

7. **Roundtrip integrity**: generate 10s of synthetic frames at 104 Hz
   for 5 IMUs with known gyro/accel patterns (e.g. sine waves at
   distinct frequencies per axis), ingest, read HDF5, assert
   `np.allclose(read_back, original, rtol=1e-5)` after scale-factor
   conversion
8. **Attempt windowing**: synthetic file spans 60s; YAML declares two
   attempts at [10s, 25s] and [35s, 50s]; assert HDF5 contains exactly
   those windows, with frame counts matching `(window_s * 104)` ± 2
9. **Multi-IMU demux**: interleave frames from 5 IMUs round-robin;
   assert per-IMU groups in HDF5 each contain only that IMU's data and
   timestamps are monotonic within each IMU
10. **Induced drops**: drop every 100th frame on imu_2; assert
    `drop_rate == 0.01` ± 0.001 in QC report
11. **Induced corruption**: flip CRC bytes on 0.01% of frames; assert
    `crc_failure_rate` matches in QC report
12. **Threshold violation in strict mode**: drop 5% of frames on one
    IMU, run with `--strict`, assert exit code 1 and
    `thresholds_violated` lists the offending IMU
13. **Boundary markers vs YAML windows**: synthesize markers at offsets
    that disagree with YAML windows by >100 ms; assert QC report flags
    the mismatch (warning, not error — YAML wins)

### Manual smoke test (once hardware is back)

14. Plug in a real `APEX0001.BIN` from a 30-second SD test, run
    pipeline, eyeball QC report. Expected: drop rate < 0.1%, sample
    rate 103–105 Hz, gyro near zero with sensor flat, accel z near 1g.

---

## Project structure

```
apex_pipeline/
├── apex_ingest/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point
│   ├── frame_parser.py      # binary → frame tuples
│   ├── crc.py               # crc8_ccitt
│   ├── hdf5_writer.py       # frames + session → HDF5
│   ├── qc.py                # QC metrics + threshold checks
│   └── metadata_models.py   # COPY from /apex_metadata, do not duplicate logic
├── tests/
│   ├── test_frame_parser.py
│   ├── test_ingest_synthetic.py
│   └── fixtures/
│       ├── synth_session.yaml
│       └── synth_make.py    # generates synth_*.BIN on the fly
├── pyproject.toml           # depends on: numpy, h5py, pydantic>=2, pyyaml, pytest
└── README.md
```

Keep `metadata_models.py` as the single source of truth for the schema
— don't redefine it in the pipeline. Import it.

---

## Out of scope (v1)

- Multi-session ingestion in one command (do it with a shell loop)
- HDF5-to-anything-else exporters (parquet, tfrecord)
- Online / streaming ingest (this is batch-only; live ingest is a
  visualizer concern, not a training-data concern)
- Attempt segmentation from continuous streams (the YAML provides
  windows; auto-segmentation is its own research project)
- Feature extraction (model trains on raw — preprocessing is a separate
  module downstream of this)