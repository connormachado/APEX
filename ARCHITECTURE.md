# APEX Architecture

## Context and goals

APEX is a research pipeline for predicting climbing route difficulty (V-grade) from wearable IMU data. Built for ENGG 192 (Spring 2026, Dartmouth). The system spans three domains: embedded firmware, an offline data pipeline, and a machine learning model.

Goal: strap 5 IMUs onto a climber, log raw motion during a session, turn that into a labeled dataset, and train a model that predicts perceived/actual difficulty from motion alone — eventually running on-device via TFLite.

The repo is a monorepo covering all three layers, plus the Claude Code tooling used to develop it.

## High-level design

Three stages, each with a clear handoff artifact:

**1. Firmware (`APEX_H7/`)** — STM32H753ZI on a NUCLEO-H753ZI board. Polls 5× LSM6DSO IMUs over SPI1 (one CS line per IMU) at 104 Hz, packs readings into 32-byte binary frames, and streams them to a microSD card over SPI3 via FatFS. UART3 is used for debug/status output. Output: raw `.BIN` files on the SD card.

**2. Ingestion (`ingest.py`)** — Offline Python script. Parses a `.BIN` file byte-by-byte, validates each frame's sync byte and CRC8, discards corrupt frames, converts raw LSB values to physical units (g, dps), and writes a structured HDF5 file. Each session also has a hand-authored YAML sidecar (`session_template.yaml`) describing the climbers, routes, sensor placement, and per-attempt labels (posted grade, perceived grade, subjective difficulty, outcome). Output: one HDF5 file per session, samples + QC metrics + attempt labels all in one place.

**3. Modeling (`model.ipynb`, `data_generator.ipynb`)** — Loads HDF5 datasets (`/X` shape `(N, 4500, 30)`: N attempts × 4500 timesteps × 30 channels [5 IMUs × 6 axes], `/y` shape `(N,)` int64 class labels). Trains a 1D-CNN (Conv1D → BatchNorm → ReLU → MaxPool, stacked) to classify attempt difficulty. Exports both a full Keras model (`.keras`) and a quantized TFLite model for eventual on-device inference. `data_generator.ipynb` currently produces synthetic data (`dataset/synthetic_v1.h5`) so the training pipeline can be validated before real climbing sessions are collected.

`apex_visualizer.py` is a standalone QC tool — points at a raw `.BIN` file, independently re-implements the frame parser, and plots gyro/accel time series so you can eyeball a session right after pulling the SD card, before it goes through the full ingest pipeline.

## Data flow

```
5× LSM6DSO --SPI1--> STM32H753ZI --SPI3/FatFS--> SD card (.BIN, 32B frames)
                                                        |
                                          apex_visualizer.py (quick QC, optional)
                                                        |
                          session.yaml  ---->  ingest.py  ----> session.hdf5
                        (labels, metadata)   (parse, CRC check,      |
                                               unit conversion,      |
                                               QC)                   |
                                                                      v
                                                    model.ipynb (1D-CNN train/eval)
                                                                      |
                                                          model.keras + model.tflite
```

Attempt-to-IMU-data linkage runs entirely on the MCU's `timestamp_us` domain (from TIM5), not wall clock — wall clock in the YAML is for humans only. This means ingestion and modeling never need session-relative clock reconciliation, only offsets within a single `.BIN` file.

## Key decisions and trade-offs

**Binary frame format over CSV/JSON on-device.** A fixed 32-byte packed struct (sync + timestamp + IMU index + gyro/accel + CRC8 + reserved) keeps the MCU's write path allocation-free and fast enough to sustain 104 Hz × 5 IMUs (~17.7 KB/s) to SD without buffer overruns. Cost: the format is undocumented outside this repo, so both `ingest.py` and `apex_visualizer.py` maintain independent parsers that must be kept in sync by hand — there's no single source of truth for the frame layout beyond the C struct and the two Python docstrings.

**CRC8 with single-byte resync.** On CRC failure, the parser advances by one byte (not a full frame) and rescans for the sync byte. This recovers from a corrupted region without permanently losing alignment, at the cost of a small chance of false-accepting a corrupted frame if two conditions align (rare given CRC8 + sync byte together).

**Reserved bytes (13 of 32).** Frame format has headroom for future fields (e.g. magnetometer, battery voltage) without a breaking format change — `session_template.yaml` already versions this via `frame_format_version`, so ingestion can branch on format version once it changes.

**Metadata lives in YAML, not the binary.** Labels (grade, outcome, fatigue) are inherently manual/subjective and change after the fact (a climber might revise their perceived difficulty). Keeping them in a separate, human-editable YAML sidecar avoids re-flashing or re-parsing binary data every time a label changes — only `ingest.py` needs to re-run.

**Synthetic data before real data.** The model training pipeline (windowing, normalization, CNN architecture) is being validated against `data_generator.ipynb` synthetic output before real sessions are collected, so pipeline bugs are caught independent of data collection bugs. Explicit note in the code: don't normalize synthetic data the same way real data might need, since normalization would destroy the class-distinguishing variance that was synthetically injected.

**DMA buffers pinned to AXI SRAM, not DTCM.** Hardware constraint on the H753: DMA2 cannot reach DTCM at all — this is a silent failure mode (transfer "succeeds," buffer never updates) if violated, so it's enforced via a linker section attribute rather than left to convention.

## Integration points

- **Firmware ↔ SD card**: FatFS over SPI3. CS on PD2 (verify against CubeMX — this has drifted before).
- **SD card ↔ ingestion**: filesystem only — no live/streaming link. `.BIN` files are pulled off the card manually.
- **Ingestion ↔ modeling**: HDF5 file schema (`/imu_N/{timestamps_us, accel_g, gyro_dps}` plus per-attempt windows at `/attempts/<attempt_id>/imu_N/{timestamps_us, gyro_dps, accel_g}` with labels as group attrs, from `ingest.py` — the old flat `/attempts/labels` dataset was replaced by those groups; `/X`, `/y` from `data_generator.ipynb` — note these are two different HDF5 shapes for two different purposes, raw-per-session vs. windowed-for-training). Anyone building a real-data loader for `model.ipynb` needs to write the raw-session-HDF5 → windowed-`(N, 4500, 30)` transform that `data_generator.ipynb` currently does synthetically.
- **Modeling ↔ firmware (future)**: `model.tflite` is exported but not yet loaded back onto the MCU. On-device inference is the stated end goal but isn't wired up yet — this is the biggest open integration gap in the repo today.

## Open gaps for engineers picking this up

- No automated test coverage for `crc8()` / frame packing on the C side (the CLAUDE.md flags this as a to-do: host-testable pure logic should live in a `tests/` dir).
- Real climbing session data hasn't flowed through `ingest.py` → `model.ipynb` end-to-end yet — the two HDF5 schemas (session-level vs. windowed-training) still need a bridging transform.
- `model.ipynb` has several stub markdown sections ("View the data???", "Splitting it up into windows???", "Pre-Processing???") — windowing and normalization strategy for real (non-synthetic) data is undecided.
- **KNOWN: synthetic scale factors are placeholders.** `synthetic_imu.ACCEL_G_PER_UNIT` (0.4) and `GYRO_DPS_PER_UNIT` (200) convert the generator's unitless channels into g and dps. They were chosen to fill the sensor's full scale without clipping, **not measured from real climbing motion**. Every generated HDF5 records `scale_factors_version` in its root `notes` attr, and `run_pipeline.py` prints it on every run. When real pilot data arrives and these are replaced, expect: clipping warnings if the new scale exceeds ±4 g / ±1000 dps, and collapsed CNN class separation (the classes are separated by variance, so rescaling rescales the learned feature). Timing/CRC/drop-rate symptoms are *not* caused by this — see the `KNOWN: SYNTHETIC SCALE` block in `synthetic_imu.py`.
- **Every real `.BIN` currently in `Data/` is 34-byte v1, not 32-byte v2.** They parse cleanly only because the byte-wise resync walks past the 2-byte mismatch each frame, so frame-aligned `sync_hit_rate` reads ~6% on them. `ingest.py` reports this as `bytes_per_frame_observed` + `frame_alignment_note` rather than letting it read as corruption. A dedicated v1 reader has not been written.
