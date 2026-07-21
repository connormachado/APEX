# Claude Code Task: Audit and Fix `lsm6dso.h` Register Constants

## Context

This is firmware for an APEX wearable IMU system on a NUCLEO-H753ZI (STM32H753ZI).
The driver talks to an LSM6DSO IMU over SPI. Several bit-field constants in `lsm6dso.h`
have been found to have incorrect values — the bit positions were wrong, causing
misconfigured registers. We need a systematic audit of every `#define` against the
official LSM6DSO datasheet register map.

The file to audit and fix is:
```
Core/Inc/lsm6dso.h
```

---

## Verified Ground Truth: LSM6DSO Register Map

Use ONLY the values below. Do NOT guess or infer. These come directly from the
LSM6DSO datasheet (DS12140 Rev 3) and application note AN5192 Rev 5.

---

### Register Addresses (all already correct — verify but do not change unless wrong)

| Name            | Address |
|-----------------|---------|
| WHO_AM_I        | 0x0F    |
| CTRL1_XL        | 0x10    |
| CTRL2_G         | 0x11    |
| CTRL3_C         | 0x12    |
| STATUS_REG      | 0x1E    |
| OUTX_L_G        | 0x22    |
| OUTX_H_G        | 0x23    |
| OUTY_L_G        | 0x24    |
| OUTY_H_G        | 0x25    |
| OUTZ_L_G        | 0x26    |
| OUTZ_H_G        | 0x27    |
| OUTX_L_A        | 0x28    |
| OUTX_H_A        | 0x29    |
| OUTY_L_A        | 0x2A    |
| OUTY_H_A        | 0x2B    |
| OUTZ_L_A        | 0x2C    |
| OUTZ_H_A        | 0x2D    |
| INT1_CTRL       | 0x0D    |

---

### CTRL1_XL (0x10) — Accelerometer config
Bit layout: `[ODR_XL3 | ODR_XL2 | ODR_XL1 | ODR_XL0 | FS1_XL | FS0_XL | LPF2_XL_EN | 0]`
- ODR bits are [7:4]
- FS bits are [3:2] (NOT [4:3] or [5:4])

ODR values (bits [7:4]):

Correct ODR_XL encoding (Table 44 from datasheet screenshot provided by user):
- 104 Hz: ODR_XL[3:0] = 0100 → bits [7:4] = 0100 → **0x40** ✓

FS_XL values (bits [3:2]):
| FS      | FS[1:0] | Bits [3:2] | Hex  |
|---------|---------|------------|------|
| ±2g     | 00      | 0b0000     | 0x00 |
| ±16g    | 01      | 0b0100     | 0x04 |
| ±4g     | 10      | 0b1000     | 0x08 |
| ±8g     | 11      | 0b1100     | 0x0C |

**Therefore: FS_XL_4G = 0x08** (NOT 0x10 — that was wrong, bit 4 is LPF2_XL_EN territory)

---

### CTRL2_G (0x11) — Gyroscope config
Bit layout: `[ODR_G3 | ODR_G2 | ODR_G1 | ODR_G0 | FS1_G | FS0_G | FS_125 | 0]`
- ODR bits are [7:4]
- FS bits are [3:1]: FS1_G=[3], FS0_G=[2], FS_125=[1]
  - FS_125=1 selects ±125 dps regardless of FS1/FS0
  - For standard ranges, FS_125=0 and use FS[1:0] at bits [3:2]

ODR_G values (bits [7:4]) — same encoding as accel:
- 104 Hz: ODR_G[3:0] = 0100 → **0x40** ✓

FS_G values (FS_125=0, bits [3:2]):
| FS        | FS1_G FS0_G | Bits [3:2] | Hex  |
|-----------|-------------|------------|------|
| ±250 dps  | 00          | 0b0000     | 0x00 |
| ±500 dps  | 01          | 0b0100     | 0x04 |
| ±1000 dps | 10          | 0b1000     | 0x08 |
| ±2000 dps | 11          | 0b1100     | 0x0C |

**Therefore: FS_G_1000DPS = 0x08** (NOT 0x04 — that was ±500 dps)

---

### CTRL3_C (0x12) — Control register 3
Bit layout: `[BOOT | BDU | H_LACTIVE | PP_OD | SIM | IF_INC | 0 | SW_RESET]`

| Bit name  | Bit position | Hex value |
|-----------|-------------|-----------|
| SW_RESET  | bit 0       | 0x01      |
| IF_INC    | bit 2       | 0x04      |
| BDU       | bit 6       | 0x40      |
| BOOT      | bit 7       | 0x80      |

All three currently in the header match. Verify SW_RESET=0x01, IF_INC=0x04, BDU=0x40.

---

### STATUS_REG (0x1E)
Bit layout: `[0 | 0 | 0 | 0 | 0 | TDA | GDA | XLDA]`

| Flag | Bit | Hex  |
|------|-----|------|
| XLDA | 0   | 0x01 |
| GDA  | 1   | 0x02 |
| TDA  | 2   | 0x04 |

Currently: XLDA=0x01, GDA=0x02 — both correct. Verify.

---

### SPI Protocol Masks
- READ_MASK = 0x80 (set bit 7 to indicate read) ✓
- WRITE_MASK = 0x7F (clear bit 7 to indicate write) ✓

---

## Task

1. **Open** `Core/Inc/lsm6dso.h`

2. **Audit every `#define`** in the "Bit Masks for configuration" section against the
   ground truth table above. For each constant, state:
   - Current value in the file
   - Correct value from datasheet
   - Whether it needs to change (YES/NO)
   - If YES: why (which bits are wrong)

3. **Fix all incorrect values** in-place. Add a comment next to each fixed line
   explaining the bit field, e.g.:
   ```c
   #define FS_XL_4G (0x08)    // FS[1:0]=10 at bits[3:2] of CTRL1_XL → 0b00001000
   #define FS_G_1000DPS (0x08) // FS[1:0]=10 at bits[3:2] of CTRL2_G → 0b00001000
   ```

4. **Also update the sensitivity constants** in the header (if they exist — add them
   if not) to match the corrected full-scale settings:
   ```c
   // Physical unit conversion (LSM6DSO datasheet Table 2)
   #define ACCEL_SENSITIVITY_MG_PER_LSB  (0.122f)  // mg/LSB at ±4g
   #define GYRO_SENSITIVITY_MDPS_PER_LSB (35.0f)   // mdps/LSB at ±1000 dps
   ```

5. **Do NOT touch** anything outside of `lsm6dso.h`. Do not modify `.c` files,
   CMakeLists, or any other file.

6. **Output a summary** at the end listing:
   - Every constant that was wrong and what it was changed to
   - Every constant that was already correct
   - The final composed byte values for CTRL1_XL and CTRL2_G as a sanity check
     (e.g., "CTRL1_XL will be written as 0x48 = 0b01001000: ODR=104Hz, FS=±4g")

---

## Expected final values summary (use to verify your work)

| Constant          | Expected value | Notes                              |
|-------------------|---------------|------------------------------------|
| ODR_XL_104HZ      | 0x40          | ODR[3:0]=0100 in bits[7:4]        |
| FS_XL_4G          | 0x08          | FS[1:0]=10 in bits[3:2]           |
| ODR_G_104HZ       | 0x40          | ODR[3:0]=0100 in bits[7:4]        |
| FS_G_1000DPS      | 0x08          | FS[1:0]=10 in bits[3:2]           |
| SW_RESET          | 0x01          | bit 0 of CTRL3_C                  |
| IF_INC            | 0x04          | bit 2 of CTRL3_C                  |
| BDU               | 0x40          | bit 6 of CTRL3_C                  |
| READ_MASK         | 0x80          | bit 7 set = read operation        |
| WRITE_MASK        | 0x7F          | bit 7 clear = write operation     |
| XLDA              | 0x01          | bit 0 of STATUS_REG               |
| GDA               | 0x02          | bit 1 of STATUS_REG               |

Composed register bytes:
- CTRL1_XL = ODR_XL_104HZ | FS_XL_4G = 0x40 | 0x08 = **0x48** = `0100 1000`
- CTRL2_G  = ODR_G_104HZ  | FS_G_1000DPS   = 0x40 | 0x08 = **0x48** = `0100 1000`
- CTRL3_C  = BDU | IF_INC = 0x40 | 0x04 = **0x44** = `0100 0100`