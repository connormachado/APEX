#!/usr/bin/env python3
"""
APEX Real-Time IMU Visualizer
─────────────────────────────
Reads LSM6DSO data from the Nucleo's serial output, converts raw counts
to physical units, fuses accel+gyro through a Madgwick AHRS filter, and
displays:
    • A 3D orientation view (body-frame axes rotating in world frame)
    • 6 strip charts (gyro XYZ in dps, accel XYZ in g)

Usage:
    python apex_visualizer.py                       # auto-detect port
    python apex_visualizer.py /dev/tty.usbmodem1103 # explicit port

Dependencies (install inside your venv):
    pip install pyserial matplotlib numpy

IMPORTANT: Close CoolTerm first — only one program can hold the serial
port at a time.

Controls:
    Close the matplotlib window to quit.
    The 3D view auto-rotates with the IMU.
"""

import sys
import re
import math
import time
from collections import deque

import numpy as np
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401 (needed for 3d projection)
import matplotlib.animation as animation


# ═══════════════════════════════════════════════════════════════════
# LSM6DSO Scale Factors (from datasheet, Table 2 & 3)
# ═══════════════════════════════════════════════════════════════════
#   ±4g  accel  →  0.122 mg  per LSB
#   ±1000 dps gyro → 35.0 mdps per LSB

ACCEL_SCALE = 0.122e-3      # raw LSB → g
GYRO_SCALE  = 35.0e-3       # raw LSB → dps
DEG2RAD     = math.pi / 180.0


# ═══════════════════════════════════════════════════════════════════
# Madgwick AHRS Filter
# ═══════════════════════════════════════════════════════════════════
# Fuses accelerometer (noisy but knows "down") with gyroscope (smooth
# but drifts) into a single orientation quaternion.
#
# beta controls the tradeoff:
#   higher → trust accel more → less drift but more jitter
#   lower  → trust gyro more  → smoother but may drift
#
# For validation on a bench at ~16 Hz, beta=0.1 is a good starting
# point.  Increase to 0.3–0.5 if you see slow drift when the sensor
# is stationary.

class MadgwickAHRS:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])   # [w, x, y, z]
        self._last_time = None

    def update(self, gyro_rads, accel_g):
        """
        Call once per sample.
        gyro_rads : [gx, gy, gz] in rad/s
        accel_g   : [ax, ay, az] in g  (will be normalised internally)
        """
        now = time.time()
        if self._last_time is None:
            self._last_time = now
            return
        dt = now - self._last_time
        self._last_time = now
        if dt <= 0 or dt > 1.0:
            return      # skip bogus intervals

        q0, q1, q2, q3 = self.q
        gx, gy, gz = gyro_rads
        ax, ay, az = accel_g

        # ── normalise accel ──
        norm_a = math.sqrt(ax*ax + ay*ay + az*az)
        if norm_a < 1e-10:
            return
        ax /= norm_a;  ay /= norm_a;  az /= norm_a

        # ── objective function (accel vs expected gravity) ──
        f1 = 2.0*(q1*q3 - q0*q2) - ax
        f2 = 2.0*(q0*q1 + q2*q3) - ay
        f3 = 2.0*(0.5 - q1*q1 - q2*q2) - az

        # ── Jacobian^T · f  →  gradient step ──
        s0 = -2.0*q2*f1 + 2.0*q1*f2
        s1 =  2.0*q3*f1 + 2.0*q0*f2 - 4.0*q1*f3
        s2 = -2.0*q0*f1 + 2.0*q3*f2 - 4.0*q2*f3
        s3 =  2.0*q1*f1 + 2.0*q2*f2

        norm_s = math.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3)
        if norm_s > 1e-10:
            s0 /= norm_s;  s1 /= norm_s;  s2 /= norm_s;  s3 /= norm_s

        # ── quaternion rate from gyroscope ──
        qd0 = 0.5*(-q1*gx - q2*gy - q3*gz)
        qd1 = 0.5*( q0*gx + q2*gz - q3*gy)
        qd2 = 0.5*( q0*gy - q1*gz + q3*gx)
        qd3 = 0.5*( q0*gz + q1*gy - q2*gx)

        # ── integrate ──
        q0 += (qd0 - self.beta*s0) * dt
        q1 += (qd1 - self.beta*s1) * dt
        q2 += (qd2 - self.beta*s2) * dt
        q3 += (qd3 - self.beta*s3) * dt

        # ── normalise quaternion ──
        norm_q = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        self.q = np.array([q0/norm_q, q1/norm_q, q2/norm_q, q3/norm_q])

    def rotation_matrix(self):
        """3×3 rotation matrix from body frame → world frame."""
        w, x, y, z = self.q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
            [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
            [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])

    def euler_deg(self):
        """Roll, pitch, yaw in degrees."""
        w, x, y, z = self.q
        roll  = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
        pitch = math.degrees(math.asin(max(-1, min(1, 2*(w*y - z*x)))))
        yaw   = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
        return roll, pitch, yaw


# ═══════════════════════════════════════════════════════════════════
# Serial helpers
# ═══════════════════════════════════════════════════════════════════

_LINE_RE = re.compile(r'-?\d+')

def parse_imu_line(line):
    """
    Parse the firmware's printf format:
      'Gyro X: 123, Gyro Y: 456, Gyro Z: ..., Accel X: ..., Accel Y: ..., Accel Z: ...'
    Returns [gx, gy, gz, ax, ay, az] as ints, or None.
    """
    nums = _LINE_RE.findall(line)
    if len(nums) == 6:
        return [int(n) for n in nums]
    return None


def find_serial_port():
    """Auto-detect the Nucleo's USB serial port on macOS."""
    for p in serial.tools.list_ports.comports():
        low = p.device.lower()
        if 'usbmodem' in low or 'stlink' in low:
            return p.device
    # nothing obvious — list what we see
    print("Could not auto-detect port.  Available:")
    for p in serial.tools.list_ports.comports():
        print(f"  {p.device}  —  {p.description}")
    return None


# ═══════════════════════════════════════════════════════════════════
# Visualiser
# ═══════════════════════════════════════════════════════════════════

HIST_LEN = 300   # samples to show in strip charts (~18 s at 16 Hz)

# body-frame unit vectors (columns of identity matrix)
BODY_X = np.array([1, 0, 0], dtype=float)
BODY_Y = np.array([0, 1, 0], dtype=float)
BODY_Z = np.array([0, 0, 1], dtype=float)


def main():
    # ── serial ──
    port = sys.argv[1] if len(sys.argv) > 1 else find_serial_port()
    if port is None:
        print("Usage:  python apex_visualizer.py /dev/tty.usbmodemXXXX")
        sys.exit(1)

    print(f"Opening {port} …  (close CoolTerm first if you get 'port busy')")
    ser = serial.Serial(port, 115200, timeout=0.005)
    ser.reset_input_buffer()
    print("Connected.\n")

    # ── state ──
    ahrs = MadgwickAHRS(beta=0.15)
    gyro_hist  = [deque([0]*HIST_LEN, maxlen=HIST_LEN) for _ in range(3)]
    accel_hist = [deque([0]*HIST_LEN, maxlen=HIST_LEN) for _ in range(3)]
    line_buf = ""

    # ── figure layout ──
    #  Row 0 : 3D orientation           (spans all 3 cols)
    #  Row 1 : Gyro  X | Y | Z  strip charts
    #  Row 2 : Accel X | Y | Z  strip charts
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle("APEX  IMU  Visualiser", fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1],
                          hspace=0.35, wspace=0.30)

    # ── 3D axes ──
    ax3d = fig.add_subplot(gs[0, :], projection='3d')
    ax3d.set_xlim(-1.2, 1.2);  ax3d.set_ylim(-1.2, 1.2);  ax3d.set_zlim(-1.2, 1.2)
    ax3d.set_xlabel('X');  ax3d.set_ylabel('Y');  ax3d.set_zlabel('Z')
    ax3d.set_title('Orientation  (Madgwick)')
    # draw faint world-frame reference lines
    for axis_vec, c in [([1,0,0],'r'), ([0,1,0],'g'), ([0,0,1],'b')]:
        ax3d.plot([0, axis_vec[0]*1.1], [0, axis_vec[1]*1.1],
                  [0, axis_vec[2]*1.1], c+'--', alpha=0.25, linewidth=1)

    # quiver arrows for body axes (will be updated each frame)
    origin = np.zeros(3)
    quivers = []
    for color in ['red', 'green', 'blue']:
        q = ax3d.quiver(0, 0, 0, 1, 0, 0, color=color, linewidth=2.5,
                        arrow_length_ratio=0.12)
        quivers.append(q)

    euler_text = ax3d.text2D(0.02, 0.95, '', transform=ax3d.transAxes,
                             fontsize=10, family='monospace',
                             verticalalignment='top')

    # ── strip chart axes ──
    strip_axes = []
    strip_lines = []
    labels_gyro  = ['Gyro X (dps)', 'Gyro Y (dps)', 'Gyro Z (dps)']
    labels_accel = ['Accel X (g)',  'Accel Y (g)',  'Accel Z (g)']
    colors_gyro  = ['#e74c3c', '#27ae60', '#2980b9']
    colors_accel = ['#e67e22', '#8e44ad', '#16a085']

    for col in range(3):
        # gyro row
        ax_g = fig.add_subplot(gs[1, col])
        ax_g.set_ylabel(labels_gyro[col], fontsize=9)
        ax_g.set_xlim(0, HIST_LEN)
        ax_g.set_ylim(-500, 500)
        ax_g.tick_params(labelsize=8)
        ln_g, = ax_g.plot([], [], color=colors_gyro[col], linewidth=0.8)
        strip_axes.append(ax_g)
        strip_lines.append(ln_g)

        # accel row
        ax_a = fig.add_subplot(gs[2, col])
        ax_a.set_ylabel(labels_accel[col], fontsize=9)
        ax_a.set_xlim(0, HIST_LEN)
        ax_a.set_ylim(-4, 4)
        ax_a.tick_params(labelsize=8)
        ln_a, = ax_a.plot([], [], color=colors_accel[col], linewidth=0.8)
        strip_axes.append(ax_a)
        strip_lines.append(ln_a)

    x_indices = np.arange(HIST_LEN)

    # ── animation update ──
    def update(frame):
        nonlocal line_buf

        # drain serial buffer — process every complete line
        try:
            raw_bytes = ser.read(ser.in_waiting or 1)
            line_buf += raw_bytes.decode('ascii', errors='ignore')
        except Exception:
            pass

        while '\n' in line_buf:
            line, line_buf = line_buf.split('\n', 1)
            parsed = parse_imu_line(line.strip())
            if parsed is None:
                continue
            gx_r, gy_r, gz_r, ax_r, ay_r, az_r = parsed

            # scale
            gx_dps = gx_r * GYRO_SCALE
            gy_dps = gy_r * GYRO_SCALE
            gz_dps = gz_r * GYRO_SCALE
            ax_g   = ax_r * ACCEL_SCALE
            ay_g   = ay_r * ACCEL_SCALE
            az_g   = az_r * ACCEL_SCALE

            # history
            for i, v in enumerate([gx_dps, gy_dps, gz_dps]):
                gyro_hist[i].append(v)
            for i, v in enumerate([ax_g, ay_g, az_g]):
                accel_hist[i].append(v)

            # Madgwick
            ahrs.update(
                [gx_dps*DEG2RAD, gy_dps*DEG2RAD, gz_dps*DEG2RAD],
                [ax_g, ay_g, az_g],
            )

        # ── update 3D quivers ──
        R = ahrs.rotation_matrix()
        body_vecs = [R @ BODY_X, R @ BODY_Y, R @ BODY_Z]
        for i, q_artist in enumerate(quivers):
            # matplotlib has no quiver.set_data for 3D — remove & redraw
            q_artist.remove()
        quivers.clear()
        for bv, color in zip(body_vecs, ['red', 'green', 'blue']):
            q = ax3d.quiver(0, 0, 0, bv[0], bv[1], bv[2],
                            color=color, linewidth=2.5,
                            arrow_length_ratio=0.12)
            quivers.append(q)

        roll, pitch, yaw = ahrs.euler_deg()
        euler_text.set_text(
            f"Roll  {roll:+7.1f}°\nPitch {pitch:+7.1f}°\nYaw   {yaw:+7.1f}°"
        )

        # ── update strip charts ──
        for col in range(3):
            g_line = strip_lines[col*2]
            a_line = strip_lines[col*2 + 1]
            g_line.set_data(x_indices, list(gyro_hist[col]))
            a_line.set_data(x_indices, list(accel_hist[col]))

            # auto-scale Y to data range (with some padding)
            g_data = list(gyro_hist[col])
            a_data = list(accel_hist[col])
            if g_data:
                g_min, g_max = min(g_data), max(g_data)
                pad = max(abs(g_max - g_min) * 0.15, 10)
                strip_axes[col*2].set_ylim(g_min - pad, g_max + pad)
            if a_data:
                a_min, a_max = min(a_data), max(a_data)
                pad = max(abs(a_max - a_min) * 0.15, 0.1)
                strip_axes[col*2 + 1].set_ylim(a_min - pad, a_max + pad)

        return strip_lines + quivers + [euler_text]

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False,
                                  cache_frame_data=False)
    plt.show()

    ser.close()
    print("Done.")


if __name__ == "__main__":
    main()