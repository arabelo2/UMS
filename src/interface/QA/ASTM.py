#!/usr/bin/env python3

"""
ASTM E2491 Type B Phased-Array Assessment Block - schematic generator
Author: ChatGPT (OpenAI o3) - 2025-05-03

Edit the “USER‑CONFIGURABLE PARAMETERS” section to match a supplier-specific
drawing (hole counts, pitches, arc radii, etc.).  Dimensions are in inches.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
  
# Import services (assuming these are defined elsewhere)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --------------------------------------------------------------------------
# USER‑CONFIGURABLE PARAMETERS
# --------------------------------------------------------------------------
BLOCK_W, BLOCK_H = 6.0, 4.0          # overall block outline (in)

# Radial hole arcs (centres & radii)
ARC_CENTER = (1.0, 2.0)              # X, Y of arc centre
ARC_RADII  = [1.0, 2.0]              # two radii (in)
ARC_DSTART, ARC_DEND, ARC_STEP = -45, 45, 5   # start, end, step angles (°)

# Vertical column: 16 holes @ 0.120‑in pitch, starting 0.5 in above bottom
COL_X          = 4.0                 # X‑location of column
COL_Y_START    = 0.5
COL_PITCH      = 0.120
COL_COUNT      = 16

# Angled row: 12 holes @ 0.200‑in pitch on a 30° slope
ROW_START      = np.array([1.0, 0.5])          # first hole location
ROW_SLOPE_DEG  = 30.0
ROW_PITCH      = 0.200
ROW_COUNT      = 12

# Four side‑drilled‑hole (SDH) centrelines (just for illustration)
SDH_ORIGIN     = (0.0, 2.0)          # where the SDHs emerge on the side
SDH_ANGLES_DEG = [30, 45, 60, 75]    # exit angles (°)
SDH_VIS_LEN    = 1.5                 # visible length of dashed lines (in)

# --------------------------------------------------------------------------
# DERIVED GEOMETRY
# --------------------------------------------------------------------------
# Radial arcs --------------------------------------------------------------
arc_angles = np.arange(ARC_DSTART, ARC_DEND + ARC_STEP, ARC_STEP)
arc_t      = np.deg2rad(arc_angles)                        # radians

radial_points = []
for r in ARC_RADII:
    x = ARC_CENTER[0] + r * np.cos(arc_t)
    y = ARC_CENTER[1] + r * np.sin(arc_t)
    radial_points.append((x, y))

# Vertical column ----------------------------------------------------------
col_y = COL_Y_START + COL_PITCH * np.arange(COL_COUNT)
col_x = np.full_like(col_y, COL_X)

# Angled row ---------------------------------------------------------------
row_dir = np.array([np.cos(np.deg2rad(ROW_SLOPE_DEG)),
                    np.sin(np.deg2rad(ROW_SLOPE_DEG))])
row_pts = np.array([ROW_START + i * ROW_PITCH * row_dir
                    for i in range(ROW_COUNT)])
row_x, row_y = row_pts[:, 0], row_pts[:, 1]

# SDH centrelines (dashed) -------------------------------------------------
sdh_lines = []
for ang in SDH_ANGLES_DEG:
    θ = np.deg2rad(ang)
    end_x = SDH_ORIGIN[0] + SDH_VIS_LEN * np.cos(θ)
    end_y = SDH_ORIGIN[1] + SDH_VIS_LEN * np.sin(θ)
    sdh_lines.append(((SDH_ORIGIN[0], end_x), (SDH_ORIGIN[1], end_y)))

# --------------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_aspect('equal', adjustable='box')

# Block outline
outline_x = [0, BLOCK_W, BLOCK_W, 0, 0]
outline_y = [0, 0, BLOCK_H, BLOCK_H, 0]
ax.plot(outline_x, outline_y, lw=2, color='orange')

# Radial arcs
markers = ['o', 'o']               # one marker style per radius
for (x, y), m in zip(radial_points, markers):
    ax.scatter(x, y, s=35, marker=m)

# Vertical column
ax.scatter(col_x, col_y, s=35, marker='o')

# Angled row
ax.scatter(row_x, row_y, s=35, marker='o')

# SDH dashed lines
for (x_pair, y_pair) in sdh_lines:
    ax.plot(x_pair, y_pair, ls='-')

# Labels & cosmetics
ax.set_xlim(-0.5, BLOCK_W + 0.5)
ax.set_ylim(-0.5, BLOCK_H + 0.5)
ax.set_xlabel("Width (in)")
ax.set_ylabel("Height (in)")
ax.set_title("ASTM E2491 Type B Phased-Array Assessment Block (schematic)")
ax.grid(True, ls=':')

plt.tight_layout()
plt.show()
