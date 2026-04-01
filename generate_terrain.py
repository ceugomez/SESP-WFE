"""
generate_terrain.py
-------------------
Synthesises an idealised complex terrain for FastEddy ensemble runs and
writes it to a FastEddy binary terrain file.

Terrain design: asymmetric ridge-valley system
  - Main ridge    : N-S elongated Gaussian, H=200 m
                    sigma_x (E-W, cross-ridge) = 450 m
                    sigma_y (N-S, along-ridge) = 1200 m
                    Centre at (0.38*Lx, 0.50*Ly)
  - Satellite knoll: isotropic Gaussian, H=100 m, sigma=350 m
                    Centre at (0.55*Lx, 0.63*Ly)

Output binary format: [int Nx][int Ny][float32 topo(Ny, Nx) row-major].
"""

import numpy as np
import struct
import os

# ---------------------------------------------------------------------------
# Grid parameters – must match the FastEddy namelist
# ---------------------------------------------------------------------------
Nx = 242
Ny = 242
dx = 20.0   # m
dy = 20.0   # m

x = np.arange(Nx) * dx          # 0 … 4785 m
y = np.arange(Ny) * dy
Lx = Nx * dx
Ly = Ny * dy

xx, yy = np.meshgrid(x, y)      # shape (Ny, Nx)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def gaussian_hill(xx, yy, cx, cy, H, sx, sy):
    """Axis-aligned Gaussian hill with independent x/y half-widths."""
    return H * np.exp(-0.5 * ((xx - cx)**2 / sx**2 + (yy - cy)**2 / sy**2))

def create_1d_taper(N, margin):
    """Creates a smooth cosine taper from 0 to 1 and back to 0."""
    taper = np.ones(N, dtype=np.float32)
    if margin > 0:
        tx = np.linspace(0, np.pi, margin)
        curve = 0.5 * (1.0 - np.cos(tx))
        taper[:margin] = curve
        taper[-margin:] = curve[::-1]
    return taper

def print_ascii_profile(z_slice, width=70, height=12, y_label=""):
    """Prints a terminal-friendly ASCII cross-section of the terrain."""
    # Downsample the 1D slice to fit the requested terminal width
    idx = np.linspace(0, len(z_slice)-1, width, dtype=int)
    profile = z_slice[idx]
    max_z = np.max(profile)
    
    print(f"\n--- X-Z Elevation Profile ({y_label}) ---")
    for r in range(height, 0, -1):
        threshold = (r / height) * max_z
        row_str = ""
        for z in profile:
            if z >= threshold:
                row_str += "█"
            elif z >= threshold - (max_z/height)*0.5:
                row_str += "▄" # Half block for smoother edges
            else:
                row_str += " "
        print(f"{threshold:>5.0f} m | {row_str}")
    print("    0 m |" + "▀" * width)
    print("         " + "West".ljust(width-4) + "East")

# ---------------------------------------------------------------------------
# Build terrain
# ---------------------------------------------------------------------------
# Main ridge – elongated N-S, offset slightly west of centre
h_ridge = gaussian_hill(xx, yy,
                        cx=0.38 * Lx, cy=0.50 * Ly,
                        H=200.0,
                        sx=450.0,   
                        sy=1200.0)  

# Satellite knoll – isotropic, northeast of main ridge
h_knoll = gaussian_hill(xx, yy,
                        cx=0.55 * Lx, cy=0.63 * Ly,
                        H=100.0,
                        sx=350.0,
                        sy=350.0)

topo_raw = h_ridge + h_knoll

# Apply edge taper to force periodic boundaries to exactly 0.0
margin_cells = 15
taper_x = create_1d_taper(Nx, margin_cells)
taper_y = create_1d_taper(Ny, margin_cells)
mask_2d = np.outer(taper_y, taper_x)

topo = topo_raw * mask_2d

# ---------------------------------------------------------------------------
# Summary and ASCII Art
# ---------------------------------------------------------------------------
print(f"Terrain summary:")
print(f"  Domain   : {Lx:.0f} m x {Ly:.0f} m  (Nx={Nx}, Ny={Ny}, dx={dx} m)")
print(f"  z_min    : {topo.min():.5f} m  <-- Boundary forced to zero")
print(f"  z_max    : {topo.max():.1f} m")
print(f"  Max slope: {np.max(np.sqrt(np.gradient(topo, dx, axis=1)**2 + np.gradient(topo, dy, axis=0)**2)):.3f} (dz/dh)")

# Render a slice straight through the middle of the main ridge
y_center_idx = int(0.50 * Ny)
print_ascii_profile(topo[y_center_idx, :], width=70, height=12, y_label=f"y = {y[y_center_idx]:.0f} m")

# ---------------------------------------------------------------------------
# Write FastEddy binary terrain file
# ---------------------------------------------------------------------------
outdir = os.path.dirname(os.path.abspath(__file__))
outfile = os.path.join(outdir, "terrain.bin") 

with open(outfile, "wb") as f:
    f.write(struct.pack("i", Nx))
    f.write(struct.pack("i", Ny))
    f.write(topo.astype("f4").tobytes()) 

print(f"\nWrote: {outfile}")