#!/usr/bin/env python
"""Convert learned diffusion coefficient to physical units."""

from idw_pinn.data.csv_loader import convert_diffusion_coefficient

# === USER INPUT ===
D_learned = 0.000927  # Replace with your value from training output

# Your experimental calibration
pixel_size_um = 54.0    # 0.053 mm = 53 µm per pixel
frame_time_s = 60.0     # 30 seconds per frame

# From your CSV data (300x100 ROI, 8 frames)
metadata = {'x_scale': 299.0, 't_scale': 7.0}

# Target D range (m²/s) - literature values to compare against
D_target_min_m2s = 3.15e-10  # m²/s
D_target_max_m2s = 4.05e-10  # m²/s

# === CONVERSION ===
result = convert_diffusion_coefficient(
    D_learned, 
    metadata,
    pixel_size_um=pixel_size_um,
    frame_time_s=frame_time_s
)

# === REVERSE CALCULATION ===
# D_m2s = D_normalized * (x_scale² / t_scale) * (pixel_size_m)² / frame_time_s
# So: D_normalized = D_m2s * frame_time_s / ((x_scale² / t_scale) * pixel_size_m²)

pixel_size_m = pixel_size_um * 1e-6
scale_factor = (metadata['x_scale']**2 / metadata['t_scale']) * (pixel_size_m**2) / frame_time_s

D_norm_min = D_target_min_m2s / scale_factor
D_norm_max = D_target_max_m2s / scale_factor

# === OUTPUT ===
print("=" * 50)
print("Diffusion Coefficient Unit Conversion")
print("=" * 50)
print(f"D (normalized):    {result['D_normalized']}")
print(f"D (pixel²/frame):  {result['D_pixel_frame']:.4f}")
print(f"D (µm²/s):         {result['D_um2_s']:.2f}")
print(f"D (cm²/s):         {result['D_cm2_s']:.2e}")
D_m2s = result['D_cm2_s'] * 1e-4
print(f"D (m²/s):          {D_m2s / 1e-10:.2f} × 10⁻¹⁰")
print("=" * 50)
print()
print("=" * 50)
print("Target Range")
print("=" * 50)
print(f"Target D range:    {D_target_min_m2s:.2e} - {D_target_max_m2s:.2e} m²/s")
print(f"Required D_norm:   {D_norm_min:.6f} - {D_norm_max:.6f}")
print("=" * 50)