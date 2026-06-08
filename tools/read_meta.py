#!/usr/bin/env python3
"""Tiny metadata reader: prints "key=value" lines from a nabu/ESRF metadata.json.
The ONLY python in the fysics pipeline -- all math is in C. argv[1] = metadata.json."""
import json, sys

m = json.load(open(sys.argv[1]))
scan = m.get("scan", {}); tomo = scan.get("tomo", m.get("tomo", scan))
acq = tomo.get("acquisition", {}); det = acq.get("detector", {})
proc = tomo.get("processing", {}) or {}
phase = (proc.get("preprocessing", {}) or {}).get("phase", {}) or {}
px = det.get("samplePixelSize")
out = {
    "delta_beta": phase.get("delta_beta"),
    "energy_kev": acq.get("energy"),
    "distance_mm": acq.get("sampleDetectorDistance"),
    "pixel_um": (px * 1000.0) if px is not None else None,
    "unsharp_sigma": phase.get("unsharp_sigma"),
    "unsharp_coeff": phase.get("unsharp_coeff"),
    "machine_current_start": acq.get("machineCurrentStart"),
    "machine_current_stop": acq.get("machineCurrentStop"),
    "window_lo": m.get("zarr_export", {}).get("target_window_f32_min"),
    "window_hi": m.get("zarr_export", {}).get("target_window_f32_max"),
}
for k, v in out.items():
    if v is not None:
        print(f"{k}={v}")
