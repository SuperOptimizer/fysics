# Empirical analysis of real ESRF/BM18 PHerc0139 data

Analysis run 2026-06-05 on real volumes pulled from `s3://vesuvius-challenge-open-data`
(PHerc0139, the only scroll with the full resolution range incl. the finest 1.129µm voxel
in the bucket). Central, ~99–100% occupied papyrus cubes (128³ each) at four resolutions.
This is the *measured* ground truth that the literature review and pipeline audit should be
checked against — it overrides assumptions baked into the code from earlier visual tuning.

## 1. What the metadata actually says (per-volume, NOT constant)

`metadata.json` exists on every volume. The reconstruction parameters vary **per volume**
and must be read from metadata — our hardcoded `delta_beta=1000, unsharp_sigma=1.2` are wrong
for the fine scan:

| volume   | delta_beta | unsharp σ | unsharp coeff | energy | dist  | pixel(µm) | recon | beam-drift |
|----------|-----------:|----------:|--------------:|-------:|------:|----------:|-------|-----------:|
| 1.129µm  | **500**    | **2.5**   | 4.0           | 59 keV | 200mm | 1.129     | GHBP  | (n/a)      |
| 2.399µm  | 1000       | 1.2       | 4.0           | 78 keV | 220mm | 2.399     | GHBP  | 1.5%       |
| 2.403µm  | 1000       | 1.2       | 4.0           | 77 keV | 220mm | 2.403     | GHBP  | 4.8%       |
| 9.362µm  | 1000       | 1.2       | 4.0           | 113keV | 1200mm| 9.362     | GHBP  | **13.7%**  |

Findings:
- **delta_beta and unsharp_sigma are NOT constant** (500/2.5 at 1.129µm vs 1000/1.2 elsewhere).
  The deconv MUST read these from metadata.json, never assume.
- **Reconstruction is GHBP** (Gridded Hierarchical BackProjection), not plain FBP. The README's
  "FBP ramp" language should say GHBP.
- **Beam-current drift is in the metadata** (`machineCurrentStart/Stop`): 1.5%→13.7%. Our zdrift
  correction is justified, and its magnitude can be *auto-set* from these two numbers rather than
  estimated blind.
- Acquisition is **helical half-acquisition**, and the fine volume is a **19-tile fused mosaic**
  (T0 + ring of 6 + ring of 12). Tile seams are a real artifact class we do **not** yet address.
- Air masking is **SAM2 (a fine-tuned neural net)**, `mask_scale` 8–32 — not an intensity
  threshold. This is why our Otsu air-threshold disagrees with the "masked" data: the ground-truth
  mask is semantic, not intensity-based.
- The u8/u16 we read is a **known linear window of f32 attenuation** (`target_window_f32_min/max`,
  `window_u16_min/max`), and the window **differs per volume** (f32 max 0.145 / 0.21 / 0.19 /
  0.145; u16 start 0 / 5276 / 4752 / 0). So a fixed intensity threshold cannot be physical across
  volumes — thresholds must be per-volume (Otsu) or derived from this window. The f32 physical
  attenuation is recoverable by inverting the window.

## 2. Resolution / spectrum (radial power spectrum of each cube)

Normalized radial PSD (DC-removed), value at fractional Nyquist:

| volume  | 0.1Nyq | 0.2  | 0.3  | 0.5  | 0.7  | 0.9  | signal→noise crossover |
|---------|-------:|-----:|-----:|-----:|-----:|-----:|------------------------|
| 1.129µm | 1.4e-3 |9.4e-5|1.9e-5|2.7e-6|8.4e-7|4.3e-7| ~0.5 Nyq               |
| 2.399µm | 2.2e-3 |2.9e-4|4.8e-5|2.6e-6|5.4e-7|2.5e-7| ~0.5 Nyq               |
| 2.403µm | 1.3e-2 |1.3e-3|2.6e-4|1.1e-5|1.9e-6|9.0e-7| ~0.5–0.6 Nyq           |
| 9.362µm | 2.8e-2 |7.9e-3|2.2e-3|1.6e-4|1.1e-5|2.8e-6| ~0.7 Nyq               |

- The PSD falls **~4 orders of magnitude** from 0.1→0.5 Nyquist, then **flattens into a noise
  floor**. That flat tail IS the noise — there is no recoverable signal above ~0.5 Nyquist on the
  fine scans. This is the empirical ceiling on deconvolution: lift the 0.1–0.5 Nyq band, do NOT
  push past it.
- The finer the voxel, the **lower** the fractional cutoff (1.129µm signal dies by ~0.10 Nyq,
  9.362µm carries to ~0.25). I.e. the fine scans are more oversampled relative to their grid —
  more empty high-frequency band, so deconvolution has genuine headroom there.
- Per-voxel noise scales with voxel size (Laplacian-MAD ≈ 0.66 / 1.99 / 2.65 / 5.30 for
  1.129/2.399/2.403/9.362) — finer voxels are *cleaner* per-voxel (more integrated photons).

## 3. Running fysics deconv on the real cubes (default reg=0.015, metadata params)

Spectral gain (out/in PSD) after `fy_deconvolve`:

| volume  | gain @0.2Nyq | @0.4Nyq | @0.7Nyq | verdict |
|---------|-------------:|--------:|--------:|---------|
| 1.129µm | 13.5×        | 2.5×    | **0.3×**| ✅ lifts mid-band, suppresses noise tail |
| 2.399µm | 16.1×        | 7.8×    | 1.9×    | ✅ good |
| 2.403µm | 16.1×        | 7.7×    | 1.9×    | ✅ good |
| 9.362µm | 6.3×         | 13.9×   | **14.5×**| ❌ AMPLIFIES the noise tail 14× |

**Discovered bug:** with the default `reg=0.015`, the 9.362µm deconv has gain *increasing* with
frequency and blows the noise floor up 14.5× at 0.7 Nyq. Physical cause: the Paganin filter
`1/(1+δβ·λ·D·π·f²)` is far stronger at 9.362µm (D=1200mm vs 200–220mm, ~6× larger), so its
Wiener inverse over-amplifies high-f where only noise lives.

reg sweep on 9.362µm (gain @0.4 / @0.7 Nyq):

| reg   | @0.4Nyq | @0.7Nyq | out std |
|-------|--------:|--------:|--------:|
| 0.015 | 13.9×   | 14.5×   | 0.417   |
| 0.05  | 4.9×    | 2.2×    | 0.301   |
| **0.10**| **1.9×**| **0.6×**| 0.230 |
| 0.20  | 0.6×    | 0.2×    | 0.165   |

**Fix direction:** `reg` must scale with the recon-filter strength (≈ δβ·λ·D / pixel²), not be a
constant. For 9.362µm the noise-safe reg is ~0.1 (≈7× the default). A metadata-driven
`reg = base · f(δβ, λ, D, pixel)` would self-tune. Until then, fysics should at minimum warn /
raise the floor when D·δβ is large.

## Action items for fysics (from this analysis)

1. **Read deconv params from metadata.json** per volume — stop hardcoding δβ/σ. (s3-side schema:
   `tomo.processing.preprocessing.phase` on rich files, `scan.tomo.processing...` on thin ones.)
2. **Auto-scale `reg`** with filter strength (δβ·λ·D/pixel²) so coarse/long-distance volumes don't
   amplify noise. Validate that the deconv gain stays ≤1 beyond the ~0.5-Nyq noise crossover.
3. **Cap deconv at the empirical noise crossover (~0.5 Nyq for fine scans)** — there is no signal
   to recover above it; lifting it is pure noise gain.
4. **Per-volume intensity handling**: thresholds/normalization from the f32↔u16 window in metadata,
   or Otsu — never a global constant. (Confirms earlier finding.)
5. **zdrift magnitude can be seeded** from `machineCurrentStart/Stop` instead of estimated blind.
6. **New artifact class to consider: mosaic tile seams** (19-tile fused volumes). Not yet handled.
7. The ground-truth air mask is **SAM2-semantic**, not intensity — our intensity mask will always
   differ at thin/faint papyrus edges. Document this; don't chase the SAM2 boundary with a threshold.

## Reproduce

Data + scripts under `superresolution/analysis/PHerc0139/` (not committed; data is CC BY-NC):
`fetch_cubes.py` (pull occupied 128³ cubes), `analyze.py` / `analyze2.py` / `analyze3.py`
(spectrum/noise/drift), `deconv_test.py` + `reg_sweep.py` (fysics deconv via ctypes),
`viz.py` (overview figure).
