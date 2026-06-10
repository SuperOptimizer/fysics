# fysics

Fast CPU physics kernels for Vesuvius Challenge scroll volumes.

Pure modern C (no SIMD intrinsics — clean auto-vectorizable code), CPU-only,
dependency-free. Inverts the **known ESRF/nabu reconstruction operators** (Paganin
single-distance phase-retrieval low-pass + unsharp mask) to restore **contrast and
sharpness** in reconstructed volumes — a physics deblur with no machine learning and
no hallucination, since it analytically inverts a known transfer function.

**Honest scope (FRC-verified):** this is **contrast / sharpness / normalization
restoration**, *not* super-resolution. Poisson-thinning FRC across 7+ volumes shows
the deblur amplifies the high-frequency band ~2.5–3× but produces **no SNR-limited
resolution gain** — it restores the high-frequency *contrast* the Paganin low-pass
suppressed (an 8–15× mid-band power lift), it does not recover detail below the noise
floor. That contrast/legibility restoration is genuinely useful for viewing and for
consistent downstream input; just don't expect new resolvable structure.

Intended for import by viewers/tools (e.g. **vc3d**) as on-the-fly post-processing,
or as batch preprocessing.

## What's physics-grounded vs. generic (read this)

Be honest about what we can defensibly claim on ESRF/BM18 data:

| operation | grounded? | where / how |
|---|---|---|
| **Paganin deconvolution** | ✅ **exact** | metadata gives `delta_beta`, energy, distance; we invert the *exact* nabu filter (verified to 1e-16). Restores contrast (not resolution — see above). |
| **Unsharp accounting** | ✅ **exact** | metadata gives `coeff`/`sigma`; modeled in the net transfer. Confirmed matching nabu; *matters* (22–37% of signal). |
| **u8 → physical attenuation** | ✅ **exact** | metadata window (`target_window_f32_min/max`) inverted per-volume → consistent physical units across volumes (`fy_u8_to_phys`). |
| **per-volume noise model** | ✅ measured | `fy_estimate_noise` fits `var=g·I+b` from the data; the level varies **1.5–3.3× scroll-to-scroll** so it must be estimated per-volume, not hardcoded. |
| **partial delta_beta (fine vols)** | ✅ measured | full inversion over-inverts ≤4.3 µm volumes; `fy_auto_deltabeta_scale` backs off to ~0.35× (measured optimum). |
| **guided denoise** | ✅ safe, generic | guided filter is the recommended denoiser; strength auto-set from the measured noise. Apply *after* deconv. |
| **radially varying eps** | ✅ measured | flat-noise rises ~3× center→rim on large scrolls (PHercParis3, fixed-intensity; half-acq coverage + path length). Pass 1 fits `fn(r)` from the sample tiles (gated on slope/correlation/span); pass 2 scales the guided eps per tile. |
| **anisotropic PSF (z)** | ✅ measured | helical scans blur ~1.17× broader in z (noise-autocorrelation, PHerc0139 2.4 µm). Pass 1 auto-measures `σz/σxy` (`fy_noise_aniso`, gated ≥1.05, clamped 1.6); the matched deconv inverts the anisotropic PSF. |
| FBP ramp filter inversion | ❌ not clean | sinogram-domain, pre-backprojection — not invertible from the reconstructed volume. **Not implemented.** |
| **residual ring removal** | ✅ detect-then-subtract | `fy_dering_*`: rings detected by an angular **sector sign-consistency vote** at the metadata rotation axis (a spiral wrap drifts in radius with angle and fails the vote; a true ring doesn't — verified on PHerc0139 2.4 µm: 267 ring radii, 93 % of ring energy removed in one pass). Only the detected component is subtracted; not a physics inverse, but measured, gated, and structure-safe. On by default in the pipeline (`--no-dering` to disable; `--dering-center y x` to override the axis). |

So fysics provides a **physics-exact Paganin deblur** (contrast restoration), a
**per-volume-calibrated denoise**, and **exact physical-unit recovery**. The deblur
and dewindow are grounded in the metadata; the denoise is a measured-from-data
complement. We do *not* claim to invert the FBP filter or rings as "physics."

### Ruled OUT empirically (don't re-add these)

Measured across many volumes; each made results *worse* or wasn't worth it:

- **Noise whitening** before denoise — the textbook colored-noise step. Leaks real
  papyrus structure into the residual; *hurts* RMSE at every voxel size (3× confirmed).
- **TV denoising** — erodes mid-band detail (drops to ~76%), eating the contrast the
  deblur just restored. Use the guided filter instead.
- **BM4D colored-noise (PSD) variant** — lost to plain BM4D on this data.
- **Full BM4D** — ~10% better RMSE than guided but ~150× slower (~30 s/cube); not
  worth it for streaming 20 TB. Guided is the pragmatic choice.
- **Aggressive deconv (low reg) / full delta_beta on fine volumes** — over-amplifies
  noise for no resolution gain. `reg=0.015` + `fy_auto_deltabeta_scale` is the knee.
- **Radial cupping correction** — measured ABSENT (PHerc0139 + PHercParis3, 2026-06):
  the air-gap mode shows no radial bowl, only a uniform ~5–10 u8 scatter offset.
  Don't add a radial flattening stage.
- **Helical pitch-periodic z-banding correction** — measured ABSENT: no spectral peak
  at the metadata pitch (2131 slices/rev at 2.4 µm); the per-z spectrum is wrap
  structure. The existing slow z-drift correction is sufficient.
- **Repeat-scan averaging** — scan-to-scan noise/artifacts are CORRELATED (bad SNR in
  one scan predicts bad SNR in the other), so the naive sqrt(2) gain doesn't materialize.
- **Saturation handling in deconv** — clipped voxels are only 0.01–0.13 % of the data;
  measured too rare to justify an inpainting stage.

## What it does

The BM18/ESRF reconstructions apply a Paganin phase-retrieval filter (a deliberate
low-pass that trades resolution for contrast — it makes the near-invisible
carbonized papyrus visible). That filter is analytic and *partly invertible*. fysics
applies a Wiener-regularized inverse of the exact recon transfer:

```
T_paganin(f) = 1 / (1 + delta_beta * lambda * D * pi * f^2)     (matches nabu exactly)
U_unsharp(f) = 1 + coeff * (1 - exp(-2 pi^2 sigma^2 f^2))
H(f)         = T_paganin(f) * U_unsharp(f)
deconv:  F_out = F_in * H / (H^2 + reg)
```

`reg` is the **sharpening-strength dial** (lower = sharper + noisier). The Paganin
parameters (`delta_beta`, `energy`, `distance`, `pixel_size`, unsharp) come from each
volume's `metadata.json`.

## Locality (why it's viewer-friendly)

The transfer is a convolution, so each output voxel depends only on a **local
neighborhood** (~15 voxels for BM18 Paganin), not the whole volume. So you can
sharpen just the region being viewed plus a small halo. Use `fy_kernel_halo()` to
get the margin, process `region + halo`, keep the inner region.

## API (`fysics.h`)

```c
typedef struct {
    double delta_beta, energy_kev, distance_mm, pixel_um, unsharp_sigma, unsharp_coeff;
} fy_physics;

/* sharpen a volume (row-major, x fastest) in place-or-out; reg = strength */
int  fy_deconvolve(const float *in, float *out, int nz, int ny, int nx,
                   const fy_physics *p, double reg);

int  fy_kernel_halo(const fy_physics *p);          /* tile/viewer margin (voxels) */
double fy_recon_transfer(double f_cyc_per_voxel, const fy_physics *p);
/* plus fy_fft1d/3d if you need the FFT directly */
```

### vc3d usage sketch

```c
fy_physics p = { .delta_beta=1000, .energy_kev=78, .distance_mm=220,
                 .pixel_um=2.4, .unsharp_sigma=1.2, .unsharp_coeff=4.0 };
int halo = fy_kernel_halo(&p);                 /* ~15 */
/* fetch the viewed region grown by `halo`, then: */
fy_deconvolve(region_in, region_out, dz, dy, dx, &p, reg);
/* keep the inner [halo..dz-halo] etc. for display */
```

## Whole-volume processing at 20TB+ (streaming)

Real scroll volumes are **20+ TB of dense u8** — they never fit in RAM. fysics is
built for this: it does the per-chunk **math**, and the **caller owns chunk
iteration + I/O** (use your existing fast zarr reader). Operators split two ways:

**LOCAL ops** (deconv, denoise, mask) — each output voxel depends only on a small
neighborhood, so process one chunk **+ a halo** independently (embarrassingly
parallel). Halo = `fy_kernel_halo()` (~15 vox).

**GLOBAL ops** (normalization, ring detection) — need whole-volume statistics,
so they are **two-pass streaming**:

```c
/* PASS 1: accumulate a tiny 256-bin histogram over every chunk */
fy_hist_state st; fy_hist_init(&st);
for (each chunk)            /* your I/O loop */
    fy_hist_accumulate_u8(&st, chunk_u8, chunk_n);
/* (parallel? accumulate per-thread, then fy_hist_merge) */

/* FINALIZE: compute the mapping ONCE from the global histogram */
unsigned char lo, hi; fy_norm_finalize(&st, 0.5, 99.5, &lo, &hi);

/* PASS 2: apply the SAME mapping to every chunk -> consistent everywhere */
for (each chunk) {
    fy_norm_apply_u8(chunk_u8, tmpf, chunk_n, lo, hi);      /* global normalize */
    /* + local ops on chunk+halo: fy_deconvolve, fy_guided_denoise, ... */
    fy_float_to_u8(tmpf, out_u8, chunk_n);                  /* write back u8 */
}
```

The global state is a 256-long histogram (a few KB) — **20TB never sits in RAM**,
and a per-chunk operation never sees inconsistent global stats. (The ring
detector follows the same two-pass pattern with a per-slab radial profile.)

## Streaming process+export (vca_export)

fysics owns the FULL pipeline: `vca_export <zarr|s3://...> <out.mc>` streams an
uncompressed OME-zarr (local or S3) through calibration + the preprocessing
chain into a matter-compressor archive with all 8 LODs, bounded RAM, no
intermediate zarr. Two passes over the input by necessity (the global
operators -- norm histogram, z-drift profile, ring detection -- must finalize
before any tile can be processed): a ~0.25x subsampled calibration sweep,
then ONE streaming read of LOD0 from which every pyramid level is produced
(L1 in RAM, L2+ from the archive itself -- no per-LOD source re-reads). Architecture ported from volume-compressor's
optimized vc_export_stream: occupancy from a coarse pyramid level (one tiny
GET replaces thousands of HEADs; absent bands skipped), a downloader pool
feeding a bounded queue so S3 latency never blocks the compute pool, chunk-
aligned independent (XY-tile x Z-band) units appending L0/L1 lock-free, and
the coarse tail (L2+) built from the archive itself. Hard I/O errors fail
loudly (never silent fill). `--no-process` exports raw; `--threads/--io-
threads/--queue/--sb/--band` tune the pipeline.

## Build (CMake)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build          # correctness tests
./build/bench_fysics            # timing
```

Flags: `-Ofast -ffast-math -funroll-loops -march=native` (override host arch with
`-DFYSICS_ARCH=...` for a portable build). Builds both a static (`libfysics.a`) and
shared (`libfysics.so`) library plus the public header.

## Performance (CPU, -Ofast -march=native)

| region | time | throughput |
|---|---|---|
| 8×512×512 (viewer slab) | ~30 ms | ~71 Mvox/s |
| 64×256×256 | ~73 ms | ~58 Mvox/s |
| 256×256×256 (batch tile) | ~374 ms | ~45 Mvox/s |
| 176×176×176 (real halo'd 128-tile) | ~112 ms | ~49 Mvox/s (was 1.69 s: **15×**) |

End-to-end pipeline throughput (640³ synthetic, page-cached local zarr, full
default chain): **~270 Mvox/s** machine-wide. Hot-path notes: the air-zero
scratch smooth runs on a 2× decimated copy (it only feeds a histogram + fuzzy
threshold); dering radius/sector plane-maps are computed once per tile and
reused across z; local chunk reads are mmap'd (no intermediate copy; halo
slivers fault only the pages they touch — verified byte-identical to the fread
path).

Interactive for viewer-sized regions; the FFT is the cost (self-contained, sizes
2^k and 3·2^k via a radix-3 split, precomputed twiddle tables so the butterflies
vectorize). Sizes pad to `fy_next_fft_size()` — a 176³ halo'd tile costs a 192³
FFT, not 256³ — and the deconv runs on the Hermitian HALF-spectrum (real input):
two-for-one packed X-pass with the reflect-padding fused in, Y/Z passes on half
the columns, inverse transforms only for rows that survive the crop. The Wiener
weight is applied through a radial LUT (no per-voxel sqrt/exp). The guided denoise uses subsampled coefficients by default (He & Sun
fast guided filter, s=2): ~3× faster, visually identical; set
`cfg.guided_subsample = 1` for the exact path. Output u8 quantization is
dithered by default (hash of the global voxel coordinate: unbiased, seam-free,
reproducible) to prevent banding in flat papyrus; `cfg.no_dither = 1` restores
round-to-nearest.

## Verification

The C kernels are tested against (a) a naive DFT, (b) FFT round-trip, (c) the exact
nabu Paganin formula, and (d) the Python reference implementation in
[SuperOptimizer/superresolution](https://github.com/SuperOptimizer/superresolution)
(`superres/paganin.py`) — bit-identical on matching sizes.

## License

See `LICENSE`.
