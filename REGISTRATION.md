# Volume Registration — Layer 1 (affine)

Intensity-based rigid/affine registration of 3D CT volumes, plus the affine
resampler it is built on. The goal is to align multiple scans of the **same**
scroll taken at different resolution / energy / time (e.g. PHerc0139 at 1.129um
vs 2.399um) so they can later be fused.

This is **Layer 1 of 3**. It handles the *global* geometric relationship
(scale from differing voxel size, rotation, translation, shear from optics /
mounting). It does **not** handle physical warping (temperature, movement,
sample creep) — that is Layer 2 (deformable / Demons). Fusion is a later,
separate concern.

Everything is in `register.c` (declarations in `fysics.h`), pure C, CPU-only,
libc + libm only, wired into both the `fysics` and `fysics_shared` CMake targets.

---

## API

```c
/* Trilinear sample at continuous (z,y,x), input-voxel units. OOB -> 0.
 * The reusable interpolation primitive; Layer 2's displacement-field warp
 * calls this directly. */
float fy_sample_trilinear(const float *in, int nz,int ny,int nx,
                          float z, float y, float x);

/* Resample `in` (nz,ny,nx) into `out` (onz,ony,onx) under a 3x4 affine M
 * mapping OUTPUT voxel coords -> INPUT voxel coords. Trilinear; OOB -> 0. */
int fy_warp_affine(const float *in, float *out, int nz,int ny,int nx,
                   const double *M, int onz,int ony,int onx);

/* 2x anti-aliased downsample (gaussian sigma~0.8 then decimate) for a pyramid.
 * `out` holds (nz/2)*(ny/2)*(nx/2) floats; dims returned in *onz,*ony,*onx. */
int fy_downsample2x(const float *in, float *out, int nz,int ny,int nx,
                    int *onz,int *ony,int *onx);

/* NCC between `fixed` and `moving` warped by M, over the in-bounds overlap.
 * Range [-1,1] (1 = perfect); -2 if overlap too small. */
double fy_ncc_warped(const float *fixed, const float *moving,
                     int nz,int ny,int nx, const double *M);

/* Coarse-to-fine registration: maximize NCC(fixed, warp(moving,M)).
 * M_out is BOTH the initial guess (in) and the optimized 3x4 map (out).
 * rigid_only=1 -> 7 dof (rotation + translation + isotropic scale);
 * rigid_only=0 -> 12 dof full affine. */
int fy_register_affine(const float *fixed, const float *moving,
                       int nz,int ny,int nx, double *M_out, int rigid_only);
```

### Geometry convention

Row-major, x fastest: `idx = (z*ny+y)*nx + x` (matches the rest of fysics).

`M` is a 3×4 matrix, 12 doubles, row-major, in **(z, y, x)** ordering, mapping
**output → input** voxel coordinates (the standard backward / "pull" warp every
resampler uses):

```
[zi]   [ M0  M1  M2 ] [zo]   [ M3 ]
[yi] = [ M4  M5  M6 ] [yo] + [ M7 ]
[xi]   [ M8  M9  M10] [xo]   [ M11]
```

So the translation column is `M[3], M[7], M[11]`. Out-of-bounds input
coordinates sample as 0.

### Seeding the initial matrix

`M_out` is read on entry as the initial guess:

* **Same grid, unknown alignment** → identity (`diag(1,1,1)`, zero translation).
* **Known voxel-size ratio** → diagonal scale. To overlay a *coarse* moving scan
  (large voxels) onto a *fine* fixed grid, the output→input scale is
  `voxel_fine / voxel_coarse < 1`. The real-data harness shows this.

For rigid mode the initial isotropic scale is extracted from `cbrt(|det|)` of
the supplied linear block; rotation starts at 0. For affine mode all 12 entries
are seeded directly from `M_out`.

---

## Metric: why NCC

The two scans are taken at different X-ray energies, so the **same material maps
to different intensities** in each volume. Sum-of-squared-differences (SSD) is
therefore **wrong** — it would penalize a correct alignment just because the
brightness/contrast differs.

**Normalized Cross-Correlation** is invariant to any affine intensity transform
`I → a·I + b` of either image (it subtracts the mean and divides by the standard
deviation before correlating). That covers the bulk of an energy-induced contrast
change, which is why NCC is the metric here. It is computed only over the
**in-bounds overlap** (voxels whose moving-sample lands inside the moving grid),
so a partial overlap doesn't get punished for the empty region.

The contrast-robustness test below confirms this: a `gamma=1.8` nonlinear
intensity remap of the moving volume still recovers the geometry at NCC > 0.97.

### Future: Mutual Information

NCC is **not** invariant to arbitrary *nonlinear* contrast or true cross-modality
(e.g. CT vs a stain map). For that, **Mutual Information** (joint-histogram
entropy) is the standard choice. There is a documented hook in `fysics.h`
(`TODO(MI)` on `fy_ncc_warped`): a `fy_mi_warped(fixed, moving, nz,ny,nx, M,
nbins)` with the same signature shape would drop straight into the optimizer
(`optimize_level` only needs a scalar "higher is better" metric). NCC first
because it is simpler, has no binning/parameter choices, and is already
sufficient for energy-difference contrast.

---

## Pyramid (coarse-to-fine) approach

A single-scale optimizer on full-resolution data gets stuck in local optima and
is slow. Instead:

1. Build an image pyramid: up to **4 levels**, each `fy_downsample2x` (gaussian
   anti-alias `sigma~0.8` then 2× decimate). Stop early if a dimension would
   drop below 16.
2. Optimize on the **coarsest** level first (cheap, smooth NCC landscape, large
   capture range), then carry the parameters down to the next finer level and
   refine. Translation parameters are rescaled by `2^level` between levels.
3. At each level, **coordinate descent**: sweep each parameter ±step, keep
   improvements, halve all steps when a full sweep finds none, stop when steps
   are tiny. Derivative-free, robust, no line-search pathologies. Rotation/scale
   are parameterized **about the volume center** so a rotation doesn't fling the
   content off the grid.

Parameterization: rigid = 7 params `[rz,ry,rx, tz,ty,tx, log(s)]` (log-scale
keeps the search symmetric around scale 1); affine = the 12 matrix entries
directly (9 linear about center + 3 translation).

---

## Validation results

All tests live in `tests/test_fysics.c` (run: `OMP_NUM_THREADS=1
./build/test_fysics`). Synthetic volume = 6 gaussian blobs + a low-frequency
sinusoid (asymmetric, so rotations/translations are distinguishable and NCC has
gradient to follow).

### Resampler correctness

| test | result |
|------|--------|
| identity `M` reproduces input | MAD < 1e-6 ✅ |
| integer translation shifts exactly | max err < 1e-5 ✅ |
| round-trip warp by `M` then `M⁻¹` | interior MAD < 0.03 ✅ (interpolation error only) |
| `downsample2x` halves dims, preserves dynamic range | ✅ |
| NCC(v, v) == 1, NCC invariant to `0.4·I + 0.2` | both > 0.999 ✅ |

### Self-registration (recover a known transform)

A `moving` volume is generated from `fixed` by a known transform `Mgen`
(rotation ~0.08 rad on all three axes + scale + translation). Registration is run
from an **identity init** and must recover the transform that undoes it
(i.e. `warp(moving, M_out) ≈ fixed`). Measured on a 64³ volume:

| mode | recovered NCC | interior MAD (warped moving vs fixed) |
|------|---------------|----------------------------------------|
| rigid (7 dof) | **0.9981** | **0.0065** |
| affine (12 dof, +6% scale) | **0.9977** | **0.0075** |

So the recovered transform reproduces the fixed volume to within ~0.7% mean
absolute intensity error — essentially interpolation-limited. The geometry is
recovered accurately; what residual remains is trilinear blur, not
misalignment.

### Contrast robustness (multimodal readiness)

`moving` is given a `gamma=1.8` nonlinear remap **plus** rescale (very different
contrast from `fixed`) on top of the geometric transform:

| | recovered NCC |
|-|---------------|
| rigid, gamma=1.8 moving | **0.9802** |

The geometry is still recovered (NCC > 0.97) despite a contrast curve NCC is not
formally invariant to — this is the evidence that the metric is ready for the
energy-difference case. (The image MAD is large here *by design*, because the
intensities genuinely differ; NCC is the correct geometric measure.)

### Runtime (single-threaded, -Ofast -march=native)

On a **128³** volume:

| mode | time | final NCC |
|------|------|-----------|
| affine (12 dof) | **~9.8 s** | 0.9994 |
| rigid (7 dof) | **~5.2 s** | 0.9994 |

Dominated by repeated NCC evaluations (each samples the whole overlap). The
coarse levels are cheap; most time is the finest level. Straightforward speedups
if needed: limit the finest-level sweep budget, subsample voxels in the NCC, or
cap refinement to the top 2 levels.

### Real-data smoke test

`tests/register_real_smoke.py` (ctypes) loads the two **real** PHerc0139 cubes
(1.129um and 2.399um, 128³ uint8), seeds the voxel-ratio scale (≈0.4706), and
runs full affine registration.

```
voxel-ratio init (scale=0.4706):  initial NCC = -0.319
final NCC = 0.871   (improvement +1.190)
```

**This is NOT a successful physical registration and is not claimed to be.** The
two cubes are from *different spatial chunk indices*, so they almost certainly do
not cover the same physical region. The optimizer happily drives NCC up to 0.87
by warping the moving volume onto whatever coincidentally-correlated texture it
can find — exactly the kind of spurious alignment you get without true overlap.
The harness's only purpose is to prove the **API runs end-to-end on real CT
data**. Finding truly-corresponding regions is a fusion-layer concern.

Run it single-threaded:

```
OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 python3 tests/register_real_smoke.py
```

---

## Known limitations / honest failure modes

* **Capture range.** Coordinate descent from identity reliably recovers
  *moderate* transforms (rotations up to ~0.1–0.15 rad, scale within ~±20%,
  translations within roughly a coarse-level voxel step). For large
  misalignments you must seed `M_out` closer (voxel-ratio scale, a coarse manual
  shift, or centroid pre-alignment). It is a local optimizer, not global.
* **NCC ≠ nonlinear-contrast invariant.** It tolerates affine intensity changes
  and survives moderate gamma, but a genuinely different modality needs MI
  (hooked, not implemented).
* **Same-grid assumption.** `fy_register_affine` samples fixed and moving on the
  same `(nz,ny,nx)` grid. Different-voxel-size scans must be resampled/cropped to
  a common grid first, or have the scale baked into the initial `M_out`. The
  registration optimizes the *map*, not the grids.
* **No masking / weighting.** Air, mounting hardware, and reconstruction
  artifacts all contribute to NCC equally. A papyrus/foreground mask would
  improve robustness on real scans (and is cheap to add as an optional weight in
  `fy_ncc_warped`).
* **Single-precision sampler, double-precision matrix.** Sampling is float
  (fast, vectorizable); the matrix and NCC accumulators are double. Round-trip
  error is interpolation-dominated (~0.7% MAD), not numerically dominated.

---

## What Layer 2 (deformable / Demons) and fusion will need from this

* **The trilinear sampler is the shared primitive.** `fy_sample_trilinear` is
  factored out exactly so a Demons displacement field `(dz,dy,dx)` per voxel can
  reuse it: `out[v] = fy_sample_trilinear(in, ..., z+dz, y+dy, x+dx)`. No new
  interpolation code needed.
* **Layer 1 output is the initialization for Layer 2.** Run affine first to
  remove the global transform, warp the moving volume with the recovered `M`,
  then let Demons solve only the small *residual* nonrigid field. Deformable
  registration converges far better (and can't fix a global rotation/scale)
  starting from an affine-aligned pair.
* **The pyramid + NCC infrastructure carries over.** Demons is also coarse-to-
  fine; `fy_downsample2x` builds the same pyramid, and NCC (or its local-window
  variant / MI) is the same driving metric.
* **Fusion** consumes a *single resampled moving volume* on the fixed grid
  (`fy_warp_affine` for the affine part, then a displacement-field warp for the
  deformable part), at which point both scans live in the fixed coordinate system
  and can be averaged / weighted / super-resolved voxel-for-voxel.
* **Likely additions for Layer 2:** an optional foreground mask/weight in the
  metric, MI as an alternative metric, and a displacement-field warp entry point
  (`fy_warp_field`) alongside `fy_warp_affine`.
```
