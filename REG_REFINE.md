# Sub-voxel registration refinement for multi-resolution scroll fusion

Goal: take the coarse seed transform that ships with a PHerc volume
(`transform.json`) and **refine it to sub-voxel accuracy** so the high-frequency
band of a fine scan survives multi-resolution fusion. Plain intensity NCC is
non-discriminative globally on self-similar laminar papyrus, so this uses
phase-correlation + mutual information on **local textured patches**.

Test pair: **PHerc0139 1.129 µm (fine / "moving")** ← **2.399 µm (coarse /
"fixed")**, an overlapping region selected via the shipped landmarks. Experiments
and data live in `/home/forrest/superresolution/analysis/reg_refine/` (the C
kernels are here in the `fysics-reg-refine` worktree).

## New C kernels (committed, tested)

| kernel | what it does | test result |
|---|---|---|
| `fy_phase_correlate(fixed,moving,nz,ny,nx,shift,peak,window)` | sub-voxel translation via the Fourier shift theorem: phase-only cross-power spectrum → integer peak → **Foroosh (2002)** sub-pixel estimator `δ=cs/(cs±c0)`. Contrast/brightness-robust (phase-only discards magnitude). | synthetic known sub-voxel shift recovered to **~0.012 vox/axis**; survives an affine intensity change (L1 err 0.036); **on real CT patches with a known shift, MAE = 0.0016 vox** |
| `fy_mutual_information(fixed,moving,nz,ny,nx,M,nbins)` | joint-histogram MI in nats; the multimodal gold standard (no linear-intensity assumption, unlike NCC) | MI higher at correct alignment than shifted; MI(dependent) ≫ MI(independent noise); rises under a **nonlinear** intensity remap where NCC's assumption breaks |

Both build into `libfysics`, are declared in `fysics.h`, and have passing
entries in `tests/test_fysics.c` (`test_phase_correlate_subvoxel`,
`test_mutual_information_peaks`). MI was previously only a documented hook in
`register.c`; it is now implemented.

Implementation note (why Foroosh, not parabola/upsampled-DFT): the
phase-correlation peak is a Dirichlet kernel, not a smooth parabola. A 3-point
parabola biases toward the integer (measured ~0.1 vox bias); an upsampled-DFT on
the phase-only surface picks the wrong real/imag lobe. Foroosh is the
model-correct inverse of the Dirichlet sampling and is exact to ~0.01 vox.

## Coordinate reconciliation (this alone fixed the "NCC ~ 0" problem)

The shipped `transform.json` for the 1.129 µm volume has a 3×4
`transformation_matrix` + 12 fixed/moving landmark pairs. Two conventions had to
be reconciled before the seed lands in the masked-zarr voxel frame:

1. **Axis order is (X, Y, Z), not (Z, Y, X).** The masked zarr and all of fysics
   index `(z, y, x)`. Proof: the moving (1.129 µm) landmark column maxima are
   `[19142, 29587, 19142]`; the zarr shape is `(z,y,x) = (19297, 35971, 32318)`.
   Only the **(x,y,z)** reading fits every landmark inside the volume
   (`z_max 19142 < 19297`); the naive `(z,y,x)` reading puts `z = 22128 > 19297`,
   out of bounds. So the matrix and landmarks are XYZ.
2. **Direction:** the matrix maps **moving (1.129 µm) → fixed (2.399 µm)** voxel
   coords, isotropic scale **0.4701 ≈ 1.129/2.399** (`det^(1/3)`). The
   `fixed_volume` field confirms the 2.399 µm scan is the fixed frame. Landmarks
   are level-0 masked-zarr voxels (post-crop); no extra crop offset was needed
   for the overlap we fetched.

With axes reversed and the matrix inverted into a fysics pull-map
`P: fixed-voxel(zyx) → moving-voxel(zyx)`, the seed lands correctly:

| seed quality | global NCC |
|---|---|
| naive scale-only diagonal, **no** reconciliation (the old smoke-test path) | NCC measured but **meaningless** — coincidental self-similar texture |
| coord-reconciled shipped transform | **NCC = +0.92** |

The ~0 NCC reported earlier was the coordinate-convention mismatch, not a
registration failure. Landmark fit residual (M @ moving vs fixed) is **0.86 vox
median / 1.04 mean**, matching the "~1 voxel" figure.

## Refinement pipeline (landmark-seeded local-patch)

1. Reconcile coords → seed pull-map `P` (above).
2. Resample the fine scan onto the coarse grid via `P` (`fy_warp_affine`).
3. Sample ~400 **textured** sub-patches (std > thresh) across the overlap. For
   each: extract the fixed patch and the seed-resampled moving patch (same grid),
   `fy_phase_correlate` → a sub-voxel local residual displacement; keep the
   high-confidence half (phase-peak gate).
4. RANSAC-fit (`fy_affine_from_points`, full affine) a global correction from the
   patch-center correspondences `c → c+shift`; compose with `P` → `P_refined`.
5. Re-measure on a fresh patch set.

(Driver scripts in the analysis dir: `fetch_overlap.py`, `seed.py`, `refine.py`,
`floor_probe.py`, `known_shift_real.py`, `demons_step.py`.)

## Achieved accuracy on real data (PHerc0139, landmark lm1)

| stage | global NCC | median local residual (2.399 µm voxels) | per-axis residual std |
|---|---|---|---|
| shipped transform, wrong convention | ~0 (non-discriminative) | — | — |
| **seed**, coord-reconciled | +0.922 | **0.81 vox** (mean 1.09) | mean offset (0.65,−0.62,0.40) + scatter ~0.4 |
| **+ patch-refined affine** | +0.923 | **0.74 vox** (mean 0.77) | scatter ~0.42 |

The refinement **removes the ~1-voxel global offset** (translation
`(−1.36, 0.90, −1.42) → (−0.31, 0.01, −0.37)`), leaving a **~0.42 vox/axis
residual scatter** that is the floor.

### Is the floor an artifact or real? (rigorous checks)

* **Estimator is not the limit.** On real CT patches shifted by a *known*
  sub-voxel amount, `fy_phase_correlate` is accurate to **MAE 0.0016 vox**
  (fixed) / 0.0012 vox (fine). The estimator is ~250× finer than the floor.
* **Floor is patch-size independent.** Median residual is 0.72–0.74 vox for
  half = 16, 24, 32, 40 — not averaging-noise from small patches.
* **Residual is a smooth field, not white noise.** Residual-vector dot-product is
  **larger for nearby patch pairs (0.33) than far pairs (0.21)**, both positive →
  a low-amplitude **non-rigid warp** between the two scans (thermal/remount/creep
  + cross-energy decorrelation), plus the resolution gap.

**Conclusion: the ~0.4 vox/axis floor is real physical scan-to-scan disagreement,
not an algorithm limit.** In physical units the residual is
**0.73 coarse-vox = 1.76 µm = 1.56 fine (1.129 µm) voxels.**

### Why Demons does NOT help here (honest negative result)

Adding `fy_register_demons` after the affine refine **raised** NCC (0.92 → 0.96)
but **worsened** the independent phase-correlation residual (0.73 → 0.83 vox),
moving content by a median ~2.2 vox. The intensity metric is maximized by
snapping onto **spurious self-similar laminar matches**, not true
correspondence — exactly the failure mode that motivated patch-based
registration. **Do not run intensity-driven Demons on this material.** The
patch-refined affine is the correct stopping point.

## Is it good enough to fuse the high-frequency band?

A residual misalignment `d` between the two scans attenuates a fused wavelength λ
by ≈ cos(π d/λ); detail stays faithful (>~3 dB) only for λ ≳ 3d. With
**d = 1.56 fine voxels**:

* smallest faithfully-fused wavelength ≈ **4.7 fine voxels ≈ 5.3 µm**;
* the fine scan natively resolves ~2.3 µm (2-voxel Nyquist).

So **registration is now accurate to ≈0.73 coarse-voxel (1.76 µm), which means
fusion preserves detail down to ≈5.3 µm but smears the fine scan's top ~2× octave
(≈2.3–5 µm).** That is a **net win for the low/mid band** (fusion denoises the
shared band, which it already did) but it does **not** yet deliver the fine
scan's full high-frequency band into the fused product.

To recover the last octave you need to beat the *physical* floor, not the
algorithm: (a) register fine↔fine (1.129 vs another high-res scan) so there is no
resolution gap and no cross-energy contrast change; (b) a **patch-validated**
(not intensity-validated) local warp that is regularized hard enough not to chase
self-similar matches; (c) accept the current accuracy and fuse only up to ~5 µm,
which is still a large denoising gain over either scan alone.

## Reproduce

```
# C kernels + tests
cd /home/forrest/fysics-reg-refine && cmake --build build && \
  OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 ./build/test_fysics   # ALL PASSED

# real-data experiment (single-threaded)
cd /home/forrest/superresolution/analysis/reg_refine
ENV="OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
env $ENV python3 fetch_overlap.py 1 128     # overlapping cubes via transform.json
env $ENV python3 refine.py lm1              # seed -> patch-refine -> measure
env $ENV python3 known_shift_real.py lm1    # estimator floor on real patches
env $ENV python3 floor_probe.py lm1         # floor vs patch size + smoothness
env $ENV python3 demons_step.py lm1         # demons overfits (negative result)
env $ENV python3 make_figs.py lm1
```
