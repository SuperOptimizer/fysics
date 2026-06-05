# Multi-resolution fusion of scroll CT scans — policy, verification, and honest result

Two scans of the *same* scroll at different voxel sizes (PHerc0139: 1.129 um "fine"
and 2.399 um "coarse") carry **independent** measurements of the same object. Where
two independent noise realizations *agree*, that agreement is real signal, so
averaging the shared (low/mid) band denoises it while the fine scan still supplies
the high band. That is the premise of `fy_fuse_multiscale`.

This document records (1) the **policy** for using a dataset's shipped registration
transform, (2) the **real-data verification** of that transform on PHerc0139, and
(3) the **honest fusion result**: on this real pair, naive fusion does **not** beat
the fine scan alone, and **why** (registration, not the fusion math, is the blocker).

The fusion kernels (`fy_affine_from_points`, `fy_fuse_multiscale`, `fy_warp_affine`)
are correct — the C `test_fysics` synthetic suite passes (fusion: low-band noise
0.81x fine-alone at 0.56 high-band retention), and `fy_affine_from_points`
reproduces the shipped 3x4 matrix from the shipped landmark pairs to 1e-11. The
failure below is entirely in the **dataset's coordinate convention**, not the code.

Real-data scripts and artifacts live in
`/home/forrest/superresolution/analysis/fusion/` (zarrio.py, fetch_level.py,
fysics_fuse.py, verify_lm11.py, run_fusion.py, and the *.json / *.png outputs).

---

## 1. Policy: use-if-exists, verify, refine, fallback

**(1) USE the shipped transform if it exists.** Many Vesuvius zarrs ship a
`transform.json` next to the array:

    aws s3 cp --no-sign-request \
      s3://vesuvius-challenge-open-data/{SCROLL}/volumes/{ZARR}/transform.json -

It contains a 3x4 `transformation_matrix` plus matched `fixed_landmarks` /
`moving_landmarks` (12 pairs for PHerc0139 fine). The matrix maps *moving* (this
scan) coords -> *fixed* (a reference scan) coords; `fixed_volume` names the
reference. For PHerc0139 the **fine 1.129 um** zarr ships a transform to the
**2.399 um** scan, so it is exactly what we need to bring fine -> coarse. The
**2.399 um** zarr ships a transform to a 4.681 um reference; the **9.362 um** scan
ships none.

`fysics_fuse.load_transform_json()` parses it and returns the matrix + landmark
arrays + the matrix's self-residual on its own landmarks.

**(2) DO NOT blindly trust it — VERIFY.** Measure three things:

  * **landmark self-residual**: RMS of `M @ moving_landmark` vs `fixed_landmark`.
    (PHerc0139: **1.04 voxels** — the matrix fits its own landmarks well, and
    `fy_affine_from_points(moving, fixed, model=affine)` reproduces it to 1e-11.)
  * **coordinate reconciliation**: the landmark numbers are *not* automatically
    masked-zarr level-0 voxel indices. Check `metadata.json -> zarr_export`
    (`z_crop_start`, window scaling) and `.zattrs` multiscale `scale` to see what
    frame they live in (pre-crop reconstruction voxels? physical mm? a binned
    reference?). The matrix scale tells you the unit: PHerc0139's linear scale is
    **0.4701 == 1.129/2.399**, i.e. the landmark frames are *voxel-like* with each
    scan in its own voxel size — **not mm** (mm would give scale ~1).
  * **image agreement on an overlapping textured region**: apply the (reconciled)
    transform to bring a matched region together and measure voxel-level agreement.
    Plain NCC is weak on self-similar papyrus, so blur to the coarse scan's
    resolution and/or use gradient-magnitude NCC; it should rise clearly above
    chance when truly aligned. Also do a **matrix-free** check: a patch centered on
    `moving_landmark[i]` (downsampled to coarse res) should correlate with a patch
    centered on `fixed_landmark[i]` — this tests the correspondence without trusting
    the matrix or any frame assumption.

  Classify the transform **GOOD** (image NCC clearly positive at matched regions),
  **APPROXIMATE** (gross structure overlays, NCC modestly positive, residual a few
  voxels — refine before fusing), or **MISALIGNED** (image NCC ~0 even at coarse
  blur — the frame mapping is wrong; do not fuse with it).

**(3) REFINE if needed.** If APPROXIMATE, refit the matrix *in the correct voxel
frame* with `fy_affine_from_points(moving_voxels, fixed_voxels, model=similarity,
ransac>0)` on the reconciled landmark pairs (they are the ground-truth
correspondence; intensity NCC on papyrus is non-discriminative and finds spurious
poses — see below), then optionally a short local refinement. Do **not** trust
`fy_register_affine` to *find* the pose from scratch on two papyrus scans: on this
data it climbed to NCC 0.69 at a spurious ~90-degree rotation, a textbook false
maximum on a self-similar laminar medium.

**(4) FALLBACK when there is no transform.json** (coarse 9.362 um; other scrolls):
chain through a scan that does have one (9.362 has none, but 2.399 does, and 1.129
-> 2.399 exists), composing the 3x4 maps (`fysics_fuse.compose34`). A full
from-scratch *feature-based* registration (3D keypoint detect + descriptor match +
`fy_affine_from_points` with RANSAC) is the principled path when no chain exists;
it is noted here as future work — intensity registration is not a substitute.

---

## 2. Real-data verification of the PHerc0139 fine->coarse transform

**Coordinate reconciliation.** The shipped landmark values (fine z 6457..22128,
coarse z 8773..16136) at first appear to be level-0 voxel indices: at scale 32 they
fall onto non-zero data in the level-5 pyramids (coarse 11/12, fine 7/12), the
matrix scale equals the voxel ratio, and the two landmark clouds are a consistent
rigid body — their pairwise-distance ratio is **2.1274 +/- 0.0005**, matching
2.399/1.129 = 2.1249 to 0.1%. So the matrix is a *valid* similarity in the landmark
frame, and `z_crop`/window metadata do not need to be folded in to make the matrix
self-consistent.

**But image agreement fails.** Treating the landmarks as masked-zarr level-0 voxels
and applying the matrix:

  * **matched region (around landmark 11), full-res NCC = -0.005**; gradient-NCC
    0.03; a +/-4..20 voxel translation search never finds a peak (best 0.12-0.28,
    always at the search-box corner).
  * **even at heavy common blur** (to ~9-14 um, well inside the 2.399 um scan's
    resolution) fine vs warped-coarse stays **anti-correlated** (NCC -0.06..-0.11).
  * **matrix-free landmark-patch test** (no matrix, no frame assumption): a fine
    patch at `moving_landmark[i]` vs a coarse patch at `fixed_landmark[i]`, fine
    downsampled to coarse grid: NCC **-0.06 to -0.15** at every landmark tested
    (lm1,2,3,5,11), best small-shift <= 0.02.
  * **whole-FOV silhouette**: the fine FOV maps (correctly, by the matrix) into a
    thin z-slab of the coarse volume (coarse z 5722..14816, ~12% of coarse height;
    y,x ~60% — the fine scan is a *nested sub-volume* of the coarse one, as
    expected). Overlaying the bulk silhouettes there gives IoU 0.48 and bulk NCC ~0.

**Verdict: MISALIGNED in the masked-zarr voxel frame.** The landmark *pairs* are an
internally consistent correspondence at the right scale (so the matrix is the
correct similarity *in whatever frame the landmarks were authored*), but that frame
is **not** the masked-zarr level-0 voxel index — each scan's landmark frame is a
per-scan rigid offset (and possibly axis flip/permutation) away from its zarr voxel
grid. Because the matrix folds *both* per-scan landmark-frame offsets into its
translation, it cannot be repaired by reconciling a single crop origin; you would
need each scan's landmark-frame -> zarr-voxel map, which the shipped metadata does
not give explicitly. This is the "coordinate-convention mismatch" a prior agent
flagged, now confirmed three independent ways (silhouette IoU, matched-region NCC,
matrix-free landmark-patch NCC).

The honest interpretation: **the shipped transform is correct in its own landmark
coordinate system but unusable as-is to align the masked zarrs**, and the metadata
shipped alongside is insufficient to reconstruct the missing per-scan frame maps.

---

## 3. Honest fusion result on real data

Despite the misalignment, we ran the full pipeline (warp coarse onto the fine grid
with the shipped matrix, then `fy_fuse_multiscale`, split_sigma=2.5 ~ coarse
Nyquist) on the landmark-11 region and measured fused-vs-fine:

| variant                         | flat-region noise (fused/fine) | high-band retention |
|---------------------------------|--------------------------------|---------------------|
| fusion, shipped matrix          | 0.881                          | 0.893               |
| fusion, best +/-5 vox "oracle"  | 0.882                          | 0.894               |
| control: just shrink fine's own low band 0.85x | 0.914           | 0.923               |

and the **shared-band correlation** (the thing that makes fusion *real*):
low-band NCC = **-0.05** (shipped), best +/-5-voxel search **0.19** (at the search
corner — not a true peak).

**Conclusion: real-data fusion does NOT beat the fine scan alone here.** The 12%
"noise drop" is barely better than the 9% you get by trivially attenuating the fine
scan's *own* low band (the control), and the low-band NCC is ~0. So the apparent
denoising is just the variance reduction of averaging-in an essentially
**uncorrelated** second volume — which equally attenuates *real* low-band signal.
With two *correctly registered* scans the low-band NCC would be strongly positive
and the averaging would denoise without blurring; here it does not, because the two
volumes are not voxel-aligned.

**This is a registration failure, not a fusion-math failure.** The synthetic
`test_fysics` fusion test (perfectly aligned inputs) gives the expected 0.81 noise /
0.56 retention. Real-data fusion will deliver the same gain **once the fine and
coarse scans are genuinely co-registered** — i.e. once the landmark frame -> zarr
voxel mapping is recovered (or the scans are re-registered from scratch with
feature matching). As it stands, fusing through the shipped transform blurs the
fused low band rather than denoising it.

### Recommended pipeline

1. Fetch and parse `transform.json` (`load_transform_json`); report the matrix
   self-residual and scale.
2. Reconcile the landmark frame against the masked zarr **and verify with image
   agreement** (blurred / gradient NCC at matched regions + the matrix-free
   landmark-patch test). Do not proceed to fusion unless image NCC is clearly
   positive at matched regions.
3. If APPROXIMATE, refit/refine in the correct voxel frame with
   `fy_affine_from_points` (similarity + RANSAC) on the reconciled landmarks.
4. Warp coarse -> fine grid (`fy_warp_affine`), confirm voxel alignment, then
   `fy_fuse_multiscale`. Validate honestly: require low-band NCC clearly > 0 and
   a fused/fine noise drop beating the self-blur control before claiming a gain.
5. For PHerc0139 specifically: obtain each scan's landmark-frame -> masked-zarr
   voxel map from the data producers (the shipped metadata is insufficient), or
   re-register the fine and coarse masked zarrs from scratch via 3D feature
   matching feeding `fy_affine_from_points`. Until then the shipped transform
   should be treated as MISALIGNED for masked-zarr fusion.
