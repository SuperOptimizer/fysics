# ESRF / nabu reconstruction-pipeline audit

**Status:** durable reference. Last revised 2026-06-05.
**Scope:** the BM18 / nabu reconstruction pipeline as applied to Herculaneum scroll
volumes (dense u8/u16, ~1.1–9.4 µm voxels, single-distance Paganin phase retrieval
applied during reconstruction). For *every* operator in that pipeline, in order, this
document records: (a) what it does and its exact math/form where known; (b) whether it
acts in the **sinogram/projection domain** or **post-reconstruction**; (c) whether it is
analytically **invertible from the final reconstructed u8 volume** (not from the
sinogram); and (d) the **verdict** for a post-processing library:
*invert-as-exact-physics* / *apply-safe-generic-complement* / *leave-alone* /
*not-recoverable*.

This synthesizes two evidence streams:

1. **Adversarially verified deep-research findings**
   (`superresolution/analysis/research_result.json`). Findings cited below as
   *[research: claim]* carry a 3-0 verification vote unless noted.
2. **Empirical metadata + processing analysis of real PHerc0139 volumes**
   (`docs/empirical-analysis.md`). Cited as *[empirical]*.

The canonical nabu pipeline order is:

> flat/dark-field → double-flat-field (CCD) → intensity normalization →
> **phase retrieval (none / Paganin / CTF)** (+ optional unsharp mask) →
> ring/stripe removal (sinogram) → distortion/alignment →
> backprojection (FBP / **GHBP**) with ramp filter → histogram rescale to u8.

A central, repeatedly verified fact frames every verdict below: **the Paganin / TIE-Hom
filter is linear and low-pass, FBP/backprojection is linear, and the two commute** — so
single-distance phase retrieval *and its inverse* can be validly applied **after** 3-D
reconstruction with spatial resolution, contrast and CNR preserved
[research: claim 7,8; Brombal et al. 2019, J. Synchrotron Rad. mo5194;
arXiv:1808.05368; Ruhlandt & Salditt 2016]. This is the single license that lets fysics
operate on the delivered volume at all. By contrast, every operator that acts on the
**sinogram or projections** (ring removal, ramp filter, flat/dark) is *not* recoverable
from the reconstructed u8.

---

## 1. Flat-field / dark-field correction

**What it does.** Per-pixel detector normalization on each projection:
`I_corrected = (I_raw − D) / (F − D)`, where `D` is the dark (no-beam) frame and `F` the
flat (beam, no sample) frame. Converts measured counts into transmission `T ∈ (0,1]`.

**Domain.** Projection domain (pre-reconstruction), per detector pixel per angle.

**Invertible from the final u8 volume?** **No.** The flat/dark frames are per-pixel,
per-acquisition 2-D arrays applied before the Radon transform; they are not encoded in,
nor separable from, the reconstructed volume. There is no transfer function to invert
post-recon.

**Verdict: not-recoverable.** Leave alone. Any residual flat-field error manifests as
low-frequency shading or rings, which are addressed (heuristically) by the ring/drift
machinery below, not by inverting flat-field.

---

## 2. Double-flat-field (CCD correction)

**What it does.** A second-pass correction (nabu "CCD"/double-flat) that removes residual
structured background — slowly varying flat-field drift and fixed-pattern noise the single
flat-field misses — typically by dividing each projection by a smoothed median of the flat
series.

**Domain.** Projection domain.

**Invertible from the final u8 volume?** **No** — same reasoning as §1; it operates on
projections with data (the flat series) not present in the reconstructed volume.

**Verdict: not-recoverable.** Leave alone.

---

## 3. Intensity normalization / beam-current drift

**What it does.** Compensates for incident-flux changes over the scan. At a storage-ring
source the beam current decays during acquisition, so later projections are systematically
dimmer; normalization rescales each projection to a common incident intensity.

**Domain.** Primarily projection domain. **But** the *residual* of imperfect normalization
appears in the reconstructed volume as a slow **axial (z) intensity drift**.

**Invertible from the final u8 volume?** The projection-domain operator itself: **no**.
The *residual axial shading* in the volume: **partially**, as a generic low-frequency
correction — and crucially, its **magnitude is recorded in metadata**.

**[empirical]** `metadata.json` carries `machineCurrentStart` / `machineCurrentStop`; the
beam-current drift across the real volumes runs **1.5 % → 13.7 %** (worst at 9.362 µm).
This is not a physics inverse of the normalization step, but the drift magnitude can be
**seeded directly from these two numbers** rather than estimated blind.

**Verdict: apply-safe-generic-complement.** A slow axial-intensity-drift correction,
seeded from `machineCurrentStart/Stop`, is justified and bounded. It is a complement, not
an exact inverse.

---

## 4. Ring / stripe artifact removal

**What it does.** Suppresses concentric ring artifacts (from detector-element gain
defects) by attenuating the corresponding **stripes in the sinogram**. nabu offers several
filters [research: claim 0,1; sarepy / Nghia Vo, Diamond]:

- **Wavelet-FFT (Münch et al. 2009).** Decompose the sinogram with a wavelet transform,
  then apply a Fourier-domain **Gaussian damping** to the vertical-detail coefficient
  bands (params: `level`, `sigma`, `order ≈ 8`). Stripes are vertical in the sinogram, so
  damping the vertical-detail Fourier rows removes them.
- **FFT-based filter.** A 1-D low-pass window multiplied with the sinogram row at the
  **zero vertical frequency** and its neighbors (params: strength `u ≈ 30`, shape
  `n ≈ 8`, rows `v ≈ 1`).

**Domain.** **Sinogram domain** — explicitly, by construction.

**Invertible from the final u8 volume?** **No.** These act on the sinogram **before**
backprojection; once reconstructed, the operation cannot be inverted from the volume
[research: claim 0,1 — "act on the sinogram, NOT the final volume … not invertible from
the reconstructed u8 data and should be left alone post-recon"].

**Verdict: not-recoverable (for the applied filter); apply-safe-generic-complement (for
residual rings only).** You cannot undo nabu's ring removal. *Residual* rings that survive
into the volume have **no metadata model** and can only be attacked **heuristically** —
e.g. a polar-domain destriping pass — which is a generic complement, explicitly **not** a
physics inverse. Do not claim ring handling as "physics."

> SOTA note for residual rings + noise: Mäkinen, Marchesini & Foi 2022
> (J. Synchrotron Rad., PMC9070695) model residual streaks **plus** Poisson noise as
> spatially correlated noise and attenuate both with a two-stage multiscale BM4D variant
> [research: claim 10,11]. This is volume-applicable and SOTA but heavy; fysics ruled out
> the full-BM4D path on cost/quality grounds ([empirical]; README "Ruled OUT").

---

## 5. Phase retrieval — Paganin (TIE-Hom) vs CTF

This is the **one operator fysics genuinely inverts as physics.**

### 5a. The three nabu options

nabu's phase stage exposes **exactly three** single-step options — `none`, `Paganin`,
`CTF` — both active methods being **non-iterative linear Fourier-domain filters**
[research: claim 2,3,22; silx/nabu phase docs; nabu MR !104]. CTF (single-distance,
contrast-transfer-function) was added for beamlines like ID16A and follows
Yu/Langer et al. 2018, Opt. Express 26, 11110 (eq. 8 single-distance, eq. 16
multi-distance). **The scroll data uses Paganin** [empirical: every PHerc0139 volume's
metadata specifies single-distance Paganin].

### 5b. Paganin / TIE-Hom exact form

nabu's `PaganinPhaseRetrieval` implements **Paganin et al. 2002** single-material TIE-Hom
retrieval [research: claim 4,21]:

> *Paganin, Mayo, Gureyev, Miller, Wilkins, "Simultaneous phase and amplitude extraction
> from a single defocused image of a homogeneous object", J. Microscopy 206:33–40 (2002).*

It is parameterized by a **single `delta_beta` ratio** (δ/β); lengths are in **meters
since nabu v2021.1.0**. It is equivalent to tomopy's `retrieve_phase` with
`alpha = 1/(4π²·delta_beta)` — a `1/(4π²)` convention factor between the two
[research: claim 4,21]. The assumption is a **homogeneous (single-material) object** where
absorption scales with electron density via a constant `β/δ` ratio — which is exactly why
*one* image recovers both phase and amplitude [research: claim 5,9; PMC7206550].

The filter is a **low-pass Lorentzian** in spatial frequency. fysics uses the
nabu-matching form (verified to ~1e-16):

```
T_paganin(f) = 1 / (1 + delta_beta · lambda · D · pi · f^2)
```

with `lambda` from `energy_kev`, `D = distance_mm`, `f` in cycles/voxel
(`pixel_um` sets the scale). Its strength is set by an **intrinsic regularization
parameter** (wavelength · effective defocus · δ/β); this parameter **can be reduced below
its nominal value to optimize spatial resolution** rather than left at default
[research: claim 6; arXiv:2601.07225, Gureyev/Paganin Jan 2026; filter form
`1/(1+(δ/β)(λz/4π)k²)`].

**REFUTED corollary (scope-relevant):** the claim that an *explicit Tikhonov-PSF
deconvolution* beats standard Paganin or the Beltran 2018 optimization was rejected
**0-3** [research: refuted; arXiv:2601.07225]. **Do not** treat Tikhonov/Wiener PSF
deconvolution as established SOTA over simply reducing the Paganin parameter. (fysics
ships `fy_deconvolve_gureyev` as an *experimental* alternative, not the default path.)

### 5c. Parameters that fix the operator

Per-volume, from `metadata.json` — **none are constant** [empirical]:

| param | role | observed values (1.129 / 2.399 / 2.403 / 9.362 µm) |
|---|---|---|
| `delta_beta` | filter strength δ/β | **500** / 1000 / 1000 / 1000 |
| `energy_kev` | sets λ | 59 / 78 / 77 / 113 |
| `distance_mm` (D) | propagation distance | 200 / 220 / 220 / **1200** |
| `pixel_um` | frequency scale | 1.129 / 2.399 / 2.403 / 9.362 |

The strong dependence on `D` is load-bearing: the 9.362 µm volume's Paganin filter is
~6× stronger (D = 1200 mm), so its Wiener inverse over-amplifies high-frequency noise
unless `reg` is raised [empirical §3 — the discovered reg-scaling bug].

**Domain.** As applied by nabu: projection domain, **but** it **commutes with FBP**, so it
is equivalently a post-reconstruction volume filter [research: claim 7,8].

**Invertible from the final u8 volume?** **Yes — approximately, and this is the key
result.** Because the filter is linear, low-pass, and commutes with backprojection, a
Wiener-regularized inverse applied to the reconstructed volume validly undoes the Paganin
blur with preserved resolution/contrast/CNR [research: claim 7,8]. Caveats:
(i) it is the *3-D* filter that strictly commutes — naive 2-D slice-wise application is an
approximation; (ii) the Paganin log-nonlinearity is only ~linear for weakly-absorbing
objects (carbonized papyrus qualifies); (iii) inversion **restores contrast, not
resolution** — FRC across 7+ volumes shows a 2.5–3× high-frequency *contrast* lift but no
SNR-limited resolution gain (README; [empirical §2] confirms the signal dies into a noise
floor by ~0.5 Nyquist on fine scans).

**Verdict: invert-as-exact-physics.** Invert the exact nabu filter from per-volume
metadata, Wiener-regularized, with `reg` auto-scaled to filter strength and capped at the
empirical noise crossover. For ≤4.3 µm volumes, **partial** δ/β inversion (~0.35×) is the
measured optimum — full inversion over-inverts [empirical; README].

---

## 6. Unsharp mask (post-Paganin sharpening)

**What it does.** nabu **optionally** applies an unsharp mask immediately **after** Paganin
filtering, to counter Paganin blur [research: claim 23; silx/nabu phase docs]:

```
UnsharpedImage = (1 + coeff) · originalPaganinImage − coeff · ConvolutedImage
```

where `ConvolutedImage` is a Gaussian blur of `unsharp_sigma`, controlled by
`unsharp_coeff` and `unsharp_sigma`. **Both default to 0 (disabled)** in upstream nabu;
nabu also supports `unsharp_method` = gaussian / laplacian / imagej.

In frequency terms (Gaussian method), the net per-voxel transfer is:

```
U_unsharp(f) = 1 + coeff · (1 − exp(−2 π² σ² f²))
```

**[empirical] — open question now ANSWERED:** the research left "is unsharp enabled per
scan?" open. The real PHerc0139 metadata shows **unsharp is enabled on every volume**,
with `coeff = 4.0` throughout and `sigma = 2.5` at 1.129 µm vs `1.2` elsewhere — **σ is
not constant.** Unsharp accounts for **22–37 % of the signal** in the net transfer
(README), so it cannot be ignored.

**Domain.** Applied with Paganin (projection domain) but, being linear, equivalently a
post-recon filter and part of the commuting net transfer.

**Invertible from the final u8 volume?** **Yes**, jointly with Paganin: it is linear and
its parameters are in metadata. fysics folds it into the net transfer
`H(f) = T_paganin(f) · U_unsharp(f)` and inverts the product.

**Verdict: invert-as-exact-physics**, as part of the combined Paganin+unsharp transfer.
Because nabu's unsharp *is* enabled here, a post-processing library must **account for it
in the inverse** and must **not** naively add a second sharpening pass on top without
first modeling the one nabu applied.

---

## 7. Distortion / alignment correction

**What it does.** Geometric corrections — detector distortion maps, center-of-rotation
alignment, tilt — applied to projections/sinograms so the Radon geometry is consistent.

**Domain.** Projection / sinogram domain (geometric resampling before backprojection).

**Invertible from the final u8 volume?** **No.** These are geometric resamplings whose
parameters (distortion map, COR, tilt) are acquisition-side and not encoded as an
invertible volume transfer.

**Verdict: not-recoverable.** Leave alone.

> Related, **not** a nabu operator but a real artifact class [empirical]: the fine volume
> is a **19-tile fused helical mosaic** (T0 + ring of 6 + ring of 12). **Tile-seam
> artifacts** are produced by the fusion, have no metadata transfer, and are **not yet
> addressed** by fysics — flagged below as a gap.

---

## 8. Backprojection (FBP / GHBP) + ramp filter

**What it does.** Filtered backprojection: each sinogram row is convolved with a **ramp /
Ram-Lak filter** (`|f|` in frequency, optionally apodized) and backprojected across the
image. **[empirical] the real scroll volumes use GHBP (Gridded Hierarchical
BackProjection), not plain FBP** — the README's "FBP ramp" language should read GHBP.
GHBP is an accelerated backprojection scheme; the ramp filtering role is the same.

**Domain.** **Sinogram / projection domain** — the ramp filter is applied to sinogram rows
**before** backprojection.

**Invertible from the final u8 volume?** **No.** The ramp filter operates on sinogram data
pre-backprojection; it is not a clean, invertible transfer of the reconstructed volume
[research: caveats — "ring-removal and FBP-ramp act on sinogram/projection data and are
NOT recoverable from the reconstructed u8, so leave them alone"]. (Naively re-filtering the
volume by `1/|f|` is **not** an inverse of FBP and would amplify low-frequency noise
catastrophically.)

**Verdict: not-recoverable.** Leave alone. fysics does **not** implement FBP/ramp
inversion and should not claim to.

---

## 9. Histogram rescale to u8 (export window)

**What it does.** The final f32 attenuation volume is mapped to u8/u16 for storage via a
**per-volume linear intensity window**.

**[empirical] — open question now ANSWERED:** the research left the exact u8 mapping open.
The real metadata records it explicitly: `target_window_f32_min/max` and
`window_u16_min/max`. The mapping is a **known linear window of f32 attenuation**, and it
**differs per volume** (f32 max 0.145 / 0.21 / 0.19 / 0.145; u16 start 0 / 5276 / 4752 /
0). Therefore a fixed intensity threshold is **not physical across volumes**.

**Domain.** Post-reconstruction (the very last step), per-volume.

**Invertible from the final u8 volume?** **Yes — exactly.** The window is linear and its
endpoints are in metadata, so `u8 → physical attenuation` is an exact per-volume inverse
(modulo quantization). This recovers consistent physical units across volumes.

**Verdict: invert-as-exact-physics.** fysics inverts the window per-volume
(`fy_u8_to_phys`) before any physics-based deconvolution, and re-applies it on export
(`fy_phys_to_u8`). Note this must happen **before** Paganin deconvolution, since the
physics transfer is defined on attenuation, not on quantized display levels.

> Air masking is **SAM2 (a fine-tuned neural net)**, not an intensity threshold
> [empirical]. It is not a nabu reconstruction operator and not an invertible transfer; an
> intensity threshold will always disagree with the semantic mask at faint papyrus edges.
> Document, don't chase it.

---

## Summary table

| # | operator | domain | invertible from final u8? | fysics verdict | what fysics does |
|---|---|---|---|---|---|
| 1 | flat/dark-field | projection | no | not-recoverable | nothing |
| 2 | double-flat-field (CCD) | projection | no | not-recoverable | nothing |
| 3 | intensity norm. / beam drift | projection (residual: volume) | residual only, generic | safe-complement | axial drift correction, seeded from `machineCurrentStart/Stop` |
| 4 | ring/stripe removal (wavelet-FFT, FFT) | **sinogram** | no | not-recoverable / residual=heuristic | `fy_remove_rings` (heuristic only, **not** physics) |
| 5 | **Paganin / TIE-Hom phase retrieval** | projection (**commutes w/ FBP**) | **yes (approx., linear low-pass)** | **invert-as-exact-physics** | `fy_deconvolve` (Wiener inverse, metadata params, partial δ/β on fine vols) |
| — | CTF phase retrieval | projection | n/a (not used on scrolls) | leave-alone | nothing (scrolls use Paganin) |
| 6 | **unsharp mask** (enabled here) | with Paganin (linear) | **yes (joint with Paganin)** | **invert-as-exact-physics** | folded into net transfer `T·U`, inverted jointly |
| 7 | distortion / alignment | projection/sinogram (geometric) | no | not-recoverable | nothing (tile seams: **gap**) |
| 8 | **GHBP/FBP + ramp filter** | **sinogram** | no | not-recoverable | nothing (does **not** invert ramp) |
| 9 | **u8 export window** | post-recon, per-volume | **yes (exact linear)** | **invert-as-exact-physics** | `fy_u8_to_phys` / `fy_phys_to_u8` |
| + | per-volume noise (not an operator) | volume | — (measured) | safe-complement | `fy_estimate_noise` → `fy_guided_denoise` |

---

## What fysics acts on and why

Every fysics kernel maps onto exactly one verdict above. The discipline is: **invert only
what is a known, linear, metadata-parameterized, post-recon-commuting transfer; complement
(don't invert) what isn't; leave sinogram-domain operators alone.**

- **`fy_deconvolve` (+ `fy_paganin_transfer`, `fy_unsharp_transfer`,
  `fy_recon_transfer`, `fy_auto_reg`, `fy_auto_deltabeta_scale`)** — inverts the **Paganin
  + unsharp** net transfer (§5, §6). This is the *invert-as-exact-physics* core. Justified
  solely by Paganin↔FBP commutativity [research: claim 7,8] and the exactness of the nabu
  filter form [research: claim 4,21]. `reg` is auto-scaled to filter strength because the
  long-distance 9.362 µm volume otherwise amplifies its noise tail 14.5× [empirical §3];
  δ/β is backed off to ~0.35× on ≤4.3 µm volumes because full inversion over-inverts.
  Honest claim: **contrast restoration, not super-resolution.**

- **`fy_u8_to_phys` / `fy_phys_to_u8` / `fy_phys_to_u8_level`** — invert the **export
  window** (§9), exactly, per-volume, from `target_window_f32_min/max`. Run **before**
  deconvolution so the physics acts on attenuation, not display levels.

- **`fy_estimate_noise` → `fy_guided_denoise` (`fy_guided_eps_for_noise`)** — the
  *safe-generic-complement* for the noise that flat/dark/ramp leave behind and that
  deconvolution amplifies. Noise is **colored and signal-dependent** (`var = g·I + b`) and
  varies **1.5–3.3× scroll-to-scroll**, so it is **estimated per-volume**, never hardcoded
  [empirical; README]. The He-Sun-Tang guided filter is the recommended O(N) edge-preserving
  smoother [research: claim 14,15; He/Sun/Tang PAMI 2012]. Applied **after** deconv.
  Empirically ruled out: noise whitening (leaks structure, hurts RMSE 3×), TV (erodes
  mid-band to ~76 %), full BM4D (~10 % better RMSE, ~150× slower) [README; empirical].

- **`fy_remove_rings`** — *heuristic only.* Residual rings have **no metadata model** and
  ring removal is sinogram-domain and **not invertible** (§4). This is a generic
  suppressor, explicitly **not** a physics inverse; documented as such.

- **Beam-drift / intensity-normalization complement** — the residual axial drift (§3) can
  be corrected and **seeded from `machineCurrentStart/Stop`** (1.5–13.7 %). Complement,
  not inverse.

- **Not implemented, by design:** FBP/GHBP **ramp-filter inversion** (§8) and flat/dark
  (§1–2) — sinogram/projection-domain, not recoverable from the volume. fysics does not
  pretend to invert these.

- **`fy_glcae*` / `fy_musica2d` / `fy_clahe2d` / `fy_norm_*`** — display-side
  contrast-enhancement / normalization. Not inverses of any physics operator; generic
  viewing aids, governed by global two-pass streaming stats (README).

---

## Open questions (from the research) — and what we've since answered

The research surfaced four open questions. The empirical PHerc0139 analysis has resolved
the first three:

1. **Exact u8 rescale mapping & invertibility** — *ANSWERED [empirical].* It is a **known
   per-volume linear window** (`target_window_f32_min/max`, `window_u16_min/max`), exactly
   invertible (§9). Differs per volume; inverted by `fy_u8_to_phys`.

2. **Is nabu's unsharp enabled per scan? Any ring removal baked in?** — *unsharp ANSWERED
   [empirical]:* **enabled on every scroll volume** (`coeff = 4.0`, `sigma` 2.5 or 1.2),
   22–37 % of signal — so fysics inverts it jointly and does **not** double-sharpen (§6).
   Whether sinogram ring removal was applied: assume yes per standard pipeline; it is
   *not-recoverable* regardless, and residual rings are handled heuristically (§4).

3. **Optimal δ/β & reduced regularization for these dense u8 volumes** — *PARTIALLY
   ANSWERED [empirical].* Full inversion over-inverts ≤4.3 µm volumes; the measured optimum
   is **partial δ/β (~0.35×)** plus `reg` auto-scaled to filter strength and capped at the
   **~0.5-Nyquist noise crossover** (no recoverable signal above it). Consistent with the
   "reduce the Paganin parameter to optimize resolution" result [research: claim 6;
   arXiv:2601.07225]; a rigorous FRC-vs-as-delivered sweep per voxel size remains future
   work.

4. **Best structure-detection (Frangi sheetness/Hessian) and contrast-enhancement
   (MUSICA/CLAHE/GLCAE) for plate-like papyrus sheets** — *STILL OPEN.* This category
   produced **no surviving verified claims** in the research and needs dedicated follow-up.
   fysics ships `fy_musica2d` / `fy_clahe2d` / `fy_glcae2d` as available tools, but their
   optimality for papyrus sheets is unvalidated.

### Current gaps (operators we are NOT handling that we maybe should)

- **Mosaic tile seams.** The fine volume is a 19-tile fused helical mosaic [empirical].
  Seam discontinuities are a real artifact class with **no metadata transfer** and are
  **not yet addressed**. Not a nabu reconstruction operator, but it lands in the delivered
  volume; a seam-aware blending/destriping pass is a candidate complement.
- **Residual-ring SOTA.** The Mäkinen 2022 correlated-noise BM4D framework jointly attacks
  residual streaks + Poisson noise [research: claim 10,11] and outperforms our heuristic
  ring removal in principle; it was ruled out on **cost** (~150× slower), not correctness.
  Worth revisiting if a faster implementation appears.

---

## Sources

Verified research findings and primary documentation cited above:

- **nabu phase docs** — https://www.silx.org/pub/nabu/doc/phase.html
  (Paganin/CTF options, `delta_beta`, tomopy `1/(4π²)` equivalence, unsharp form).
- **nabu MR !104 / issue #213** — https://gitlab.esrf.fr/tomotools/nabu/-/merge_requests/104
  (CTF single-distance support).
- **Paganin et al. 2002**, J. Microscopy 206:33–40 —
  https://onlinelibrary.wiley.com/doi/abs/10.1046/j.1365-2818.2002.01010.x (TIE-Hom).
- **Yu, Langer et al. 2018**, Opt. Express 26, 11110 — https://doi.org/10.1364/OE.26.011110
  (CTF eq. 8 / 16).
- **Homogeneous-object β/δ constraint** — https://pmc.ncbi.nlm.nih.gov/articles/PMC7206550/
- **Gureyev/Paganin, Jan 2026**, arXiv:2601.07225 — https://arxiv.org/pdf/2601.07225
  (reduced regularization to optimize resolution; **Tikhonov-PSF superiority REFUTED 0-3**).
- **Brombal et al. 2019**, J. Synchrotron Rad. mo5194 —
  https://journals.iucr.org/s/issues/2019/02/00/mo5194/ and arXiv:1808.05368
  (post-reconstruction 3-D phase retrieval preserves resolution/contrast/CNR — the
  commutativity license).
- **Ring/stripe filters (Münch 2009 wavelet-FFT, FFT)** — sarepy (Nghia Vo, Diamond) —
  https://sarepy.readthedocs.io/toc/section3_1/section3_1_2.html
- **Mäkinen, Marchesini & Foi 2022**, J. Synchrotron Rad. —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9070695/ (correlated-noise BM4D for streaks).
- **He, Sun, Tang 2012**, IEEE PAMI — guided filter —
  https://people.csail.mit.edu/kaiming/publications/pami12guidedfilter.pdf
- **BM18 multiresolution phase-contrast tomography** —
  https://www.tandfonline.com/doi/full/10.1080/08940886.2024.2414724

Empirical findings: `docs/empirical-analysis.md` (PHerc0139, 2026-06-05). fysics kernel
behavior and rule-outs: `README.md`, `fysics.h`.
