# Literature review: post-reconstruction processing for phase-contrast micro-CT

**Status:** durable reference. Last revised 2026-06-05.
**Scope:** a methods-by-category review of post-reconstruction processing for
synchrotron micro-CT phase-contrast volumes — specifically ESRF/BM18, nabu-reconstructed
Herculaneum papyrus (PHerc) data: dense u8/u16, ~1.1–9.4 µm voxels, single-distance
Paganin phase retrieval applied during reconstruction. For each method we record what it
does, its citation, **our verdict** (use / skip / optional), and **why** — tied to a
measured finding on real PHerc data, not just to the literature.

## How to read this

This is not a neutral survey. Every verdict reflects what we **measured** on real
PHerc0139 cubes (see `docs/empirical-analysis.md`) and what the deep-research findings
adversarially verified (`superresolution/analysis/research_result.json`, 12 verified
findings / 23 sources, cited below as *[research]*). **Where our data contradicted the
textbook, we say so explicitly and side with our data.** Three contradictions in
particular drive this document:

1. **Whitening hurts.** The standard colored-noise recipe says: estimate the noise
   power spectrum, whiten, then denoise with a white-noise denoiser. On our data,
   whitening *leaks real papyrus structure into the residual and raises RMSE ~3×* at
   every voxel size. We **skip it**.
2. **Gureyev–Paganin Tikhonov-PSF deconvolution is not better.** The published claim
   that an explicit Tikhonov-regularized PSF deconvolution beats standard Paganin /
   the Beltran-2018 optimization was **refuted 0-3** in adversarial verification, and we
   do not adopt it as the default.
3. **Deblur restores contrast, not resolution.** Inverting the Paganin low-pass gives
   a large mid-band *contrast* lift, but Poisson-thinning FRC shows **no SNR-limited
   resolution gain**. The textbook intuition that "deblur = sharper = more resolved
   detail" is false here; deblur and resolution are decoupled by the noise floor.

Verdict vocabulary: **USE** = in the default pipeline; **SKIP** = measured worse, or
not worth it, or refuted; **OPTIONAL** = a quality-tier or cosmetic mode that is
correct but off by default. "Why" always points at the measured reason.

The frame for the whole document — repeatedly verified — is that the Paganin/TIE-Hom
filter is linear and low-pass, FBP/backprojection is linear, and **the two commute**, so
single-distance phase retrieval *and its inverse* can be validly applied to the final
reconstructed volume with resolution/contrast/CNR preserved
[research: Brombal et al. 2019, J. Synchrotron Rad. mo5194; arXiv:1808.05368; Ruhlandt
& Salditt 2016]. That commutativity is the only license that lets us post-process the
delivered volume at all. The companion `docs/esrf-pipeline.md` audits the reconstruction
operators themselves (and which are invertible from the final u8); this document reviews
the post-processing *methods* we layer on top.

---

## 1. Phase retrieval & deconvolution

| method | what it does | citation | verdict | why (measured) |
|---|---|---|---|---|
| **Paganin / TIE-Hom single-distance retrieval** | single-material phase retrieval; a low-pass Lorentzian filter `1/(1+δβ·λ·D·π·f²)` parameterized by one δ/β ratio | Paganin et al. 2002, J. Microsc. 206:33–40 [research] | **USE (as the inverse)** | This is what nabu applied during reconstruction; we invert the *exact* nabu filter (verified to ~1e-16) from per-volume metadata. The filter↔FBP commute [research: Brombal 2019], so inverting it on the final volume is valid physics. |
| **Wiener-regularized deconvolution of the recon transfer** | `F_out = F_in·H/(H²+reg)` over the combined Paganin+unsharp transfer `H` | Paganin 2002 + standard Wiener regularization | **USE** | Our core operator. Restores **contrast, not resolution**: FRC across 7+ volumes shows a ~2.5–3× / 8–15× mid-band power lift but **no SNR-limited resolution gain** (README; empirical §2 — signal dies into a noise floor by ~0.5 Nyquist on fine scans). Honest claim: contrast/legibility restoration. |
| **Partial δ/β inversion on fine volumes** | back off the δ/β used in the inverse below its nominal metadata value (~0.35×) | Beltran et al. 2018; reduced-parameter result in arXiv:2601.07225 (Gureyev/Paganin, Jan 2026) [research] | **USE (≤4.3 µm)** | **Measured:** full inversion *over-inverts* ≤4.3 µm volumes; ~0.35× is the empirical optimum (`fy_auto_deltabeta_scale`). Consistent with the literature's "reduce the regularization parameter to optimize spatial resolution" [research: arXiv:2601.07225]. |
| **Auto-scaled regularization (`reg ∝ δβ·λ·D/pixel²`)** | scale Wiener `reg` to the recon-filter strength instead of a constant | (our finding; motivated by filter form, Paganin 2002) | **USE** | **Measured bug:** at constant `reg=0.015` the 9.362 µm volume (D=1200 mm, ~6× stronger filter) has gain *increasing* with frequency, amplifying the noise tail **14.5×** at 0.7 Nyq (empirical §3). `reg` must track filter strength and cap at the ~0.5-Nyq noise crossover. |
| **CTF (contrast-transfer-function) single-distance retrieval** | alternative single-step Fourier filter for strong-phase / multi-material objects | Yu, Langer et al. 2018, Opt. Express 26:11110; nabu MR !104 [research] | **SKIP (n/a)** | The scroll data uses Paganin, not CTF [empirical: every PHerc volume's metadata specifies single-distance Paganin]. Nothing to invert. Documented for completeness. |
| **Gureyev–Paganin Tikhonov-PSF deconvolution** | combine TIE-Hom retrieval with an explicit Tikhonov-regularized deconvolution of the imaging-system PSF, claimed to beat standard Paganin | arXiv:2601.07225 (Gureyev/Paganin, Jan 2026) [research] | **SKIP as default** | **Contradicts literature claim.** The "beats standard Paganin / Beltran-2018" claim was **refuted 0-3** in adversarial verification [research: refuted]. We ship it only as an *experimental* alternative (`fy_deconvolve_gureyev`), never the default path. |
| **Unsharp-mask accounting** | nabu optionally applies `(1+coeff)·orig − coeff·blur` after Paganin; we model it in the net transfer | nabu phase docs [research: claim 23] | **USE** | **Measured:** unsharp is *enabled on every scroll volume* (`coeff=4.0`, σ=2.5 or 1.2) and is **22–37 % of the net signal** (README; empirical §1). We fold it into `H = T_paganin·U_unsharp` and invert jointly — and crucially do **not** add a second sharpening pass on top. |

**Category verdict.** USE the exact metadata-driven Paganin+unsharp Wiener inverse, with
partial δ/β on fine volumes and filter-strength-scaled `reg`. SKIP Gureyev-Tikhonov as
default (refuted) and CTF (not the modality used). The deblur is contrast restoration,
not super-resolution — that decoupling is the single most important measured result here.

---

## 2. Denoising

Our overriding measured fact: **the noise is colored and signal-dependent**
(`var ≈ g·I + b`, fit per cube), and its level varies **1.5–3.3× scroll-to-scroll**, so
it must be estimated per-volume (`fy_estimate_noise`), never hardcoded (empirical §2;
README). That single fact decides most rows below.

| method | what it does | citation | verdict | why (measured) |
|---|---|---|---|---|
| **He–Sun–Tang guided filter** | edge-preserving smoother on a local linear model `q=a·I+b`; exact O(N), independent of window size | He, Sun, Tang, IEEE PAMI 2012 [research] | **USE (fast tier)** | The recommended O(N) edge-preserving denoiser; strength auto-set from the measured noise (`fy_guided_eps_for_noise`), applied **after** deconv. Fast enough for streaming 20 TB. The pragmatic default. |
| **GAT + small-window NLM** | variance-stabilize (generalized Anscombe), then non-local means with a small search window | Buades et al. NLM; Anscombe/GAT (§3) | **USE (quality tier)** | **Measured:** cuts structure-leak roughly in half vs the fast tier at ~2.4 s/cube. The small window is deliberate — it limits structure leakage on self-similar papyrus. Quality mode when the cost is affordable. |
| **BM4D (volumetric collaborative filtering)** | group similar 3D cubes into 4D stacks, jointly shrink transform coefficients (hard-threshold → Wiener) | Maggioni et al. 2013, IEEE TIP 22(1):119–133 [research] | **OPTIONAL (not default)** | **Measured:** ~10 % better RMSE and much lower structure-leak than guided — genuinely better quality. But ~150× too slow for streaming (~30 s/cube). SKIP for the default streaming path; keep as an optional quality mode. |
| **Colored-PSD / correlated-noise BM4D variant** | BM4D variant that models the noise power spectrum (e.g. for ring/streak correlation) | Mäkinen, Marchesini & Foi 2022, J. Synchrotron Rad. (PMC9070695) [research] | **SKIP** | **Measured:** the colored-PSD variant *lost to plain BM4D on our data*. The textbook expectation (model the colored noise, win) did not hold here — see the whitening contradiction below. |
| **Noise whitening (then white-noise denoiser)** | estimate the noise power spectrum, whiten the volume, denoise as if white | standard colored-noise pipeline | **SKIP** | **Contradicts textbook.** The canonical step for colored noise *hurts*: it leaks real papyrus structure into the residual and raises RMSE ~3× at every voxel size (3× confirmed; README). One of the three headline contradictions. |
| **Total-variation (TV) denoising** | minimize `‖∇u‖₁ + λ‖u−f‖²`; piecewise-constant prior | Rudin–Osher–Fatemi 1992 | **SKIP** | **Measured:** erodes mid-band detail to ~76 %, eating exactly the contrast the deblur just restored. The piecewise-constant prior is wrong for fibrous papyrus texture. Use the guided filter instead. |

**Category verdict.** Two-tier: guided filter (fast/default) and GAT+small-window-NLM
(quality), both calibrated per-volume from the measured `var=g·I+b`. BM4D is a correct
but ~150× too-slow optional quality mode. SKIP whitening, TV, and the colored-PSD BM4D
variant — all measured worse than the simpler choices, contradicting the textbook
preference for explicit colored-noise modeling.

---

## 3. Variance stabilization

| method | what it does | citation | verdict | why (measured) |
|---|---|---|---|---|
| **Generalized Anscombe transform (GAT)** | nonlinear transform making signal-dependent (Poisson–Gaussian) noise approximately constant-variance, so one denoiser strength works everywhere | Makitalo & Foi 2013 (generalized/exact unbiased inverse Anscombe); Anscombe 1948 | **USE** | **Measured:** our noise is signal-dependent (`var=g·I+b`); GAT lets a *single* denoiser strength work across the intensity range instead of per-intensity tuning. Pairs with the quality-tier NLM and is the reason one calibration generalizes. |

**Category verdict.** USE. Variance stabilization is the enabler that turns a
signal-dependent noise problem into a single-strength denoising problem; without it the
per-volume noise model would have to vary the denoiser strength spatially.

---

## 4. Contrast enhancement

| method | what it does | citation | verdict | why (measured) |
|---|---|---|---|---|
| **MUSICA (multiscale Laplacian-pyramid amplitude remapping)** | decompose into a Laplacian pyramid, nonlinearly amplify per-band detail amplitudes, recompose | Vuylsteke & Schoeters 1994 (MUSICA); lafith MUSICA-Python ref [research source] | **USE (cosmetic layer)** | A cheap, robust legibility boost for viewing. **But measured caveat:** it **cannot replace deblur** — MUSICA remaps existing band amplitudes for display and **restores no real mid-band power**; the Paganin inverse is what restores actual transfer-function power. Use MUSICA *on top of* deblur, never instead of it. |
| **CLAHE (contrast-limited adaptive histogram equalization)** | tiled, clip-limited local histogram equalization | Zuiderveld 1994 | **OPTIONAL (legacy)** | Available (`fy_clahe2d`) but MUSICA is preferred for these volumes. Kept for compatibility; no measured advantage over MUSICA here. |
| **GLCAE (global/local contrast-adaptive enhancement)** | two-pass global+local contrast adaptation; streamable with whole-volume histogram | (implementation; generic technique) | **OPTIONAL (legacy)** | Available (`fy_glcae*`) with verified bit-identical chunked two-pass streaming, but superseded by MUSICA as the preferred cosmetic layer. |

**Category verdict.** USE MUSICA as the cosmetic contrast layer; treat CLAHE/GLCAE as
legacy. The load-bearing finding: **contrast enhancement is display-side amplitude
remapping and is not a substitute for physical deblur** — only the Paganin inverse puts
real mid-band power back.

---

## 5. Structure / sheet detection

| method | what it does | citation | verdict | why (measured) |
|---|---|---|---|---|
| **Frangi sheetness (Hessian plate filter)** | eigen-analysis of the local Hessian to score plate-/sheet-like structure (papyrus sheets) | Frangi et al. 1998 (vesselness); sheetness variants (e.g. arXiv:2304.02084) [research source] | **OPTIONAL (implemented, untuned)** | Implemented for plate detection but **not yet tuned on real PHerc data** — the research produced *no surviving verified claims* for sheet detection on papyrus (an open question, §"Genuinely open"). Promising, unvalidated; off by default. |

**Category verdict.** OPTIONAL/unvalidated. Frangi sheetness is the right family for
plate-like papyrus, but we have not measured its parameters against ground truth here, so
it carries no USE verdict yet. This is one of the genuinely open categories.

---

## 6. Resolution verification

| method | what it does | citation | verdict | why (measured) |
|---|---|---|---|---|
| **FRC / FSC (two-image Fourier ring/shell correlation)** | frequency-domain cross-correlation of two noise-independent images; resolution = where it crosses the 1/7 (0.143) threshold | Nat. Commun. 2019 (s41467-019-11024-z) [research] | **USE** | Our objective resolution yardstick. The 1/7-threshold cutoff defines where SNR-sufficient detail ends. |
| **Single-image 1FRC (Poisson-thinning)** | binomially split (p=0.5) each pixel's counts into two exactly-independent half-images, then run FRC — resolution from one acquisition | Rieger & Stallinga 2024, Opt. Express 32:21767 [research] | **USE** | **This is what proved deblur does not recover resolution.** 1FRC across 7+ volumes shows the deblur's ~2.5–3× mid-band lift produces **no resolution-cutoff shift**. Caveat [research]: 1FRC assumes Poisson-dominated data; post-Paganin/FBP/rescaled u8 are not raw counts, so apply with care — but the *relative* before/after comparison is robust and decisive. |

**Category verdict.** USE both. FRC/FSC (incl. single-image 1FRC) is the instrument that
turned "deblur looks sharper" into the measured, falsifiable claim "deblur restores
contrast, not resolution." It is the empirical backbone of this entire document.

---

## 7. Intensity / drift correction

| method | what it does | citation | verdict | why (measured) |
|---|---|---|---|---|
| **Beam-current / axial-drift correction (metadata-seeded)** | correct the slow axial (z) intensity drift left by imperfect incident-flux normalization, seeded from `machineCurrentStart/Stop` | (storage-ring beam decay; nabu intensity normalization — see esrf-pipeline.md §3) | **USE (safe complement)** | **Measured:** beam-current drift is recorded in metadata and runs **1.5 % → 13.7 %** (worst at 9.362 µm; empirical §1). The residual axial shading is a slow low-frequency correction whose magnitude is *seeded directly from the two metadata numbers* rather than estimated blind. A complement, not a physics inverse. |
| **Per-volume intensity windowing (u8↔physical attenuation)** | invert the per-volume linear export window (`target_window_f32_min/max`) to recover physical attenuation before physics ops | (metadata window; esrf-pipeline.md §9) | **USE** | **Measured:** the window is exactly linear and **differs per volume**, so a fixed intensity threshold is not physical across volumes (empirical §1). `fy_u8_to_phys` inverts it per-volume **before** deconvolution (the physics transfer is defined on attenuation, not display levels). |
| **Residual ring / stripe suppression** | heuristically suppress rings that survive into the volume (e.g. polar-domain destriping) | sarepy/Münch 2009 wavelet-FFT (sinogram methods, not invertible) [research] | **OPTIONAL (heuristic only)** | **Honest scope:** nabu's sinogram ring removal is *not invertible from the volume* (esrf-pipeline.md §4); residual rings have *no metadata model*. `fy_remove_rings` is a generic heuristic suppressor, **explicitly not a physics inverse**. Off by default; documented as heuristic. |

**Category verdict.** USE metadata-seeded drift correction and exact per-volume
de-windowing (both grounded in metadata). Residual-ring suppression is OPTIONAL and
explicitly heuristic — we do not claim ring handling as "physics."

---

## 8. Registration & multi-resolution fusion

| method | what it does | citation | verdict | why (measured) |
|---|---|---|---|---|
| **Landmark-based registration** | align scans/resolutions via matched fiducial/landmark points | (our finding; standard landmark registration) | **USE** | **Measured contradiction of the default choice:** intensity NCC registration *fails on self-similar papyrus* — the texture is too repetitive, so cross-correlation locks onto wrong matches. Landmark-based alignment is required. |
| **Multi-resolution / multi-scan fusion** | fuse multiple scans (or resolutions) of the same region to beat single-scan noise | (our finding; multi-image averaging / super-resolution fusion) | **USE (where available)** | **Measured:** fusing registered scans can beat single-scan noise (genuine SNR gain from independent acquisitions — unlike deblur, which cannot add SNR). The honest route to *more* recoverable detail is more photons/scans, not sharper inversion. |
| **Intensity-NCC registration** | align by maximizing normalized cross-correlation of intensities | standard NCC registration | **SKIP** | **Measured:** fails on self-similar papyrus (above). Use landmarks. |

**Category verdict.** USE landmark registration and multi-scan fusion. SKIP intensity-NCC
(fails on this texture). Fusion is notable as the *only* method here that genuinely adds
SNR-limited resolution — the thing deblur cannot do.

---

## 9. Compaction / downsampling

| method | what it does | citation | verdict | why (measured) |
|---|---|---|---|---|
| **Anti-aliased downsample (compaction)** | low-pass then decimate fine volumes that are oversampled relative to their information content | standard anti-aliased decimation (e.g. Lanczos/Gaussian prefilter) | **USE** | **Measured:** fine volumes are ~2× oversampled — the radial PSD shows signal dying by ~0.10 Nyq at 1.129 µm (empirical §2), i.e. most of the high-frequency grid is empty band. They are *compactible* ~2× with a proper anti-alias prefilter at no real information loss, saving storage/throughput. |

**Category verdict.** USE. The oversampling that makes deblur futile above the noise floor
is exactly what makes anti-aliased compaction free — the same measured PSD tail justifies
both "don't deblur past 0.5 Nyq" and "you can safely halve the fine grid."

---

## 10. Reconstruction (why we do not re-reconstruct)

| method | what it does | citation | verdict | why (measured) |
|---|---|---|---|---|
| **Re-reconstruction from raw projections / re-running phase retrieval at a different δ/β** | go back to projections/sinograms and re-reconstruct, hoping for more resolution | nabu pipeline; Paganin 2002 | **SKIP** | **Measured:** not worth it. The system is **PSF-limited**, and because Paganin↔FBP commute [research: Brombal 2019] our post-recon inverse already realizes whatever the linear filter can give. FRC shows ~0 % resolution gain to be had — re-reconstruction would recover *contrast we already recover on the volume*, at enormous I/O cost, with no resolution payoff. |
| **Inverting FBP/GHBP ramp filter on the volume** | undo the backprojection ramp filter from the reconstructed volume | (esrf-pipeline.md §8) | **SKIP (not recoverable)** | **Not invertible:** the ramp acts on sinogram rows *before* backprojection [research: caveats]; naive `1/|f|` re-filtering of the volume is not its inverse and would amplify low-frequency noise catastrophically. We do not implement it and do not claim to. |
| **Inverting flat/dark/ring (sinogram-domain operators)** | undo detector normalization / ring filters from the volume | (esrf-pipeline.md §1–2,4) | **SKIP (not recoverable)** | Sinogram/projection-domain operators are not encoded in the reconstructed u8 [research: caveats]. Leave alone. |

**Category verdict.** SKIP re-reconstruction. The whole fysics thesis rests on the
measured fact that post-recon inversion of the *commuting* Paganin filter captures the
available contrast, and that the residual resolution ceiling is PSF/noise-limited — so
going back to projections buys ~0 % resolution for large cost. Sinogram-domain operators
(ramp, flat/dark, rings) are simply not invertible from the volume.

---

## What the literature recommends that we have NOT done

These are well-supported in the literature and may be worth revisiting; we have not
adopted them, and in each case the reason is cost or scope, **not** a refutation.

- **BM4D as the default denoiser.** Maggioni 2013 / Mäkinen 2022 [research] make a strong
  case for volumetric collaborative filtering as SOTA quality, and we *measured* it ~10 %
  better. We do not run it by default purely on **cost** (~150× slower, ~30 s/cube;
  infeasible for streaming 20 TB). It remains our optional quality tier. Worth revisiting
  if a fast implementation appears.
- **Mäkinen 2022 correlated-noise framework for residual rings.** Jointly attacks residual
  streaks **plus** Poisson noise as correlated noise [research]. Superior in principle to
  our heuristic `fy_remove_rings`; ruled out on cost, not correctness. (Note: its
  colored-PSD BM4D core *lost* to plain BM4D in our denoising tests, so the win here would
  be specifically the ring/streak modeling, not the noise model.)
- **Learned / deep-learning reconstruction and denoising (2024+ SOTA).** Recent 3D-CNN
  denoisers exceed BM4D on narrow CT/MRI benchmarks [research: caveats], and learned
  reconstruction is an active SOTA frontier. We deliberately avoid ML in fysics (no
  hallucination, fully analytic/auditable physics). This is a scope decision, not a claim
  the ML methods are worse — for a no-hallucination viewer pipeline, an analytic inverse
  is preferable even at some quality cost.
- **A rigorous FRC-vs-as-delivered δ/β sweep per voxel size.** The reduced-regularization
  result [research: arXiv:2601.07225] suggests there is an optimal δ/β below nominal; we
  measured ~0.35× on ≤4.3 µm volumes but have not run a full per-voxel-size FRC sweep to
  pin the optimum. Future work.

## Genuinely open

Things we cannot yet call USE or SKIP because we lack a measured verdict on PHerc data:

- **Sheet/structure detection on papyrus.** Frangi sheetness (§5) is implemented but
  untuned; the research produced **no surviving verified claims** for sheet detection on
  these volumes. The right method family is clear (Hessian plate filters); the right
  parameters — and whether it beats simpler approaches for sheet tracing — are unmeasured.
- **Mosaic tile-seam handling.** The fine volume is a 19-tile fused helical mosaic
  (empirical §1; esrf-pipeline.md §7). Seam discontinuities are a real artifact class with
  no metadata transfer and are **not yet addressed**. A seam-aware blending/destriping pass
  is a candidate complement, unmeasured.
- **Optimal δ/β / reduced regularization per voxel size.** As above — direction known
  (reduce below nominal), exact optimum and its FRC payoff not yet pinned by a sweep.
- **1FRC validity on rescaled u8.** 1FRC assumes Poisson-dominated counts; our u8 volumes
  are post-Paganin/FBP/rescaled, so the absolute resolution number is only approximate
  [research: caveats]. The *relative* before/after comparison is sound (and is what we
  rely on), but an absolute-resolution calibration on these volumes is open.

---

## Sources

Verified research findings and primary literature cited above (from
`superresolution/analysis/research_result.json` unless noted):

- **Paganin et al. 2002**, J. Microsc. 206:33–40 (TIE-Hom single-distance retrieval) —
  https://onlinelibrary.wiley.com/doi/abs/10.1046/j.1365-2818.2002.01010.x
- **Gureyev/Paganin, Jan 2026**, arXiv:2601.07225 (reduced-regularization to optimize
  resolution; **Tikhonov-PSF superiority REFUTED 0-3**) — https://arxiv.org/pdf/2601.07225
- **Yu, Langer et al. 2018**, Opt. Express 26:11110 (CTF single-distance) —
  https://doi.org/10.1364/OE.26.011110 ; nabu MR !104 —
  https://gitlab.esrf.fr/tomotools/nabu/-/merge_requests/104
- **Homogeneous-object β/δ constraint** — https://pmc.ncbi.nlm.nih.gov/articles/PMC7206550/
- **Brombal et al. 2019**, J. Synchrotron Rad. mo5194 (post-recon 3D phase retrieval
  preserves resolution/contrast/CNR — the commutativity license) —
  https://journals.iucr.org/s/issues/2019/02/00/mo5194/ and arXiv:1808.05368
- **nabu phase docs** (Paganin/CTF options, δ/β, unsharp form) —
  https://www.silx.org/pub/nabu/doc/phase.html
- **Maggioni et al. 2013**, IEEE TIP 22(1):119–133 (BM4D) —
  https://pubmed.ncbi.nlm.nih.gov/22868570/ ; BM4D vs DL —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10672137/
- **Mäkinen, Marchesini & Foi 2022**, J. Synchrotron Rad. (correlated-noise BM4D for
  streaks + Poisson) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9070695/
- **He, Sun, Tang 2012**, IEEE PAMI (guided image filter) —
  https://people.csail.mit.edu/kaiming/publications/pami12guidedfilter.pdf
- **Makitalo & Foi 2013** (generalized/exact-unbiased inverse Anscombe variance
  stabilization); **Anscombe 1948**.
- **Vuylsteke & Schoeters 1994** (MUSICA multiscale Laplacian-pyramid enhancement);
  MUSICA-Python reference — https://lafith.net/posts/20220723-musica-python.html
- **Frangi et al. 1998** (vesselness/sheetness Hessian filters); sheetness variant —
  https://arxiv.org/html/2304.02084v4
- **FRC/FSC**, Nat. Commun. 2019 — https://www.nature.com/articles/s41467-019-11024-z ;
  **single-image 1FRC**, Rieger & Stallinga 2024, Opt. Express 32:21767 —
  https://home.imphys.tudelft.nl/~brieger/publications/Rieger2024.pdf
- **Rudin–Osher–Fatemi 1992** (total-variation denoising); **Zuiderveld 1994** (CLAHE).
- **Buades et al. 2005** (non-local means).
- **BM18 multiresolution phase-contrast tomography** —
  https://www.tandfonline.com/doi/full/10.1080/08940886.2024.2414724

Empirical verdicts: `docs/empirical-analysis.md` (PHerc0139, 2026-06-05). Pipeline-operator
audit and invertibility: `docs/esrf-pipeline.md`. fysics kernel behavior and rule-outs:
`README.md`.
