/* fysics.h -- fast CPU physics kernels for Vesuvius scroll volumes.
 *
 * Pure C (no SIMD intrinsics; auto-vectorizable), CPU-only, dependency-free.
 * Inverts the known ESRF/nabu reconstruction operators (Paganin phase-retrieval
 * low-pass + unsharp) to sharpen reconstructed volumes -- a physics deblur usable
 * as viewer post-processing (e.g. vc3d) or batch preprocessing.
 *
 * The transfer function matches nabu exactly (nabu/preproc/phase.py):
 *     T(f) = 1 / (1 + delta_beta * lambda * D * pi * f^2),   f in cycles/micron,
 *     lambda = 1.23984199e-3 / energy_keV  (micron),  D = distance (micron).
 * Unsharp (gaussian): out = (1+coeff)*img - coeff*blur(img).
 */
#ifndef FYSICS_H
#define FYSICS_H

#include <stddef.h>   /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- acquisition / reconstruction physics (from a volume's metadata) ---- */
typedef struct {
    double delta_beta;        /* Paganin delta/beta ratio (e.g. 1000) */
    double energy_kev;        /* beam energy (keV) */
    double distance_mm;       /* sample-detector distance (mm) */
    double pixel_um;          /* sample pixel size (micron/voxel) */
    double unsharp_sigma;     /* unsharp gaussian sigma (voxels); 0 disables */
    double unsharp_coeff;     /* unsharp coefficient; 0 disables */
    double psf_sigma_vox;     /* Gaussian SYSTEM PSF width (voxels) for the Gureyev
                               * deconvolution; 0 -> estimate ~0.5 vox default. */
} fy_physics;

/* ---- FFT (powers of two) ---- */
int  fy_is_pow2(int n);
int  fy_next_pow2(int n);
void fy_fft1d(float *re, float *im, int n, int sign);          /* sign -1 fwd, +1 inv */
void fy_fft3d(float *re, float *im, int nz, int ny, int nx, int sign);
void fy_fft3d_normalize(float *re, float *im, int nz, int ny, int nx);

/* ======================================================================
 * STREAMING for WHOLE-VOLUME processing at 20TB+ (dense u8).
 * fysics does the per-chunk MATH; the CALLER owns chunk iteration + I/O.
 *   LOCAL ops (deconv/denoise/mask): process one chunk (+halo) independently.
 *   GLOBAL ops (normalize/GLCAE-global): TWO-PASS --
 *     pass1: fy_hist_accumulate_u8() over all chunks (tiny RAM)
 *     finalize: compute the mapping ONCE from the global histogram
 *     pass2: fy_*_apply_u8() per chunk with that mapping.
 * ====================================================================== */
typedef struct { long hist[256]; long total; } fy_hist_state;

void fy_u8_to_float(const unsigned char *in, float *out, size_t n);
void fy_float_to_u8(const float *in, unsigned char *out, size_t n);

/* ---- recover PHYSICAL attenuation from the exported u8 (per-volume window) ----
 * At export, nabu linearly mapped physical f32 attenuation in [f32_min, f32_max] to
 * the stored u8 [0,255]; that window is recorded per-volume in metadata.json
 * (zarr_export.target_window_f32_min / _max). It DIFFERS per volume, so a fixed u8
 * threshold means different physical things in different volumes (e.g. physical-zero
 * "air" lands at u8 ~32-44 depending on the scan). Inverting the window puts every
 * volume into the SAME physical units, so one physical threshold/normalization is
 * consistent everywhere. The map is exact and linear (verified bit-exact round-trip).
 *   forward:  phys = f32_min + (u8/255)*(f32_max - f32_min)
 *   inverse:  u8   = clip( (phys - f32_min)/(f32_max - f32_min) * 255 + 0.5, 0, 255 )
 * Pass the two window values straight from metadata. */
void fy_u8_to_phys(const unsigned char *in, float *out, size_t n,
                   double f32_min, double f32_max);
void fy_phys_to_u8(const float *in, unsigned char *out, size_t n,
                   double f32_min, double f32_max);
/* convenience: the u8 level a given physical value maps to in this volume's window
 * (e.g. to turn a physical air threshold into the per-volume u8 air_thresh). */
float fy_phys_to_u8_level(double phys, double f32_min, double f32_max);

void fy_hist_init(fy_hist_state *s);
void fy_hist_accumulate_u8(fy_hist_state *s, const unsigned char *chunk, size_t n);
void fy_hist_merge(fy_hist_state *dst, const fy_hist_state *src);  /* parallel accum */
int  fy_hist_percentile_u8(const fy_hist_state *s, double pct);

/* global normalization (consistent intensity across the whole volume) */
void fy_norm_finalize(const fy_hist_state *s, double lo_pct, double hi_pct,
                      unsigned char *lo_out, unsigned char *hi_out);
void fy_norm_apply_u8(const unsigned char *in, float *out, size_t n,
                      unsigned char lo, unsigned char hi);

/* global GLCAE stage: compute the lambda-blended mapping once from the global
 * histogram, then apply the 256-entry lookup per chunk in pass 2. */
void fy_glcae_global_finalize(const fy_hist_state *s, int *mapping_out /*[256]*/);
/* auto air/papyrus threshold (Otsu) from the volume's histogram -- per-volume,
 * not a hardcoded constant. Returns a [0,1] fraction for fy_recipe.air_thresh. */
float fy_auto_air_thresh(const fy_hist_state *s);

/* ---- segmentation-quality metrics from a 256-bin histogram (O(256), math in C) ----
 * Haralick-Shapiro bias-guarded Fisher J over thresholds [lo,hi); returns best J, writes
 * argmax threshold to *best_t. Higher J = cleaner 2-class separation (balance term guards
 * against degenerate/mode-collapsed splits). min_count = min voxels per class. */
double fy_haralick_shapiro(const long hist[256], int lo, int hi, int min_count, int *best_t);
/* Bimodal valley depth (rails 0/254/255 excluded): 1 - h_valley/min(h_dark,h_light) in [0,1],
 * higher=deeper. Writes dark/light/valley bins (non-NULL). -1 if not bimodal. */
double fy_valley_depth(const long hist[256], int *dark_out, int *light_out, int *valley_out);

/* ---- whole-pipeline quality metrics (math in C) ----
 * Edge/sheet sharpness: mean gradient magnitude (higher=sharper boundaries). */
double fy_edge_sharpness(const float *v, int nz, int ny, int nx);
/* Dynamic-range usage: fraction of [0,1] occupied by the 1-99 percentile span (from histogram). */
double fy_dynamic_range_usage(const long hist[256]);
/* Flat-region noise: median local std over flat bright (papyrus) blocks -- the noise floor. */
double fy_flat_noise(const float *v, int nz, int ny, int nx, int blk);
void fy_glcae_global_apply_u8(const unsigned char *in, float *out, size_t n,
                              const int *mapping);

/* ---- one-call recipe: the validated processing chain ----
 * mask air (on the raw, clean valley) -> deconvolve -> re-mask (keep air clean)
 * -> optional denoise -> optional GLCAE contrast. This is the "do the good thing"
 * entry point for importers (e.g. vc3d). Toggle stages via the params; 0/<=0
 * disables a stage. Operates on a tile (with halo for deconv correctness).
 */
typedef struct {
    double deconv_reg;       /* Wiener strength (0.015 = measured knee; <=0 skips deconv) */
    int    auto_deltabeta;   /* 1 -> scale delta_beta per regime (fy_auto_deltabeta_scale):
                              * partial inversion on fine volumes, full on coarse. Recommended. */
    float  air_thresh;       /* papyrus/air intensity threshold (0=skip masking) */
    double denoise_bilateral;/* guided-denoise eps (<=0 skips; 0.05 light) */
    int    auto_denoise;     /* 1 -> set guided eps from the volume's MEASURED noise
                              * (fy_estimate_noise + fy_guided_eps_for_noise); overrides
                              * denoise_bilateral. Self-calibrates per volume (the noise
                              * level varies 1.5-3.3x scroll-to-scroll). Recommended. */
    int    do_glcae;         /* 1 -> GLCAE contrast (legacy; prefer MUSICA) */
    float  glcae_clip;       /* CLAHE clip limit for GLCAE (default 2.0) */
    int    do_musica;        /* 1 -> MUSICA multiscale contrast (preferred) */
    float  musica_p;         /* MUSICA gain exponent (<1 boosts faint; ~0.8) */
    float  air_fill;         /* what to put in masked air: 0=zero it out (black),
                              * <0=keep the smooth original (gray). Default 0. */
} fy_recipe;

fy_recipe fy_recipe_default(void);     /* the one good pipeline (punchy; glcae off) */

int fy_process(const float *in, float *out, int nz, int ny, int nx,
               const fy_physics *p, const fy_recipe *r);

/* ---- transfer functions (evaluate H at a given radial freq in cycles/voxel) ---- */
double fy_paganin_transfer(double f_cyc_per_voxel, const fy_physics *p);
double fy_unsharp_transfer(double f_cyc_per_voxel, const fy_physics *p);
double fy_recon_transfer(double f_cyc_per_voxel, const fy_physics *p); /* paganin*unsharp */

/* Noise-safe auto regularization for fy_deconvolve.
 * The plain Wiener inverse H/(H^2+reg) keeps GROWING with frequency when the recon
 * low-pass H is strong (large delta_beta * distance), so a fixed reg that is fine for
 * a 2.4um/220mm volume amplifies the noise floor ~14x on a 9.4um/1200mm volume
 * (measured on PHerc0139). This returns a reg scaled to the filter strength so the
 * Wiener gain peaks near the signal/noise crossover instead of blowing up.
 * base ~ 0.015 reproduces the hand-tuned default on the fine volumes. */
double fy_auto_reg(const fy_physics *p, double base);

/* Regime-dependent delta_beta scaling (Gureyev-analog partial inversion).
 * Measured (90 cubes, 18 volumes): fully inverting the metadata delta_beta OVER-inverts
 * fine/strong-filter volumes (<=~4.3um) -- backing off to ~0.25-0.5x raises texture,
 * lowers noise, and rescues the high band; coarse volumes (~8.6-9.4um) want full
 * inversion. Returns a multiplier in [0.25,1.0] to apply to p->delta_beta before
 * deconvolving (e.g. fy_physics q=*p; q.delta_beta *= fy_auto_deltabeta_scale(p);). */
double fy_auto_deltabeta_scale(const fy_physics *p);

/* ---- main kernel: Wiener deconvolution of the recon transfer ----
 * Sharpens `vol` (nz*ny*nx, row-major, x fastest) in place-or-out.
 *   in:  pointer to input volume (float, any range)
 *   out: pointer to output (may equal in); same shape
 *   reg: Wiener regularization (sharpening strength; lower = sharper + noisier).
 *        Pass reg <= 0 to auto-derive a noise-safe value via fy_auto_reg() (base
 *        0.015) -- RECOMMENDED, since the right reg depends on the volume's physics.
 * The Wiener gain H/(H^2+reg) is additionally capped at FY_MAX_DECONV_GAIN so a
 * mis-set reg can never amplify the high-frequency noise floor without bound.
 * Dimensions are zero-padded internally to powers of two (reflect padding) so
 * arbitrary sizes work. Returns 0 on success, nonzero on allocation failure.
 */
int fy_deconvolve(const float *in, float *out,
                  int nz, int ny, int nx,
                  const fy_physics *p, double reg);

/* ---- Gureyev-Paganin deconvolution (arXiv 2601.07225) ----
 * Goes beyond inverting the Paganin filter: it also explicitly Tikhonov-deconvolves
 * the Gaussian SYSTEM PSF (detector+source blur), which the standard Paganin over-
 * regularizes against. NOTE: like the plain deconv this restores high-frequency
 * CONTRAST -- FRC testing (Poisson-thinning, 7+ volumes) showed NO SNR-limited
 * resolution gain from any of these linear deblurs; treat as contrast/sharpness
 * restoration, not resolution recovery. The combined Fourier filter:
 *   H(k) = (1 + b'*k^2) * [ G(k) / (G(k)^2 + gamma) ]
 * where (1 + b'*k^2) is the reduced-strength Paganin inverse (b' = paganin b minus
 * the PSF contribution) and G(k)=exp(-2 pi^2 sigma^2 k^2) is the Gaussian system
 * PSF, Tikhonov-regularized by gamma. p->psf_sigma_vox sets sigma; gamma = tikhonov.
 * Reduces to the plain Paganin inverse when psf_sigma_vox=0. */
int fy_deconvolve_gureyev(const float *in, float *out,
                          int nz, int ny, int nx,
                          const fy_physics *p, double tikhonov);

/* OPERATOR-MATCHED Wiener inverse of nabu's MEASURED effective operator: inverts ONLY
 * G_psf(psf_sigma_vox) * unsharp(coeff,sigma) -- NOT the full Paganin (nabu's unsharp already
 * removed most Paganin blur; measured effective blur ~1 vox, not the ~9.8 vox naive Paganin).
 * The ONLY linear deconv shown to move BM18 data TOWARD ground truth (RMSE 0.0225) without the
 * amplitude overshoot of the plain/gureyev inverses. tikhonov ~0.05. */
int fy_deconvolve_matched(const float *in, float *out,
                          int nz, int ny, int nx,
                          const fy_physics *p, double tikhonov);


/* ---- per-volume noise model estimation (drive the denoisers from DATA) ----
 * Measured across 145 cubes / 18 scrolls: the reconstructed-volume noise is
 * (a) SIGNAL-DEPENDENT, var ~= g*intensity + b (Poisson origin reshaped by the u8
 * export window), and (b) the level varies 1.5-3.3x scroll-to-scroll AT THE SAME
 * resolution -- so it CANNOT be predicted from voxel size or hardcoded; it must be
 * estimated per-volume. This estimates that curve from the volume's own data:
 *   - local mean & variance in small windows;
 *   - bin by intensity, take a LOW percentile of local variance per bin (the noise
 *     floor; high local variance = edges/structure, rejected);
 *   - robust (IRLS) line fit var = g*I + b.
 * The line fit can be fragile (slope sometimes ill-conditioned), so the ROBUST,
 * primary output is `noise_ref` = the estimated noise std at a reference intensity
 * (`ref_intensity`, e.g. 0.4 in [0,1]); g,b are secondary. Feed `noise_ref` to the
 * denoisers (NLM h, guided eps ~ noise_ref^2, bilateral sigma_range) so denoise
 * strength self-calibrates per volume. Operates on a representative region/chunk;
 * cheap and streaming-friendly (local arithmetic, one pass). */
typedef struct {
    double g;            /* var = g*I + b  (signal-dependence slope; may be ~0) */
    double b;            /* variance intercept */
    double noise_ref;    /* PRIMARY: noise STD at ref_intensity (robust) */
    double ref_intensity;/* intensity the ref std is reported at */
    int    n_bins_used;  /* how many intensity bins had enough flat samples */
} fy_noise_model;

/* Estimate the noise model from a volume (or representative chunk). win = local
 * window edge (e.g. 5); flat_pct = percentile taken as the per-bin noise floor
 * (e.g. 10); ref_intensity in the data's units (e.g. 0.4 for [0,1] float, or ~100
 * for u8-scaled). Returns 0 on success. */
int fy_estimate_noise(const float *in, int nz, int ny, int nx,
                      int win, double flat_pct, double ref_intensity,
                      fy_noise_model *out);

/* ---- multi-energy spectral decomposition (per-voxel; needs co-registered input) ----
 * Given N CO-REGISTERED energy volumes of the same object, already on a common
 * physical-attenuation scale (fy_u8_to_phys per energy), output per-voxel material-
 * contrast channels that surface material a single energy cannot:
 *   slope_out (optional): log-log slope d ln mu / d ln E (Z proxy; more negative=higher Z)
 *   highz_out (optional): high-Z contrast score in [0,1] (steep photoelectric slope AND
 *             a confirming low-energy excess), gated so noise-dominated low-mu voxels=0.
 * PURELY LOCAL (one voxel in/out, no halo) -> ideal vc3d streaming filter. The
 * REGISTRATION of the energies to a common grid is a PREREQUISITE done upstream (see
 * fy_phase_correlate / fy_register_affine). Validated on PHerc0343P: surfaces a real,
 * coherent high-Z population invisible at any single energy. NOTE: this is material
 * CONTRAST, not a calibrated material ID (don't claim "ink" without labels).
 * mu_floor: gate; voxels with high-energy mu below it are noise -> scored 0. <=0 auto. */
int fy_spectral_decompose(const float *const *mu, const double *energies, int n_energy,
                          int nz, int ny, int nx,
                          float *slope_out, float *highz_out, double mu_floor);

/* Map an estimated noise level to a DETAIL-SAFE guided-filter eps. Measured (145
 * cubes): the detail-safe knee is eps ~= 0.014 * var of the data; expressed via the
 * noise level, eps ~= (k * noise_ref)^2 keeps mid-band detail >=95% while removing the
 * noise-floor portion. k~3 reproduces the measured knee. Use as:
 *   fy_noise_model nm; fy_estimate_noise(vol,...,&nm);
 *   fy_guided_denoise(vol,out,nz,ny,nx, 2, fy_guided_eps_for_noise(nm.noise_ref));
 * NOTE (measured): clean denoising is fundamentally limited -- at the detail-safe
 * setting only ~12-25% of noise is removed; pushing harder eats real detail. Guided
 * filter is the recommended denoiser (BM4D is ~10% better but ~150x slower; whitening
 * and TV both HURT -- do not use them). */
double fy_guided_eps_for_noise(double noise_ref);

/* ---- generalized Anscombe variance-stabilizing transform (GAT) ----
 * The noise is signal-dependent (var = g*I + b). GAT remaps intensities so the noise
 * variance becomes ~constant (~1), letting ONE denoiser strength work across the whole
 * intensity range. Exact closed-form forward + inverse (algebraic inverse, not the
 * biased Anscombe approximation). g,b come from fy_estimate_noise. If g~0 (additive
 * noise) GAT reduces to a scale -- harmless.
 *   forward: T = (2/g)*sqrt(g*I + b + 3/8 g^2)   (g>0)
 *   inverse: I = (g/4)*T^2 - b/g - (3/8)*g       (the exact unbiased inverse) */
void fy_gat_forward(const float *in, float *out, size_t n, double g, double b);
void fy_gat_inverse(const float *in, float *out, size_t n, double g, double b);

/* ---- quality denoise tier: NLM in the GAT (stabilized) domain ----
 * Measured (4 voxel sizes): estimate the noise model, GAT-stabilize, run small-window
 * non-local-means (search_radius=1, patch_radius=1) at strength ~the stabilized noise
 * (~1), inverse-GAT. Cuts structure-leak ~half vs the guided filter (0.5 -> 0.23) at
 * ~2.4s/cube -- ~55x faster than full BM4D for nearly its quality. Self-calibrating:
 * estimates (g,b,noise_ref) from the data. Use this as the "quality" denoise; use
 * fy_guided_denoise (with fy_guided_eps_for_noise) as the fast ~0.2s tier. */
int fy_denoise_quality(const float *in, float *out, int nz, int ny, int nx);

/* ---- denoising (complements deconvolution; deconv amplifies noise) ----
 * NLM: non-local means, edge/texture preserving (papyrus is self-similar -> ideal).
 *   h = filter strength (~noise level), search_radius S, patch_radius P.
 * Bilateral: cheaper edge-preserving alternative. */
int fy_nlm_denoise(const float *in, float *out, int nz, int ny, int nx,
                   double h, int search_radius, int patch_radius);
int fy_bilateral_denoise(const float *in, float *out, int nz, int ny, int nx,
                         double sigma_spatial, double sigma_range, int radius);
/* Guided filter: O(N) edge-preserving denoise (box-filter based, ~100x faster than
 * bilateral, no gradient reversal). eps = range (smaller preserves more texture),
 * radius r. The recommended FAST default for streaming large volumes. */
int fy_guided_denoise(const float *in, float *out, int nz, int ny, int nx,
                      int radius, double eps);
/* workspace variant: caller supplies ws = fy_guided_ws_floats(nz,ny,nx) contiguous floats,
 * reused across many calls to avoid per-call malloc churn (the per-tile pipeline hot path). */
int fy_guided_denoise_ws(const float *in, float *out, int nz, int ny, int nx,
                         int radius, double eps, float *ws);
size_t fy_guided_ws_floats(int nz, int ny, int nx);
/* plain box smooth (mean over (2r+1)^3); tmp = n-float scratch; in/out may alias. */
void fy_box_smooth(const float *in, float *out, float *tmp, int nz, int ny, int nx, int r);


/* ---- ring-artifact removal (heuristic, not a physics inverse) ----
 * Concentric ring artifacts from detector defects. Removed via radial-profile
 * high-pass in polar coords. center<0 -> slice center. strength in [0,1],
 * smooth_win = radial high-pass window (e.g. 30; larger -> only sharper rings). */
int fy_remove_rings(const float *in, float *out, int nz, int ny, int nx,
                    double center_x, double center_y, double strength, int smooth_win);


/* ---- papyrus/air masking (kills "air noise" -- deconv of empty gaps) ----
 * Papyrus is bright+textured, air is dark+flat. Build a soft mask and keep the
 * sharpening only on papyrus; air stays as the smooth original (or a constant). */
int fy_papyrus_mask(const float *in, float *mask, int nz, int ny, int nx,
                    float intensity_lo, float intensity_hi,
                    float var_lo, float var_hi, int radius);
int fy_apply_mask(const float *processed, const float *original, const float *mask,
                  float *out, int nz, int ny, int nx, float air_fill);
/* local standard deviation in a (2r+1)^3 box -- the "texture" feature (low=flat air,
 * high=structured papyrus). Use for intensity x texture air/papyrus separation. */
int fy_local_std(const float *in, float *out, int nz, int ny, int nx, int r);
/* one-call sharpen-without-air-noise: deconv, keep only on papyrus */
int fy_deconvolve_masked(const float *in, float *out, int nz, int ny, int nx,
                         const fy_physics *p, double reg,
                         float intensity_thresh, float var_thresh);


/* ---- contrast enhancement: GLCAE (Global+Local Contrast Adaptive Enhancement) ----
 * Grayscale adaptation of Tian & Cohen ICCV-W 2017. Handles non-uniform
 * illumination: an adaptive GLOBAL histogram blend (auto lambda) + LOCAL CLAHE,
 * fused by local-contrast x brightness. Input/output normalized [0,1] per slice. */
int fy_glcae2d(const float *in, float *out, int ny, int nx,
               int clahe_tiles, float clahe_clip);
/* MUSICA multiscale contrast (Vuylsteke-Schoeters): Laplacian-pyramid sublinear
 * detail amplification. Better than CLAHE/GLCAE for faint detail in noisy X-ray --
 * no tile/halo artifacts, doesn't penalize rare faint features. levels=pyramid
 * depth (~4), p=gain exponent (<1 boosts faint detail, ~0.7), core=noise coring
 * (0 disables). Per [0,1] slice. The recommended contrast method. */
int fy_musica2d(const float *in, float *out, int ny, int nx,
                int levels, float p, float core);
int fy_clahe2d(const float *in, float *out, int ny, int nx,
               int tiles_y, int tiles_x, int nbins, float clip_limit);


/* ---- Fourier Shell Correlation (FSC): synchrotron resolution metric ----
 * Proves a processing step actually RECOVERED resolution (not just amplified
 * noise): if FSC extends to higher frequency after processing, resolution
 * improved. fy_fsc correlates two half-volumes shell-by-shell; fy_fsc_self splits
 * one volume (checkerboard) for a reduced-reference estimate. res_frac = resolution
 * as a fraction of Nyquist at the threshold (e.g. 0.143). */
int fy_fsc(const float *vol1, const float *vol2, int nz, int ny, int nx,
           int nbins, float *freqs, float *fsc);
int fy_fsc_self(const float *vol, int nz, int ny, int nx, int nbins,
                float threshold, float *res_frac, float *freqs, float *fsc);


/* ---- z-drift / shading correction (whole-volume, the 13% beam-current drop) ----
 * Removes the slow brightness gradient along z from beam-current drift during the
 * scan. Two-pass streaming (tiny state = one scalar per slice):
 *   pass1: fy_zdrift_accumulate() per slab -> per-slice papyrus sum/count
 *   finalize: fy_zdrift_finalize() -> smoothed per-slice correction factor
 *   pass2: fy_zdrift_apply() per slab -> multiply by factor
 * Or fy_correct_zdrift() for a whole in-RAM volume. */
void fy_zdrift_accumulate(const float *chunk, int nz_slab, int ny, int nx,
                          int z0, double *sums, long *counts, float papyrus_thresh);
void fy_zdrift_finalize(const double *sums, const long *counts, int nz, float *factor);
void fy_zdrift_apply(float *chunk, int nz_slab, int ny, int nx, int z0, const float *factor);
int  fy_correct_zdrift(float *vol, int nz, int ny, int nx, float papyrus_thresh);


/* ---- sheetness filter (Frangi plate detector) -- for SEGMENTATION/UNWRAPPING ----
 * Responds to plate/sheet-like geometry (papyrus sheets); helps separate adjacent
 * sheets. NOT for ink (emphasizes geometry over faint ink texture). Output 0..1.
 * TUNING: sigma must match sheet thickness in voxels (the key knob, ~2-4 at 2.4um);
 * use multiscale for varying thickness. alpha,beta=0.5 defaults; c<=0 auto. bright=1
 * for bright sheets on dark air. Local op -> streamable with halo ~3*sigma. */
int fy_sheetness(const float *in, float *out, int nz, int ny, int nx,
                 double sigma, double alpha, double beta, double c, int bright);
int fy_sheetness_multiscale(const float *in, float *out, int nz, int ny, int nx,
                            const double *sigmas, int ns, double alpha, double beta,
                            double c, int bright);

/* ---- coherence-enhancing anisotropic diffusion (Weickert 1999) -- CLEAN SHEETS --
 * Smooths ALONG papyrus sheet surfaces but NOT across them: noise drops, sheets
 * become smooth/continuous for tracing, and crucially the THIN DARK GAPS between two
 * touching layers are PRESERVED (CED never merges adjacent sheets -- a plain Gaussian
 * blur WOULD fill the gap). Volume-in -> volume-out.
 *
 * Builds the structure tensor J_rho = G_rho*(grad u_sigma (x) grad u_sigma^T), eigen-
 * decomposes it per voxel (sheet -> one large eigenvalue = the normal, two small =
 * in-plane), and diffuses STRONG along the in-plane directions but WEAK along the
 * sheet normal using Weickert's coherence-enhancing eigenvalue map. Evolves the
 * explicit scheme u += tau*div(D grad u) for n_iters steps.
 *   sigma   : noise scale presmoothing for the gradient (~0.5-1 vox; <0 ->0.7)
 *   rho     : integration / sheet-coherence scale (~2-4 vox; <=0 ->3)
 *   tau     : explicit time step, 3D-stable <= ~0.12 (<=0 ->0.10)
 *   n_iters : explicit iterations, 3-15 (<1 ->5)
 *   coherence_alpha : Weickert alpha = base diffusivity across the sheet normal
 *             (0<alpha<1, ~0.001-0.01; smaller = harder gap preservation; <=0 ->0.001)
 * LOCAL op -> streamable/tileable. Per-side halo: ~3*sigma + 3*rho + n_iters + 2 vox
 * (see fy_coherence_diffusion_halo). Returns 0 on success, 1 on alloc failure. */
int fy_coherence_diffusion(const float *in, float *out, int nz, int ny, int nx,
                           double sigma, double rho, double tau, int n_iters,
                           double coherence_alpha);
/* AUTO-CALIBRATED "clean sheets for tracing" -- no-knobs entry point.
 * Derives sigma from the volume's measured noise (fy_estimate_noise), rho from the
 * measured cross-sheet layer spacing (autocorrelation), tau from the scheme's
 * stability max, and n_iters from `strength` (1=gentle/12, 2=normal/24, 3=strong/40).
 * Validated to denoise sheets while preserving inter-sheet gaps (~92% gap-depth at
 * 'normal' vs a matched Gaussian's ~67%). This is the recommended call for vc3d. */
int fy_coherence_diffusion_auto(const float *in, float *out, int nz, int ny, int nx,
                                int strength);
/* Recommended per-side halo (voxels) to feed a tile so the result is seam-free. */
int fy_coherence_diffusion_halo(double sigma, double rho, int n_iters);

/* ===== IN-PLANE (surface-tangent) TEXTURE ENHANCER ========================
 * COMPLEMENT of coherence diffusion: where CED SMOOTHS along sheets, this
 * HIGH-PASSES along sheets. It LIFTS the fine in-plane "crackle" surface texture
 * on papyrus (the faint texture ink sits in, per Vesuvius Grand Prize findings)
 * while SUPPRESSING cross-sheet/layering variation and NOT amplifying noise.
 *
 * It is a STEERED, NOISE-GATED unsharp mask built on the SAME orientation field as
 * CED (structure tensor J_rho, eigen-decomposed -> sheet normal + two in-plane
 * tangents): low-pass ONLY within the tangent plane, detail = u - inplane_lowpass,
 * soft-threshold (core) the detail at the noise level, out = u + gain*cored_detail.
 * The in-plane (not isotropic) low-pass is why a cross-sheet STEP (layering) is NOT
 * boosted while in-plane texture IS -- a plain unsharp mask boosts the step too.
 *
 * HONEST SCOPE: NOT an ink detector and not validated as one (no ink labels). It is
 * the right transform IF ink is an in-plane texture signal; the proven, MEASURABLE
 * claims are: in-plane texture-band energy up, texture/noise ratio up (gating),
 * cross-sheet/layering variation NOT amplified.
 *
 * Explicit knobs: sigma (gradient presmooth, ~0.6-1), rho (orientation integration
 * scale, ~2-4), gain (>0, ~0.5-3), inplane_scale (steered low-pass radius in vox,
 * ~1.5-4: the texture band is detail finer than this within the plane), noise_floor
 * (detail magnitude below this is soft-thresholded/cored; in detail-signal units.
 * noise_floor==0 disables gating; noise_floor<0 is a SENTINEL = auto-derive the gate
 * from the detail's own robust scale (1.4826*MAD), the recommended self-calibrating
 * mode -- it measures the detail-band noise directly rather than mapping an absolute
 * intensity-domain noise model that the bright sheets/layering would corrupt).
 * Returns 0 on success, 1 on alloc failure. */
int fy_texture_enhance(const float *in, float *out, int nz, int ny, int nx,
                       double sigma, double rho, double gain,
                       double inplane_scale, double noise_floor);
/* AUTO-CALIBRATED "enhance in-plane texture over noise and over layering" -- no-knobs
 * entry point. Derives sigma & noise_floor from the volume's measured noise
 * (fy_estimate_noise for sigma; the gate is set from the detail's own MAD), rho from the
 * measured cross-sheet layer spacing (autocorrelation), inplane_scale from the measured
 * in-plane texture scale (in-plane autocorrelation), and gain from `strength`
 * (1=gentle/0.8, 2=normal/1.5, 3=strong/2.5). Recommended call. */
int fy_texture_enhance_auto(const float *in, float *out, int nz, int ny, int nx,
                            int strength);
/* Recommended per-side halo (voxels) to feed a tile so the result is seam-free:
 * 3*sigma (presmooth) + 3*rho (integration box) + ceil(inplane_scale) + margin. */
int fy_texture_enhance_halo(double sigma, double rho, double inplane_scale);

/* recommended halo (voxels) for tiled/viewer use: the kernel's spatial half-extent.
 * Process a viewed region plus this margin, then keep only the inner region. */
int fy_kernel_halo(const fy_physics *p);


/* ===== LAYER 1: affine resampler + intensity-based registration ============
 * Register multiple CT scans of the SAME object taken at different
 * resolution/energy/time (e.g. PHerc0139 1.129um vs 2.399um) so they can be
 * fused. This layer is the GLOBAL affine part (scale from differing voxel size,
 * rotation, translation, shear). Deformable warping (temperature/movement) is
 * Layer 2 (Demons) -- the trilinear sampler below is factored out so a
 * displacement-field warp can reuse it.
 *
 * Geometry: row-major, x fastest, idx=(z*ny+y)*nx+x. The 3x4 affine M (12
 * doubles, row-major) maps OUTPUT voxel coords -> INPUT voxel coords (backward/
 * pull warp), in (z,y,x) ordering. Translation column = M[3],M[7],M[11].
 */

/* Trilinear sample of `in` at continuous (z,y,x) in input-voxel units.
 * Out-of-bounds (outside [0,n-1]) -> 0. The reusable interpolation primitive;
 * Layer 2's displacement-field warp calls this directly. */
float fy_sample_trilinear(const float *in, int nz, int ny, int nx,
                          float z, float y, float x);

/* Resample `in` (nz,ny,nx) into `out` (onz,ony,onx) under 3x4 affine M mapping
 * output->input voxel coords. Trilinear; out-of-bounds -> 0. The workhorse:
 * registration optimizes M, then warps the moving volume with it. */
int fy_warp_affine(const float *in, float *out, int nz, int ny, int nx,
                   const double *M, int onz, int ony, int onx);

/* 2x anti-aliased downsample for an image PYRAMID (coarse-to-fine registration):
 * gaussian blur (sigma~0.8) then decimate. `out` must hold (nz/2)*(ny/2)*(nx/2)
 * floats; output dims are returned in *onz,*ony,*onx. */
int fy_downsample2x(const float *in, float *out, int nz, int ny, int nx,
                    int *onz, int *ony, int *onx);

/* Normalized Cross-Correlation between `fixed` and `moving` warped by M, over
 * the in-bounds overlap. Range [-1,1] (1=perfect); returns -2 if overlap is too
 * small. NCC is invariant to affine intensity change (a*I+b) -> robust to the
 * brightness/contrast difference between two energies (plain SSD is wrong here).
 * TODO(MI): for truly different MODALITIES, Mutual Information is more robust;
 * a histogram-based fy_mi_warped would slot in as an alternative metric here. */
double fy_ncc_warped(const float *fixed, const float *moving,
                     int nz, int ny, int nx, const double *M);

/* Coarse-to-fine (pyramid) intensity registration: optimize the affine (or
 * rigid: rotation+translation+isotropic-scale, 7 dof) to MAXIMIZE
 * NCC(fixed, warp(moving, M)). fixed and moving share the grid (nz,ny,nx);
 * resample differing-voxel-size scans onto a common grid first (or seed the
 * scale into M_out). M_out is BOTH input (initial guess; identity for same grid,
 * a scale-ratio diagonal for a known voxel ratio) and output (optimized 3x4 map
 * output->input). rigid_only=1 for rigid+isotropic-scale, 0 for full affine.
 * Returns 0 on success. */
int fy_register_affine(const float *fixed, const float *moving,
                       int nz, int ny, int nx, double *M_out, int rigid_only);

/* ===== SUB-VOXEL TRANSLATION + MULTIMODAL METRIC (phasecorr.c) ============
 * The local-patch refinement that bootstraps a coarse landmark seed to sub-voxel
 * on the self-similar laminar scroll. Plain intensity NCC is non-discriminative
 * GLOBALLY (papyrus is self-similar) but a SINGLE textured patch is not
 * self-similar with itself under a small shift, so a LOCAL metric on a textured
 * patch is sharp. Two tools provide that locally:
 *   - fy_phase_correlate: sub-voxel translation via the Fourier shift theorem.
 *     Robust to contrast/brightness differences (cross-energy) because the
 *     cross-power spectrum is normalized to unit magnitude -- only phase (=shift)
 *     survives. Use it to measure the residual local displacement of a patch.
 *   - fy_mutual_information: the gold-standard MULTIMODAL similarity. NCC assumes
 *     a LINEAR intensity relation between the two scans; MI assumes only a
 *     statistical dependence, so it is sharp even when the two energies map
 *     intensity nonlinearly. */

/* Sub-voxel translation between two SAME-SIZE volumes via phase correlation.
 * Returns the shift (dz,dy,dx) such that moving(p) ~ fixed(p - shift), i.e. the
 * displacement that, applied to `moving`, aligns it onto `fixed` (moving is
 * `fixed` shifted by +shift). Dims need NOT be powers of two -- the volume is
 * internally zero-padded (after mean-removal + a separable Hann window to kill
 * edge wrap-around) to the next pow2. The cross-power spectrum peak is localized
 * to integer voxel, then refined to sub-voxel by 1-D parabolic interpolation on
 * the three samples straddling the peak along each axis.
 *   shift  : out, 3 doubles (dz,dy,dx) in voxels.
 *   peak   : optional out, the normalized correlation peak height in [0,1]
 *            (a match-confidence; reject patches below a threshold). May be NULL.
 *   window : 1 = apply Hann window (recommended for non-periodic patches),
 *            0 = none (use when the patch is already apodized/periodic).
 * Returns 0 on success, 1 on bad args / degenerate (flat) input. */
int fy_phase_correlate(const float *fixed, const float *moving,
                       int nz, int ny, int nx,
                       double *shift, double *peak, int window);

/* Mutual Information between `fixed` and `moving` warped by M, over the in-bounds
 * overlap, from a joint intensity histogram (nbins x nbins). Both images are
 * scaled to [0,nbins) using their OWN overlap min/max (so MI is invariant to any
 * monotonic-affine intensity change of either image, like NCC, but unlike NCC it
 * does NOT assume the relation is linear -- it captures arbitrary statistical
 * dependence, the reason MI is the multimodal gold standard).
 *   MI = sum p(a,b) log( p(a,b) / (p(a) p(b)) ), in NATS; 0 = independent,
 *   larger = more dependent (better aligned). Returns -1.0 if overlap < 32 voxels.
 *   nbins : 32-64 typical (too many -> sparse histogram noise; too few -> blurred).
 * Same geometry/overlap convention as fy_ncc_warped. */
double fy_mutual_information(const float *fixed, const float *moving,
                            int nz, int ny, int nx, const double *M, int nbins);

/* ===== LAYER 2: deformable (non-rigid) registration via Demons ============
 * After affine removes the global transform, two scans of the SAME scroll
 * still differ by a SMOOTH physical warp (thermal/humidity expansion, creep,
 * remounting). Demons recovers a per-voxel displacement field that pulls the
 * affine-aligned moving volume onto the fixed one.
 *
 * Field convention: u=(uz,uy,ux) lives on the FIXED grid and is a PULL/backward
 * displacement in voxel units. The warped moving is
 *     warped(z,y,x) = moving( z+uz, y+uy, x+ux )
 * via fy_sample_trilinear. A zero field is the identity warp.
 */

/* Resample `in` through a displacement field (uz,uy,ux), all on the (nz,ny,nx)
 * grid: out(z,y,x) = trilinear-sample(in, z+uz, y+uy, x+ux). OOB -> 0. A zero
 * field reproduces `in`. The deformable analogue of fy_warp_affine. */
int fy_warp_field(const float *in, float *out, int nz, int ny, int nx,
                  const float *ux, const float *uy, const float *uz);

/* Coarse-to-fine Demons deformable registration. fixed & moving share the
 * (nz,ny,nx) grid -- run affine first and pass the affine-aligned moving here.
 * Variant: symmetric (active-forces) Thirion demons; each iteration computes the
 * intensity-difference * gradient force (using BOTH fixed and moving gradients),
 * adds it to the field, then Gaussian-smooths the whole field (regularization).
 * Solved on an image pyramid (fy_downsample2x), coarse->fine with x2 field
 * upsampling, for capture range + smoothness.
 *   ux/uy/uz : in = initial field (pass zeros for fresh), out = recovered field.
 *   n_iters  : iterations PER pyramid level (typical 50-150).
 *   field_sigma : Gaussian regularization sigma in voxels (elasticity; larger =
 *                 smoother/stiffer; typical 1.0-2.0).
 *   step     : displacement step scale per iteration (typical 1.0-2.0).
 * Returns 0 on success. */
int fy_register_demons(const float *fixed, const float *moving,
                       int nz, int ny, int nx,
                       float *ux, float *uy, float *uz,
                       int n_iters, double field_sigma, double step);

/* Convenience: Layer 1 THEN Layer 2. Recovers the affine map M_out, warps moving
 * onto the fixed grid with it, then recovers the residual deformation field
 * ux/uy/uz on the affine-aligned pair. The fully-registered moving volume is
 *   fy_warp_field( fy_warp_affine(moving, M_out), ux,uy,uz ).
 * M_out is in/out (initial guess / affine result); ux/uy/uz are pre-allocated
 * (nz*ny*nx each) and zeroed internally. Returns 0 on success. */
int fy_register_full(const float *fixed, const float *moving,
                     int nz, int ny, int nx, double *M_out, int rigid_only,
                     float *ux, float *uy, float *uz,
                     int n_iters, double field_sigma, double step);

/* ===== LANDMARK affine fit + MULTI-RESOLUTION FUSION (fuse.c) ==============
 * Intensity NCC/MI is non-discriminative on the self-similar laminar scroll at a
 * common COARSE scale, so the trustworthy registration of two same-scroll scans is
 * LANDMARK/feature based: detect/match distinctive 3D points across the two scans,
 * then fit the global transform from the matched pairs. fy_affine_from_points is
 * that fit (with RANSAC for mismatch robustness). See FUSION.md. */

/* Fit a 3x4 affine map dst<-src from N matched 3D point pairs.
 *   src,dst : n*3 doubles in (z,y,x) order (match the index convention).
 *   model   : 0 = full affine (12 dof, least squares), 1 = SIMILARITY
 *             (rotation + single isotropic scale + translation, 7 dof, closed-form
 *             Umeyama) -- the correct model for two scans of one rigid object
 *             differing only by voxel-size ratio + remount pose.
 *   ransac_iters  : >0 runs RANSAC (random minimal subsets, keep max-inlier model,
 *                   refit on inliers) for robustness to wrong correspondences; 0
 *                   does plain least squares on all points.
 *   inlier_thresh : RANSAC inlier distance in dst voxel units (ignored if no RANSAC).
 *   M_out   : 3x4 row-major result mapping a src point to its dst point
 *             (q = M_out[:, :3] @ p + M_out[:, 3]). To PULL the src volume onto the
 *             dst grid with fy_warp_affine (which wants a dst->src map), fit with
 *             src=fixed/dst=moving, or invert M_out.
 *   inlier_mask   : optional n ints (may be NULL), 1 = inlier.
 *   resid_rms_out : optional RMS residual (dst units) over inliers.
 * Returns 0 on success, 1 on failure (too few/degenerate points). */
int fy_affine_from_points(const double *src, const double *dst, int n,
                          int model, int ransac_iters, double inlier_thresh,
                          double *M_out, int *inlier_mask, double *resid_rms_out);

/* Fuse a FINE and a COARSE scan ALREADY resampled onto the SAME (nz,ny,nx) grid.
 * Frequency-split at split_sigma (gaussian): HIGH band comes from the fine scan
 * only; LOW band is an inverse-variance-weighted average of the two INDEPENDENT
 * measurements (so the shared low/mid band is DENOISED -- the payoff of fusion).
 * The coarse low band is intensity-matched (affine a*x+b least squares) to the fine
 * low band first, so the energy/contrast difference doesn't bias the average.
 *   var_fine,var_coarse : measured per-scan low-band noise variances; weights are
 *     w_fine=var_coarse/(var_fine+var_coarse). If both <=0 -> plain 0.5/0.5 average.
 *   high_gain : scale on the fine high band (1.0 keeps the fine detail as-is).
 *   mask : optional u8 (!=0 = valid overlap); outside it out=fine (no fusion). NULL=all.
 *   out = low_fused + high_gain*high_fine.
 * Returns 0 on success. */
int fy_fuse_multiscale(const float *fine, const float *coarse,
                       const unsigned char *mask, int nz, int ny, int nx,
                       double split_sigma, double var_fine, double var_coarse,
                       double high_gain, float *out);

/* ---- compaction: downsample OVERSAMPLED volumes (lossless of resolved detail) ----
 * Measured: fine (~1.1um) scroll volumes are ~2x oversampled -- their resolved detail
 * fits a coarser grid, so downsampling ~1.75-2x/axis loses ~nothing (8x fewer voxels,
 * ~18TB saved on one 1.1um volume). Mid/coarse (>=2.4um) volumes are critically sampled
 * -- do NOT compact them. ORDER MATTERS: if deblurring, deblur at FULL resolution FIRST,
 * then downsample (downsampling discards the high-freq band the deblur restores --
 * deblur-then-downsample keeps ~86% of restored contrast vs ~21% for downsample-raw).
 *
 * Anti-aliased downsample by an arbitrary factor (gaussian blur sigma~0.5*factor, then
 * trilinear decimate). out size = ceil(dim/factor); pass back via onz/ony/onx. */
int fy_downsample(const float *in, float *out, int nz, int ny, int nx,
                  double factor, int *onz, int *ony, int *onx);

/* Recommend the largest SAFE downsample factor for THIS volume from the data: sweeps
 * factors, measures the round-trip (downsample->upsample) L2 error over TEXTURED
 * regions only (flat/air regions can't lose detail and would mislead), and returns the
 * largest factor whose worst-textured-region error stays <= err_budget (e.g. 0.03).
 * Returns 1.0 if the volume is critically sampled (no safe shrink). Use the WORST-case
 * (not average) so no textured region loses detail. */
double fy_recommend_downsample(const float *in, int nz, int ny, int nx,
                               double err_budget);

/* ===== LOCAL zarr v2 raw-u8 I/O (zarr_io.c) ============================= */
typedef struct {
    char root[1024];
    long shape[3];      /* z,y,x voxels */
    long chunk[3];      /* chunk dims */
    unsigned char fill; /* fill_value */
} fy_zarr;

int  fy_zarr_open(fy_zarr *z, const char *root);                    /* read .zarray */
int  fy_zarr_create(fy_zarr *z, const char *root, const long shape[3], const long chunk[3]);
int  fy_zarr_read(const fy_zarr *z, long z0, long y0, long x0,
                  long dz, long dy, long dx, unsigned char *out);   /* assemble region */
int  fy_zarr_write_chunk(const fy_zarr *z, long cz, long cy, long cx,
                         const unsigned char *buf, long bz, long by, long bx);

/* ===== pipeline orchestration (pipeline.c) ============================== */
typedef struct {
    /* physics (from metadata.json, parsed by the tiny python reader, passed on the CLI) */
    double delta_beta, energy_kev, distance_mm, pixel_um, unsharp_sigma, unsharp_coeff;
    double machine_current_start, machine_current_stop;
    double window_lo, window_hi;          /* export window (f32), if known */
    /* tuned/calibrated */
    int    do_deconv;                     /* store deconv? (BM18: 0 -- view-time only) */
    int    use_matched_deconv;
    double psf_sigma_vox, deconv_tikhonov;
    double guided_eps;                    /* from (3*flat_nf)^2 */
    int    do_air_zero; int air_cut_u8, air_cut_band; double air_thresh;
    double air_cut_aggr;   /* 0 conservative (void-peak) .. 1 aggressive (valley) */
    double denoise_k;      /* eps = (denoise_k*flat_nf)^2; 0 -> default 4.2 */
    int    scratch_passes;
    int    do_normalize; int norm_lo, norm_hi;
    int    do_zdrift; float *zdrift_factor;  /* len = Z, or NULL */
    double psf_p5, psf_med;   /* measured PSF sigma map (drives the auto-deconv gate) */
    int    do_musica; double musica_p; int musica_levels; double musica_core;
    int    halo;
    /* resolved calibration STATE (set by fy_calibrate; consumed by fy_process_chunk) */
    int    have_norm, have_zdrift, have_dec_range;
    double dec_lo, dec_hi;    /* global deconv-output rescale range */
    long   vol_z;             /* full-volume Z (for zdrift_apply absolute-z indexing) */
} fy_pipeline_cfg;

/* ----- FUSED export support: calibrate once, then process arbitrary chunks on demand -----
 * fy_calibrate: run the calibration (PSF/eps/air-cut/gates + norm/zdrift global stats) on the
 * input zarr, filling cfg (incl. have_norm/have_zdrift/zdrift_factor/dec range). Call once.
 * fy_process_chunk: process the inner `tile`^3 at (z0,y0,x0) using the calibrated cfg, reading a
 * halo'd region from the open zarr; write the inner tile (tz*ty*tx u8) to out. Thread-safe given
 * a per-thread scratch (pass NULL ws to malloc internally). Returns 0; sets *all_air if empty. */
int fy_calibrate(const char *in_root, fy_pipeline_cfg *cfg, int tile, int verbose);
int fy_process_chunk(const fy_zarr *zin, const fy_pipeline_cfg *cfg,
                     long z0, long y0, long x0, int tile,
                     unsigned char *out, int *out_tz, int *out_ty, int *out_tx, int *all_air);

/* run the whole 2-pass pipeline zarr->zarr; cfg physics pre-filled, rest calibrated inside. */
int fy_run_pipeline(const char *in_root, const char *out_root, fy_pipeline_cfg *cfg,
                    int tile, int verbose);


#ifdef __cplusplus
}
#endif

#endif /* FYSICS_H */
