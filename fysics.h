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
void fy_glcae_global_apply_u8(const unsigned char *in, float *out, size_t n,
                              const int *mapping);

/* ---- one-call recipe: the validated processing chain ----
 * mask air (on the raw, clean valley) -> deconvolve -> re-mask (keep air clean)
 * -> optional denoise -> optional GLCAE contrast. This is the "do the good thing"
 * entry point for importers (e.g. vc3d). Toggle stages via the params; 0/<=0
 * disables a stage. Operates on a tile (with halo for deconv correctness).
 */
typedef struct {
    double deconv_reg;       /* Wiener strength (0.02-0.03 good; <=0 skips deconv) */
    float  air_thresh;       /* papyrus/air intensity threshold (0=skip masking) */
    double denoise_bilateral;/* guided-denoise eps (<=0 skips; 0.05 light) */
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

/* ---- Gureyev-Paganin deconvolution (arXiv 2601.07225) -- published SOTA ----
 * Resolution recovery that goes beyond inverting the Paganin filter: it also
 * explicitly Tikhonov-deconvolves the Gaussian SYSTEM PSF (detector+source blur),
 * which the standard Paganin over-regularizes against. The combined Fourier filter:
 *   H(k) = (1 + b'*k^2) * [ G(k) / (G(k)^2 + gamma) ]
 * where (1 + b'*k^2) is the reduced-strength Paganin inverse (b' = paganin b minus
 * the PSF contribution) and G(k)=exp(-2 pi^2 sigma^2 k^2) is the Gaussian system
 * PSF, Tikhonov-regularized by gamma. p->psf_sigma_vox sets sigma; gamma = tikhonov.
 * Reduces to the plain Paganin inverse when psf_sigma_vox=0. */
int fy_deconvolve_gureyev(const float *in, float *out,
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

/* recommended halo (voxels) for tiled/viewer use: the kernel's spatial half-extent.
 * Process a viewed region plus this margin, then keep only the inner region. */
int fy_kernel_halo(const fy_physics *p);

#ifdef __cplusplus
}
#endif

#endif /* FYSICS_H */
