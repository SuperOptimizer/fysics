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
    double denoise_bilateral;/* bilateral sigma_range (<=0 skips; 0.08-0.2) */
    int    do_glcae;         /* 1 -> GLCAE contrast (per XY slice) */
    float  glcae_clip;       /* CLAHE clip limit for GLCAE (default 2.0) */
} fy_recipe;

fy_recipe fy_recipe_default(void);     /* the one good pipeline (punchy; glcae off) */

int fy_process(const float *in, float *out, int nz, int ny, int nx,
               const fy_physics *p, const fy_recipe *r);

/* ---- transfer functions (evaluate H at a given radial freq in cycles/voxel) ---- */
double fy_paganin_transfer(double f_cyc_per_voxel, const fy_physics *p);
double fy_unsharp_transfer(double f_cyc_per_voxel, const fy_physics *p);
double fy_recon_transfer(double f_cyc_per_voxel, const fy_physics *p); /* paganin*unsharp */

/* ---- main kernel: Wiener deconvolution of the recon transfer ----
 * Sharpens `vol` (nz*ny*nx, row-major, x fastest) in place-or-out.
 *   in:  pointer to input volume (float, any range)
 *   out: pointer to output (may equal in); same shape
 *   reg: Wiener regularization (sharpening strength; lower = sharper + noisier)
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
int fy_clahe2d(const float *in, float *out, int ny, int nx,
               int tiles_y, int tiles_x, int nbins, float clip_limit);

/* recommended halo (voxels) for tiled/viewer use: the kernel's spatial half-extent.
 * Process a viewed region plus this margin, then keep only the inner region. */
int fy_kernel_halo(const fy_physics *p);

#ifdef __cplusplus
}
#endif

#endif /* FYSICS_H */
