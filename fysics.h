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
    double psf_sigma_vox;     /* Gaussian SYSTEM PSF width (voxels) for the matched
                               * deconvolution; 0 -> estimate ~0.5 vox default. */
    double psf_sigma_z_vox;   /* z-axis PSF width if ANISOTROPIC (helical scans measure
                               * ~1.17x broader in z); 0 -> isotropic (= psf_sigma_vox).
                               * Used by fy_deconvolve_matched only. */
} fy_physics;

/* ---- FFT (sizes 2^k and 3*2^k) ---- */
int  fy_is_pow2(int n);
int  fy_next_pow2(int n);
int  fy_next_fft_size(int n);   /* smallest supported FFT size >= n (2^k or 3*2^k) */
void fy_fft1d(float *re, float *im, int n, int sign);          /* sign -1 fwd, +1 inv */
void fy_fft3d(float *re, float *im, int nz, int ny, int nx, int sign);
void fy_fft3d_normalize(float *re, float *im, int nz, int ny, int nx);

/* ======================================================================
 * STREAMING for WHOLE-VOLUME processing at 20TB+ (dense u8).
 * fysics does the per-chunk MATH; the CALLER owns chunk iteration + I/O.
 *   LOCAL ops (deconv/denoise/mask): process one chunk (+halo) independently.
 *   GLOBAL ops (normalize): TWO-PASS --
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

/* Bimodal valley depth (rails 0/254/255 excluded): 1 - h_valley/min(h_dark,h_light) in [0,1],
 * higher=deeper. Writes dark/light/valley bins (non-NULL). -1 if not bimodal. */
double fy_valley_depth(const long hist[256], int *dark_out, int *light_out, int *valley_out);

/* Flat-region noise: median local std over flat bright (papyrus) blocks -- the noise floor. */
double fy_flat_noise(const float *v, int nz, int ny, int nx, int blk);

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
 * Dimensions are reflect-padded internally to FFT-friendly sizes (2^k or 3*2^k)
 * so arbitrary sizes work. Returns 0 on success, nonzero on allocation failure.
 */
int fy_deconvolve(const float *in, float *out,
                  int nz, int ny, int nx,
                  const fy_physics *p, double reg);

/* OPERATOR-MATCHED Wiener inverse of nabu's MEASURED effective forward operator
 * (system PSF x unsharp boost: the unsharp already largely undid the Paganin blur,
 * so the net volume blur is only ~1 vox). p->psf_sigma_vox sets the PSF width
 * (measure it; pipeline pass 1 does); unsharp coeff/sigma come from metadata.
 * tikhonov <= 0 -> 0.05. The pipeline's default STORED deconv. */
int fy_deconvolve_matched(const float *in, float *out,
                          int nz, int ny, int nx,
                          const fy_physics *p, double tikhonov);

/* recommended halo (voxels) for tiled/viewer use: the kernel's spatial half-extent.
 * Process region+halo, keep the inner region -> seam-free tiling. */
int fy_kernel_halo(const fy_physics *p);

/* ---- guided filter: the recommended fast edge-preserving denoiser ----
 * (He, Sun, Tang; O(N) box-filter based). eps = range parameter (smaller preserves
 * more texture; set from the measured noise via fy_guided_eps_for_noise), radius r
 * voxels. Apply AFTER deconvolution. */
int fy_guided_denoise(const float *in, float *out, int nz, int ny, int nx,
                      int radius, double eps);
/* workspace variant: caller supplies ws = fy_guided_ws_floats(nz,ny,nx) contiguous
 * floats, reused across calls (the per-tile pipeline hot path). */
int fy_guided_denoise_ws(const float *in, float *out, int nz, int ny, int nx,
                         int radius, double eps, float *ws);
size_t fy_guided_ws_floats(int nz, int ny, int nx);
/* FAST variant (He & Sun 2015): coefficients computed on an s-times-decimated grid
 * and trilinearly upsampled -- ~s^3 less box-filter work, visually identical at s=2.
 * s<=1 (or a tile too small to decimate) falls back to the exact path. Same ws. */
int fy_guided_denoise_fast_ws(const float *in, float *out, int nz, int ny, int nx,
                              int radius, double eps, int s, float *ws);
/* plain box smooth (mean over (2r+1)^3); tmp = n-float scratch; in/out may alias. */
void fy_box_smooth(const float *in, float *out, float *tmp, int nz, int ny, int nx, int r);
/* map a measured noise level to a detail-safe guided eps (~(3*noise_ref)^2) */
double fy_guided_eps_for_noise(double noise_ref);

/* ---- MUSICA multi-scale contrast enhancement (per-slice viewing aid) ----
 * Laplacian pyramid with sublinear gain |x|^p; p in (0,1], levels ~4, core
 * protects the low-contrast center band. */
int fy_musica2d(const float *in, float *out, int ny, int nx,
                int levels, float p, float core);

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
 * denoiser (guided eps ~ noise_ref^2, fy_guided_eps_for_noise) so denoise
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

/* ---- noise-correlation anisotropy (PSF z/xy ratio) ----
 * The system PSF shapes the noise correlation: lag-1 autocorrelation of the
 * residual (voxel - local mean) in FLAT 8^3 blocks, per axis, gives the relative
 * PSF width (rho = exp(-1/(4 sigma^2)) for a Gaussian PSF). Helical BM18 scans
 * measure ~1.17x broader in z (PHerc0139 2.4um). Returns 0 and writes rho_z and
 * rho_xy (mean of y,x); nonzero if too few flat blocks. Feed
 * sigma_z/sigma_xy = sqrt(ln(1/rho_xy)/ln(1/rho_z)) to fy_physics.psf_sigma_z_vox. */
int fy_noise_aniso(const float *v, int nz, int ny, int nx,
                   double *rho_z, double *rho_xy);

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

/* ===== detect-then-subtract residual ring removal (dering.c) ============
 * Rings = angularly-invariant radial features centered on the ROTATION AXIS
 * (metadata rotation_axis_position; BM18 places it at the slice center).
 * Streaming 2-pass: accumulate per (z-slab, sector, radius) sums from raw u8
 * tiles; finalize runs a per-radius SECTOR SIGN-CONSISTENCY VOTE (a spiral
 * papyrus wrap drifts in radius with angle and fails the vote; a true ring
 * does not -- verified on PHerc0139 2.4um); apply subtracts only the detected
 * component. fy_dering_apply is a LOCAL op (tileable, no halo). */
typedef struct {
    long Z, Y, X;
    double cy, cx;       /* rotation axis (voxels); init with <0 -> volume center */
    int slab_z, nslab;   /* z-slab granularity (rings drift slowly with z) */
    int ns, nr;          /* angular sectors, radial bins */
    float *sum;          /* [nslab][ns][nr] intensity sums (u8 units) */
    unsigned int *cnt;   /* [nslab][ns][nr] sample counts */
    float *ring;         /* [nslab][nr] detected ring profile (u8), after finalize */
    int have_rings;
} fy_dering;

int  fy_dering_init(fy_dering *d, long Z, long Y, long X,
                    double cy, double cx, int slab_z, int ns);
void fy_dering_free(fy_dering *d);
/* per-thread single-slab scratch + merge (caller locks the merge) */
int  fy_dering_tile_init(fy_dering *t, const fy_dering *d);
void fy_dering_tile_reset(fy_dering *t);
void fy_dering_accumulate_u8(fy_dering *t, const unsigned char *buf,
                             long y0, long x0, long dz, long dy, long dx, int ss);
void fy_dering_merge_tile(fy_dering *d, int slab, const fy_dering *t);
/* detection: hp_win ~15 vox, min_cnt ~100, min_amp ~0.5 u8, max_amp ~6 u8.
 * Returns # detected ring radii (all slabs); sets d->have_rings (>=2). */
long fy_dering_finalize(fy_dering *d, int hp_win, unsigned int min_cnt,
                        double min_amp, double max_amp);
/* subtract detected rings from a float tile at global offset; scale converts
 * u8 ring units to tile units (1/255 plain, 1/(hi-lo) after normalize). */
void fy_dering_apply(const fy_dering *d, float *f, long z0, long y0, long x0,
                     long dz, long dy, long dx, double scale);

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
    int    do_dering;                        /* detect+subtract residual rings */
    double calib_budget_gb;                  /* pass-1b sampling I/O budget (0 -> 200 GB).
                                              * Ring-detection sensitivity scales with it. */
    double dering_cy, dering_cx;             /* rotation axis; <=0 -> volume center */
    double psf_p5, psf_med;   /* measured PSF sigma map (drives the auto-deconv gate) */
    int    do_musica; double musica_p; int musica_levels; double musica_core;
    int    guided_subsample;  /* guided-filter coefficient subsample: 0 -> 2 (fast,
                               * recommended); 1 -> exact full-res path; else s */
    int    no_dither;         /* 0 -> dithered u8 export quantization (default,
                               * kills banding in flat papyrus); 1 -> round-to-nearest */
    int    halo;
    /* resolved calibration STATE (set by fy_calibrate; consumed by fy_process_chunk) */
    int    have_norm, have_zdrift, have_dec_range;
    int    have_dering; fy_dering *dering;   /* detected rings (caller frees, like zdrift_factor) */
    double psf_z_ratio;                      /* measured sigma_z/sigma_xy (0 or 1 = isotropic) */
    /* radially varying guided eps: flat-noise rises toward the rim on large scrolls
     * (measured 3x center->edge on PHercParis3). Linear fit fn(r) = a + b*r from the
     * pass-1 sample tiles; per-tile eps = guided_eps * clamp((fn(r)/fn_med)^2, 0.4, 8). */
    int    have_eps_r; double eps_fn_a, eps_fn_b, eps_fn_med;
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
/* the I/O-free half of fy_process_chunk: process one inner tile from an ALREADY-READ
 * halo'd u8 buffer (hz*hy*hx at global origin rz0/ry0/rx0; inner tile at z0/y0/x0,
 * span tz/ty/tx -- caller clamps both to the volume). vol_y/vol_x = full volume Y/X
 * (rotation-axis center for radial eps + dering). Thread-safe. */
int fy_process_buffer(const fy_pipeline_cfg *cfg, const unsigned char *u8buf,
                      long rz0, long ry0, long rx0, long hz, long hy, long hx,
                      long z0, long y0, long x0, long tz, long ty, long tx,
                      long vol_y, long vol_x,
                      unsigned char *out, int *all_air);

/* run the whole 2-pass pipeline zarr->zarr; cfg physics pre-filled, rest calibrated inside. */
int fy_run_pipeline(const char *in_root, const char *out_root, fy_pipeline_cfg *cfg,
                    int tile, int verbose);


#ifdef __cplusplus
}
#endif

#endif /* FYSICS_H */
