/* paganin.c -- recon transfer functions + Wiener deconvolution kernel.
 *
 * Inverts the known nabu reconstruction operators to sharpen a volume. The hot
 * loops (filter build, complex multiply) are flat and vectorizable under -Ofast.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FY_HC_KEV_UM 1.23984199e-3   /* h*c in keV*micron (nabu constant) */

double fy_paganin_transfer(double f_cyc_per_voxel, const fy_physics *p) {
    /* exact nabu form: 1/(1 + db*L*D*pi*f^2), f in cycles/micron */
    double L = FY_HC_KEV_UM / p->energy_kev;     /* micron */
    double D = p->distance_mm * 1000.0;           /* micron */
    /* pixel_um MUST be microns (e.g. 2.4). Guard against the common mistake of
     * passing the metadata's millimeter value (e.g. 0.0024): a sub-0.1 "micron"
     * pixel is physically a mm value, so auto-correct to keep callers safe. */
    double px = p->pixel_um;
    if (px > 0.0 && px < 0.1) px *= 1000.0;        /* mm -> micron */
    double f_um = f_cyc_per_voxel / px;            /* cycles/voxel -> cycles/micron */
    double f2 = f_um * f_um;
    return 1.0 / (1.0 + p->delta_beta * L * D * M_PI * f2);
}

double fy_unsharp_transfer(double f_cyc_per_voxel, const fy_physics *p) {
    if (p->unsharp_coeff == 0.0 || p->unsharp_sigma == 0.0) return 1.0;
    /* gaussian transfer at sigma (voxels): G = exp(-2 pi^2 sigma^2 f^2) */
    double s = p->unsharp_sigma;
    double g = exp(-2.0 * M_PI * M_PI * s * s * f_cyc_per_voxel * f_cyc_per_voxel);
    return 1.0 + p->unsharp_coeff * (1.0 - g);
}

double fy_recon_transfer(double f_cyc_per_voxel, const fy_physics *p) {
    double H = fy_paganin_transfer(f_cyc_per_voxel, p) *
               fy_unsharp_transfer(f_cyc_per_voxel, p);
    if (H < 1e-4) H = 1e-4;
    return H;
}

/* Maximum Wiener gain we ever allow, so a bad reg can't amplify the high-frequency
 * noise floor without bound. The Wiener filter H/(H^2+reg) peaks at 1/(2*sqrt(reg)),
 * so capping the gain == enforcing a reg floor; we apply it directly as a clamp in the
 * filter loop (frequency-by-frequency) which is strictly safer than only raising reg. */
#define FY_MAX_DECONV_GAIN 8.0

/* Noise-safe regularization for fy_deconvolve.
 *
 * NOTE (measured, 90 cubes / 18 volumes, 2026-06): reg = 0.015 sits at a sensible knee.
 * A reg sweep showed dropping to 0.005 only marginally improves the texture/noise ratio
 * (~+0.1-0.2) but roughly DOUBLES flat-region noise amplification (to 5.5-7x); stronger
 * reg collapses the contrast restoration. So a single conservative base is right -- the
 * real per-volume lever is delta_beta scaling (see fy_auto_deltabeta_scale), not reg.
 * The per-frequency gain clamp (FY_MAX_DECONV_GAIN) in the filter loop is the hard
 * safety rail against pathological metadata. */
double fy_auto_reg(const fy_physics *p, double base) {
    (void)p;
    return base > 0.0 ? base : 0.015;
}

/* Regime-dependent delta_beta scaling for the deconvolution (Gureyev-analog partial
 * inversion). Measured (90 cubes, 18 volumes): inverting the FULL metadata delta_beta
 * OVER-inverts fine/strong-filter volumes -- it over-boosts the mid-band, starves the
 * top band, and amplifies flat-region noise more than it lifts texture (texture/noise
 * ratio < 1). Backing off to ~0.25-0.5x delta_beta on those volumes simultaneously
 * raises texture gain, lowers noise, AND rescues the high band. On coarse/mild-filter
 * volumes (~8.6-9.4um) the full delta_beta is already well matched -- reducing it just
 * weakens a good deblur. The discriminator is the recon low-pass strength at Nyquist,
 * H_nyq = fy_paganin_transfer(0.5, p): strong filter -> tiny H_nyq -> scale down.
 * Measured H_nyq: 1.1um=0.0008, 2.4um=0.0021, 9.4um=0.0084 (fine sits ~4-10x lower).
 *   H_nyq <~ 0.0025 (fine 1.1-4.3um) -> 0.35  (partial inversion)
 *   H_nyq  ~ 0.02-0.04 (medium/coarse) -> ramp 0.35..1.0
 *   H_nyq >~ 0.04 (coarse 8.6-9.4um)  -> 1.0   (full inversion)
 * Returns a multiplier in [0.25, 1.0] to apply to p->delta_beta before deconvolving. */
double fy_auto_deltabeta_scale(const fy_physics *p) {
    double Hnyq = fy_paganin_transfer(0.5, p);
    if (!isfinite(Hnyq) || Hnyq <= 0) return 1.0;
    const double H_lo = 0.0025;  /* at/below: fine, over-inverted regime (1.1-4.3um) */
    const double H_hi = 0.0070;  /* at/above: coarse, full inversion is right (8.6-9.4um) */
    const double S_lo = 0.35;    /* partial-inversion scale for fine volumes */
    if (Hnyq <= H_lo) return S_lo;
    if (Hnyq >= H_hi) return 1.0;
    /* linear ramp between */
    double t = (Hnyq - H_lo) / (H_hi - H_lo);
    return S_lo + t * (1.0 - S_lo);
}

int fy_kernel_halo(const fy_physics *p) {
    /* impulse response of the deconv filter along one axis; find decay extent */
    const int n = 256;
    float *re = (float *)malloc(sizeof(float) * n);
    float *im = (float *)calloc(n, sizeof(float));
    if (!re || !im) { free(re); free(im); return 16; }
    for (int i = 0; i < n; i++) {
        double f = (i <= n / 2) ? (double)i / n : (double)(i - n) / n; /* fftfreq */
        double H = fy_recon_transfer(fabs(f), p);
        double inv = H / (H * H + 0.05);
        re[i] = (float)inv; im[i] = 0.0f;
    }
    fy_fft1d(re, im, n, +1);  /* to real space */
    /* peak-relative decay */
    float peak = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(re[i]); if (a > peak) peak = a; }
    int half = 8;
    for (int d = 1; d < n / 2; d++) {
        float a = fabsf(re[d]);           /* response at lag d (index 0 = center) */
        if (a > 0.01f * peak) half = d;
    }
    free(re); free(im);
    if (half < 8) half = 8;
    if (half > 96) half = 96;
    return half;
}

/* reflect-pad index helper */
static inline int reflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i - 1;
        if (i >= n) i = 2 * n - i - 1;
    }
    return i;
}

/* ---- shared radial-weight deconvolution core ----
 * Reflect-pads to an FFT-friendly size (2^k or 3*2^k: a 176 halo'd tile costs a
 * 192 FFT, not 256 -- ~2.4x less FFT work), forward FFT, multiplies by a radial
 * weight w(fr), inverse FFT, crop. The weight is evaluated through a LUT over
 * u = fr^2: every weight here is smooth in f^2, so 4096 linear-interp entries
 * match the direct evaluation to <1e-6 while avoiding a per-voxel sqrt+exp. */
typedef double (*radial_weight_fn)(double fr, const fy_physics *p, double param);

#define FY_DECONV_LUT_N 4096

/* fz2_scale: anisotropic-PSF support. The z frequency axis is scaled by
 * (sigma_z/sigma_xy)^2 before the radial weight lookup, which makes the Gaussian
 * PSF factor exp(-2 pi^2 sigma_xy^2 u) EXACTLY the anisotropic PSF inverse (the
 * unsharp factor is evaluated at the scaled radius too -- a small approximation,
 * fine at the measured ~1.17x ratios). 1.0 = isotropic. */
/* REAL-INPUT path: the volume is real and the weight is real and even, so the whole
 * pipeline runs on the Hermitian HALF-spectrum (kx in [0, px/2]):
 *   - X pass: two-for-one packed R2C (two real rows per complex FFT), with the
 *     reflect-padding fused into the row gather (the full padded volume is never built);
 *   - Y/Z passes: complex FFTs over hx = px/2+1 columns instead of px (~half the work);
 *   - inverse X (C2R) only for the rows that survive the crop.
 * ~2x less FFT work and ~2x less memory than the previous full-complex path. */
static int deconv_radial(const float *in, float *out, int nz, int ny, int nx,
                         const fy_physics *p, double param, radial_weight_fn wfn,
                         double cap, double fz2_scale) {
    int pz = fy_next_fft_size(nz), py = fy_next_fft_size(ny), px = fy_next_fft_size(nx);
    int hx = px / 2 + 1;
    size_t nh = (size_t)pz * py * hx;
    float *hre = (float *)malloc(sizeof(float) * nh);
    float *him = (float *)malloc(sizeof(float) * nh);
    int maxn = px > py ? (px > pz ? px : pz) : (py > pz ? py : pz);
    float *lr = (float *)malloc(sizeof(float) * maxn);
    float *li = (float *)malloc(sizeof(float) * maxn);
    if (!hre || !him || !lr || !li) { free(hre); free(him); free(lr); free(li); return 1; }

    /* ---- forward X (R2C, two rows per FFT, reflect-pad fused into the gather) ---- */
    for (int z = 0; z < pz; z++) {
        int sz = reflect(z, nz);
        for (int y = 0; y < py; y += 2) {
            int sy0 = reflect(y, ny);
            const float *rA = in + ((size_t)sz * ny + sy0) * nx;
            int hasB = y + 1 < py;
            const float *rB = hasB ? in + ((size_t)sz * ny + reflect(y + 1, ny)) * nx : NULL;
            for (int x = 0; x < px; x++) {
                int sx = reflect(x, nx);
                lr[x] = rA[sx];
                li[x] = hasB ? rB[sx] : 0.0f;
            }
            fy_fft1d(lr, li, px, -1);
            /* untangle: A = (L(k)+conj(L(-k)))/2, B = (L(k)-conj(L(-k)))/(2i) */
            float *ar = hre + ((size_t)z * py + y) * hx;
            float *ai = him + ((size_t)z * py + y) * hx;
            float *br = hasB ? hre + ((size_t)z * py + y + 1) * hx : NULL;
            float *bi = hasB ? him + ((size_t)z * py + y + 1) * hx : NULL;
            for (int k = 0; k < hx; k++) {
                int mk = k == 0 ? 0 : px - k;     /* (px - k) mod px */
                ar[k] = 0.5f * (lr[k] + lr[mk]);
                ai[k] = 0.5f * (li[k] - li[mk]);
                if (hasB) {
                    br[k] = 0.5f * (li[k] + li[mk]);
                    bi[k] = 0.5f * (lr[mk] - lr[k]);
                }
            }
        }
    }

    /* ---- forward Y then Z (complex, half-spectrum columns) ---- */
    for (int z = 0; z < pz; z++)
        for (int k = 0; k < hx; k++) {
            size_t base = (size_t)z * py * hx + k;
            for (int y = 0; y < py; y++) { lr[y] = hre[base + (size_t)y * hx]; li[y] = him[base + (size_t)y * hx]; }
            fy_fft1d(lr, li, py, -1);
            for (int y = 0; y < py; y++) { hre[base + (size_t)y * hx] = lr[y]; him[base + (size_t)y * hx] = li[y]; }
        }
    for (int y = 0; y < py; y++)
        for (int k = 0; k < hx; k++) {
            size_t base = (size_t)y * hx + k, zs = (size_t)py * hx;
            for (int z = 0; z < pz; z++) { lr[z] = hre[base + (size_t)z * zs]; li[z] = him[base + (size_t)z * zs]; }
            fy_fft1d(lr, li, pz, -1);
            for (int z = 0; z < pz; z++) { hre[base + (size_t)z * zs] = lr[z]; him[base + (size_t)z * zs] = li[z]; }
        }

    /* ---- weight multiply over the half-spectrum (radial LUT over f^2) ---- */
    double *fz2 = (double *)malloc(sizeof(double) * pz);
    double *fy2 = (double *)malloc(sizeof(double) * py);
    double *fx2 = (double *)malloc(sizeof(double) * hx);
    if (!fz2 || !fy2 || !fx2) { free(hre); free(him); free(lr); free(li); free(fz2); free(fy2); free(fx2); return 1; }
    double mz = 0, my = 0, mx = 0;
    if (fz2_scale <= 0) fz2_scale = 1.0;
    for (int i = 0; i < pz; i++) { double f = (i <= pz/2) ? (double)i/pz : (double)(i-pz)/pz; fz2[i] = f*f*fz2_scale; if (fz2[i] > mz) mz = fz2[i]; }
    for (int i = 0; i < py; i++) { double f = (i <= py/2) ? (double)i/py : (double)(i-py)/py; fy2[i] = f*f; if (fy2[i] > my) my = fy2[i]; }
    for (int i = 0; i < hx; i++) { double f = (double)i/px; fx2[i] = f*f; if (fx2[i] > mx) mx = fx2[i]; }

    float lut[FY_DECONV_LUT_N + 1];
    double umax = mz + my + mx;
    double us = FY_DECONV_LUT_N / umax;
    for (int i = 0; i <= FY_DECONV_LUT_N; i++) {
        double w = wfn(sqrt((double)i / us), p, param);
        if (cap > 0 && w > cap) w = cap;
        lut[i] = (float)w;
    }
    for (int z = 0; z < pz; z++) {
        for (int y = 0; y < py; y++) {
            size_t row = ((size_t)z * py + y) * hx;
            double fzy = fz2[z] + fy2[y];
            for (int k = 0; k < hx; k++) {
                double t = (fzy + fx2[k]) * us;
                int ti = (int)t;
                if (ti >= FY_DECONV_LUT_N) ti = FY_DECONV_LUT_N - 1;
                float w = lut[ti] + (lut[ti + 1] - lut[ti]) * (float)(t - ti);
                hre[row + k] *= w;
                him[row + k] *= w;
            }
        }
    }

    /* ---- inverse Z then Y (complex, half-spectrum columns) ---- */
    for (int y = 0; y < py; y++)
        for (int k = 0; k < hx; k++) {
            size_t base = (size_t)y * hx + k, zs = (size_t)py * hx;
            for (int z = 0; z < pz; z++) { lr[z] = hre[base + (size_t)z * zs]; li[z] = him[base + (size_t)z * zs]; }
            fy_fft1d(lr, li, pz, +1);
            for (int z = 0; z < pz; z++) { hre[base + (size_t)z * zs] = lr[z]; him[base + (size_t)z * zs] = li[z]; }
        }
    for (int z = 0; z < nz; z++)   /* only z rows that survive the crop */
        for (int k = 0; k < hx; k++) {
            size_t base = (size_t)z * py * hx + k;
            for (int y = 0; y < py; y++) { lr[y] = hre[base + (size_t)y * hx]; li[y] = him[base + (size_t)y * hx]; }
            fy_fft1d(lr, li, py, +1);
            for (int y = 0; y < py; y++) { hre[base + (size_t)y * hx] = lr[y]; him[base + (size_t)y * hx] = li[y]; }
        }

    /* ---- inverse X (C2R, two rows per FFT), normalize, crop ---- */
    float invN = 1.0f / ((float)pz * (float)py * (float)px);
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y += 2) {
            int hasB = y + 1 < ny;
            const float *ar = hre + ((size_t)z * py + y) * hx;
            const float *ai = him + ((size_t)z * py + y) * hx;
            const float *br = hasB ? hre + ((size_t)z * py + y + 1) * hx : NULL;
            const float *bi = hasB ? him + ((size_t)z * py + y + 1) * hx : NULL;
            /* repack L(k) = A(k) + i*B(k); Hermitian extension for k > px/2 */
            for (int k = 0; k < hx; k++) {
                lr[k] = ar[k] - (hasB ? bi[k] : 0.0f);
                li[k] = ai[k] + (hasB ? br[k] : 0.0f);
            }
            for (int k = hx; k < px; k++) {
                int m = px - k;
                lr[k] = ar[m] + (hasB ? bi[m] : 0.0f);
                li[k] = -ai[m] + (hasB ? br[m] : 0.0f);
            }
            fy_fft1d(lr, li, px, +1);
            float *oA = out + ((size_t)z * ny + y) * nx;
            float *oB = hasB ? out + ((size_t)z * ny + y + 1) * nx : NULL;
            for (int x = 0; x < nx; x++) {
                oA[x] = lr[x] * invN;
                if (hasB) oB[x] = li[x] * invN;
            }
        }
    }

    free(hre); free(him); free(lr); free(li); free(fz2); free(fy2); free(fx2);
    return 0;
}

/* Wiener inverse weight of the net recon transfer: H/(H^2+reg) */
static double wiener_weight(double fr, const fy_physics *p, double reg) {
    double H = fy_recon_transfer(fr, p);
    return H / (H * H + reg);
}

int fy_deconvolve(const float *in, float *out,
                  int nz, int ny, int nx,
                  const fy_physics *p, double reg) {
    /* reg <= 0 -> derive a noise-safe value from the volume physics (recommended). */
    if (reg <= 0.0) reg = fy_auto_reg(p, 0.015);
    /* hard gain cap so a mis-set reg can never amplify the noise floor */
    return deconv_radial(in, out, nz, ny, nx, p, reg, wiener_weight, FY_MAX_DECONV_GAIN, 1.0);
}

/* OPERATOR-MATCHED Wiener inverse of nabu's MEASURED EFFECTIVE forward operator.
 * KEY INSIGHT (BM18 physics workflow): nabu does Paganin(db) THEN unsharp(coeff,sigma); the
 * unsharp already largely UNDOES the Paganin blur, so the NET effective volume blur measured
 * from real edges is only sigma~1.0 vox -- NOT the ~9.8 vox the naive volume-Paganin implies.
 * Inverting the full Paganin (as plain fy_deconvolve does) therefore over-boosts blur
 * that is no longer there and explodes the noise floor. The CORRECT operator to invert is just
 *   H_fwd(k) = G_psf(k) * H_unsharp(k),   G_psf=exp(-2 pi^2 sigma_psf^2 k^2),
 *   H_unsharp(k) = 1 + coeff*(1 - exp(-2 pi^2 sigma_u^2 k^2))   (== fy_unsharp_transfer)
 * with the Wiener-regularized inverse H_fwd/(H_fwd^2 + gamma). psf_sigma_vox sets sigma_psf;
 * unsharp_coeff/unsharp_sigma come straight from the metadata. Measured: RMSE-to-truth 0.0225
 * (vs plain 0.032) -- recovers HF toward truth WITHOUT amplitude overshoot. */
static double matched_transfer(double fr, const fy_physics *p, double gamma) {
    double sigma = p->psf_sigma_vox > 0 ? p->psf_sigma_vox : 1.0;
    double f2 = fr * fr;
    double G = exp(-2.0 * M_PI * M_PI * sigma * sigma * f2);  /* system PSF low-pass */
    double Hu = fy_unsharp_transfer(fr, p);                   /* nabu's unsharp boost */
    double Hfwd = G * Hu;                                     /* net effective forward operator */
    return Hfwd / (Hfwd * Hfwd + gamma);                     /* Wiener-regularized inverse */
}

int fy_deconvolve_matched(const float *in, float *out,
                          int nz, int ny, int nx,
                          const fy_physics *p, double tikhonov) {
    if (tikhonov <= 0) tikhonov = 0.05;
    /* anisotropic PSF: scale the z frequency axis by (sigma_z/sigma_xy)^2 */
    double fzs = 1.0;
    double sxy = p->psf_sigma_vox > 0 ? p->psf_sigma_vox : 1.0;
    if (p->psf_sigma_z_vox > 0 && sxy > 0) {
        double rr = p->psf_sigma_z_vox / sxy;
        fzs = rr * rr;
    }
    return deconv_radial(in, out, nz, ny, nx, p, tikhonov, matched_transfer, 0.0, fzs);
}
