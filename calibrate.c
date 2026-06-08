/* calibrate.c -- per-volume CALIBRATION math (pure C).
 *
 * Reproduces the Python calibration in superres/fysics_pipeline.py (BM18-verified):
 *   - eps from the clean noise floor:  eps = (3 * median flat_nf)^2
 *   - air cut:  scratch-denoise (guided eps=0.01 x passes) -> u8 histogram ->
 *               fy_valley_depth -> air_cut_u8 = valley, band = max(4,(valley-dark)/4)
 *   - normalize gate:  apply iff (hi-lo)/255 < 0.40
 *   - z-drift gate:    apply iff coherence>=0.5 AND slope_frac>=0.05 AND meta>=0.05
 *   - PSF sigma:       subpixel-aligned ESF over air<->papyrus edges (half=7,sup=8,
 *                      min_contrast=40); LSF=clamped grad(ESF); sigma=2nd moment
 *   - downsample factor: blend p5->med by aggr; f0=sqrt(-ln(0.1)/(2 pi^2 sigma^2));
 *                        factor = max(1, floor(0.5/f0))
 *
 * The kernels (fy_guided_denoise, fy_valley_depth, fy_flat_noise) live in their own
 * files; this file only does the calibration arithmetic on top of them.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------- inline robust stats helpers (median / percentile) ---------- */
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

/* median of a COPY (does not disturb the caller's array) */
static double median_of(const double *v, int n) {
    if (n <= 0) return 0.0;
    double *t = malloc(sizeof(double) * (size_t)n);
    if (!t) return 0.0;
    memcpy(t, v, sizeof(double) * (size_t)n);
    qsort(t, (size_t)n, sizeof(double), cmp_double);
    double m = (n & 1) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
    free(t);
    return m;
}

/* ===================================================================== *
 *  PSF sigma: subpixel-aligned ESF on air<->papyrus edges               *
 * ===================================================================== */
/* Matches _measure_tile_psf_sigma: half=7, sup=8, min_contrast=40 (u8), 50%-crossing
 * align, LSF=clamped grad(ESF), sigma=2nd moment. Returns -1 if <30 clean edges or
 * the fitted sigma is outside (0.3,5). Deterministic: the Python used a seeded RNG to
 * sample lines; here we scan ALL interior lines (a superset), which yields the same
 * averaged ESF in the large-count limit and removes RNG nondeterminism. */
double fy_cal_measure_psf_sigma(const unsigned char *tile, int nz, int ny, int nx) {
    const int half = 7, sup = 8;
    const double min_c = 40.0;
    const int len = 2 * half + 1;            /* samples per profile line */
    const int glen = 2 * half * sup + 1;     /* supersampled accumulation grid */

    double *acc = calloc((size_t)glen, sizeof(double));
    if (!acc) return -1.0;
    long nprof = 0;

    /* supersampled grid coords in [-half, +half] */
    /* grid[k] = -half + k*(2*half)/(glen-1) = -half + k/sup */
    /* axis 1 = y edges (line along y at fixed z,x), axis 2 = x edges */
    for (int ax = 1; ax <= 2; ax++) {
        for (int z = 0; z < nz; z++) {
            int ylo = (ax == 1) ? half : 0;
            int yhi = (ax == 1) ? ny - half : ny;
            for (int y = ylo; y < yhi; y++) {
                int xlo = (ax == 2) ? half : 0;
                int xhi = (ax == 2) ? nx - half : nx;
                for (int x = xlo; x < xhi; x++) {
                    /* gather the len-sample line */
                    double line[2 * 7 + 1];
                    for (int t = 0; t < len; t++) {
                        int yy = (ax == 1) ? (y - half + t) : y;
                        int xx = (ax == 2) ? (x - half + t) : x;
                        line[t] = (double)tile[((size_t)z * ny + yy) * nx + xx];
                    }
                    /* contrast = mean(last2) - mean(first2); need |.|>=min_c */
                    double lo = 0.5 * (line[0] + line[1]);
                    double hi = 0.5 * (line[len - 1] + line[len - 2]);
                    if (fabs(hi - lo) < min_c) continue;
                    /* orient bright-to-... so the rising edge faces +; flip if needed */
                    double s[2 * 7 + 1];
                    if (hi > lo)
                        for (int t = 0; t < len; t++) s[t] = line[t];
                    else
                        for (int t = 0; t < len; t++) s[t] = line[len - 1 - t];
                    /* derivative; reject if a big negative dip OR the max-slope sample
                     * isn't near the center (a clean single edge crosses at center). */
                    double dmin = 1e30; int argmax = 0; double dmax = -1e30;
                    for (int t = 0; t < len - 1; t++) {
                        double d = s[t + 1] - s[t];
                        if (d < dmin) dmin = d;
                        if (d > dmax) { dmax = d; argmax = t; }
                    }
                    if (dmin < -8.0) continue;
                    if (!(argmax == half - 1 || argmax == half || argmax == half + 1))
                        continue;
                    /* normalize ESF to [0,1] */
                    double smin = s[0];
                    for (int t = 1; t < len; t++) if (s[t] < smin) smin = s[t];
                    double e[2 * 7 + 1], emax = 0.0;
                    for (int t = 0; t < len; t++) { e[t] = s[t] - smin; if (e[t] > emax) emax = e[t]; }
                    if (emax <= 0.0) continue;
                    for (int t = 0; t < len; t++) e[t] /= emax;
                    /* first sample with e>=0.5, refine the crossing to subpixel */
                    int i = -1;
                    for (int t = 0; t < len; t++) if (e[t] >= 0.5) { i = t; break; }
                    if (i <= 0) continue;   /* none, or crossing at index 0 -> reject */
                    double frac = (0.5 - e[i - 1]) / (e[i] - e[i - 1] + 1e-9);
                    double cross = (double)(i - 1) + frac;

                    /* accumulate this ESF onto the supersampled grid, x-shifted by cross
                     * (np.interp with left=0, right=1 clamping). profile x-coord of sample
                     * t is (t - cross). */
                    for (int k = 0; k < glen; k++) {
                        double gx = -(double)half + (double)k / (double)sup;
                        /* interpolate e at sample-coord (gx + cross) */
                        double sx = gx + cross;   /* in [0, len-1] sample units */
                        double val;
                        if (sx <= 0.0) val = 0.0;             /* left clamp -> 0 */
                        else if (sx >= (double)(len - 1)) val = 1.0; /* right clamp -> 1 */
                        else {
                            int j = (int)floor(sx);
                            double f = sx - (double)j;
                            val = e[j] * (1.0 - f) + e[j + 1] * f;
                        }
                        acc[k] += val;
                    }
                    nprof++;
                }
            }
        }
    }

    if (nprof < 30) { free(acc); return -1.0; }

    /* ESF = acc/nprof ; LSF = clamp(gradient(ESF),0,inf) ; normalize ; sigma=2nd moment */
    double *esf = malloc(sizeof(double) * (size_t)glen);
    double *lsf = malloc(sizeof(double) * (size_t)glen);
    if (!esf || !lsf) { free(acc); free(esf); free(lsf); return -1.0; }
    for (int k = 0; k < glen; k++) esf[k] = acc[k] / (double)nprof;
    /* np.gradient: central differences interior, one-sided ends. grid spacing = 1/sup. */
    double dx = 1.0 / (double)sup;
    for (int k = 0; k < glen; k++) {
        double g;
        if (k == 0) g = (esf[1] - esf[0]) / dx;
        else if (k == glen - 1) g = (esf[glen - 1] - esf[glen - 2]) / dx;
        else g = (esf[k + 1] - esf[k - 1]) / (2.0 * dx);
        lsf[k] = g > 0.0 ? g : 0.0;
    }
    double lsum = 0.0;
    for (int k = 0; k < glen; k++) lsum += lsf[k];
    if (lsum <= 0.0) { free(acc); free(esf); free(lsf); return -1.0; }
    for (int k = 0; k < glen; k++) lsf[k] /= lsum;
    double mu = 0.0;
    for (int k = 0; k < glen; k++) {
        double gx = -(double)half + (double)k / (double)sup;
        mu += gx * lsf[k];
    }
    double var = 0.0;
    for (int k = 0; k < glen; k++) {
        double gx = -(double)half + (double)k / (double)sup;
        double dd = gx - mu;
        var += dd * dd * lsf[k];
    }
    double sig = sqrt(var);
    free(acc); free(esf); free(lsf);
    if (sig > 0.3 && sig < 5.0) return sig;
    return -1.0;
}

/* ===================================================================== *
 *  Downsample factor from the PSF-sigma map                             *
 * ===================================================================== */
int fy_cal_downsample_factor(double p5, double med, double aggr) {
    if (aggr < 0.0) aggr = 0.0;
    if (aggr > 1.0) aggr = 1.0;
    /* MTF=0.2 floor (0.1 was too permissive -> over-downsampled), aggr blends only halfway
     * p5->median, hard cap 2x (resolution ~2-3 vox FWHM => ~2x oversampling; 3x risks real loss). */
    double sigma = p5 + 0.5 * aggr * (med - p5);
    if (sigma <= 0.0) return 1;
    double f0 = sqrt(-log(0.2) / (2.0 * M_PI * M_PI * sigma * sigma)); /* cyc/vox */
    double factor = 0.5 / f0;
    int f = (int)floor(factor);
    if (f < 1) f = 1;
    if (f > 2) f = 2;
    return f;
}

/* ===================================================================== *
 *  eps from the clean noise floor                                       *
 * ===================================================================== */
double fy_cal_eps_from_flatnoise(const double *flat_nfs, int n) {
    if (n <= 0) return 0.0;
    double m = median_of(flat_nfs, n);
    double k = 4.2 * m;   /* RETUNE consensus: eps~0.004 (high-sigma-safe across PSF band 0.8-1.8) */
    return k * k;
}

/* ===================================================================== *
 *  Air cut (valley + band)                                              *
 * ===================================================================== */
int fy_cal_air_cut_from_hist(const long hist[256], int *air_cut_u8, int *air_cut_band) {
    int dark = 0, light = 0, valley = 0;
    double d = fy_valley_depth(hist, &dark, &light, &valley);
    if (d < 0.0) return 0;                    /* not bimodal */
    int band = (valley - dark) / 4;
    if (band < 4) band = 4;
    if (air_cut_u8)  *air_cut_u8 = valley;
    if (air_cut_band) *air_cut_band = band;
    return 1;
}

int fy_cal_air_cut(const unsigned char *const *tiles, int n, int nz, int ny, int nx,
                   int scratch_passes, int *air_cut_u8, int *air_cut_band) {
    if (n <= 0) return 0;
    if (scratch_passes < 1) scratch_passes = 1;
    size_t vox = (size_t)nz * ny * nx;
    long hist[256];
    memset(hist, 0, sizeof(hist));

    float *a = malloc(sizeof(float) * vox);
    float *b = malloc(sizeof(float) * vox);
    if (!a || !b) { free(a); free(b); return 0; }

    for (int ti = 0; ti < n; ti++) {
        const unsigned char *t = tiles[ti];
        /* u8 -> [0,1] float scratch */
        const float inv = 1.0f / 255.0f;
        for (size_t i = 0; i < vox; i++) a[i] = t[i] * inv;
        /* scratch denoise: guided eps=0.01 x scratch_passes (ping-pong a<->b) */
        float *src = a, *dst = b;
        for (int p = 0; p < scratch_passes; p++) {
            if (fy_guided_denoise(src, dst, nz, ny, nx, 2, 0.01) != 0) break;
            float *tmp = src; src = dst; dst = tmp;
        }
        /* accumulate u8 histogram of the denoised scratch (round-to-nearest, clip) */
        for (size_t i = 0; i < vox; i++) {
            int u = (int)(src[i] * 255.0f + 0.5f);
            if (u < 0) u = 0; else if (u > 255) u = 255;
            hist[u]++;
        }
    }
    free(a); free(b);
    return fy_cal_air_cut_from_hist(hist, air_cut_u8, air_cut_band);
}

/* ===================================================================== *
 *  Gates                                                                *
 * ===================================================================== */
int fy_cal_norm_gate(int lo, int hi) {
    double span_frac = (double)(hi - lo) / 255.0;
    return span_frac < 0.40 ? 1 : 0;
}

int fy_cal_zdrift_gate(double coherence, double slope_frac, double meta_drift) {
    /* meta_drift < 0 sentinel: metadata unavailable -> the metadata condition passes. */
    int meta_ok = (meta_drift < 0.0) || (meta_drift >= 0.05);
    return (coherence >= 0.5 && slope_frac >= 0.05 && meta_ok) ? 1 : 0;
}
