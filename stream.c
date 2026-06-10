/* stream.c -- streaming/chunked primitives for WHOLE-VOLUME processing at 20TB+.
 *
 * fysics does the per-chunk MATH; the caller (e.g. vc3d or a driver using fast zarr
 * I/O) owns chunk iteration and I/O. Operators split into two kinds:
 *
 *   LOCAL (deconv, denoise, mask): process one chunk (+halo) independently. Use the
 *     existing fy_deconvolve / fy_bilateral_denoise / fy_papyrus_mask per chunk.
 *
 *   GLOBAL (normalization, GLCAE global stage): need whole-volume statistics, so
 *     they're TWO-PASS:
 *       pass 1: stream every chunk -> fy_*_accumulate() into a small state struct
 *       finalize:                   -> compute the global mapping ONCE
 *       pass 2: stream every chunk -> fy_*_apply() using that mapping
 *     The state is tiny (a histogram), so 20TB never sits in RAM.
 *
 * Everything here is u8-friendly: helpers convert u8<->float, since the real
 * volumes are dense u8.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define L 256

/* ---------- u8 <-> float ---------- */
void fy_u8_to_float(const unsigned char *in, float *out, size_t n) {
    const float inv = 1.0f / 255.0f;
    for (size_t i = 0; i < n; i++) out[i] = in[i] * inv;
}
void fy_float_to_u8(const float *in, unsigned char *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = in[i] * 255.0f + 0.5f;
        out[i] = v < 0 ? 0 : (v > 255 ? 255 : (unsigned char)v);
    }
}

/* ---- recover physical attenuation from the per-volume u8 export window ----
 * Exact linear inverse of nabu's export windowing (see fysics.h). Per-voxel and
 * streaming-friendly: no global state, process any chunk independently. */
void fy_u8_to_phys(const unsigned char *in, float *out, size_t n,
                   double f32_min, double f32_max) {
    float lo = (float)f32_min;
    float scale = (float)((f32_max - f32_min) / 255.0);
    for (size_t i = 0; i < n; i++) out[i] = lo + in[i] * scale;
}

void fy_phys_to_u8(const float *in, unsigned char *out, size_t n,
                   double f32_min, double f32_max) {
    double span = f32_max - f32_min;
    float inv = (span != 0.0) ? (float)(255.0 / span) : 0.0f;
    float lo = (float)f32_min;
    for (size_t i = 0; i < n; i++) {
        float v = (in[i] - lo) * inv + 0.5f;
        out[i] = v < 0 ? 0 : (v > 255 ? 255 : (unsigned char)v);
    }
}

float fy_phys_to_u8_level(double phys, double f32_min, double f32_max) {
    double span = f32_max - f32_min;
    if (span == 0.0) return 0.0f;
    return (float)((phys - f32_min) / span * 255.0);
}

/* ================= GLOBAL: histogram-based ops (two-pass) ================= */
/* State accumulates a 256-bin histogram of u8 values over the whole volume.
 * Tiny + mergeable, so passes can even be parallelized (accumulate per thread,
 * then sum the histograms). */

void fy_hist_init(fy_hist_state *s) {
    memset(s->hist, 0, sizeof(s->hist));
    s->total = 0;
}

/* pass 1: call per chunk (u8) */
void fy_hist_accumulate_u8(fy_hist_state *s, const unsigned char *chunk, size_t n) {
    for (size_t i = 0; i < n; i++) s->hist[chunk[i]]++;
    s->total += n;
}

/* merge two states (for parallel accumulation) */
void fy_hist_merge(fy_hist_state *dst, const fy_hist_state *src) {
    for (int i = 0; i < L; i++) dst->hist[i] += src->hist[i];
    dst->total += src->total;
}

/* finalize: percentile value (0..255) from the global histogram */
int fy_hist_percentile_u8(const fy_hist_state *s, double pct) {
    long target = (long)(pct / 100.0 * (double)s->total);
    long acc = 0;
    for (int i = 0; i < L; i++) { acc += s->hist[i]; if (acc >= target) return i; }
    return L - 1;
}

/* ---- GLOBAL normalization (consistent across the whole volume) ---- */
/* finalize: compute lo/hi (u8) from global percentiles; pass 2 applies per chunk */
void fy_norm_finalize(const fy_hist_state *s, double lo_pct, double hi_pct,
                      unsigned char *lo_out, unsigned char *hi_out) {
    *lo_out = (unsigned char)fy_hist_percentile_u8(s, lo_pct);
    *hi_out = (unsigned char)fy_hist_percentile_u8(s, hi_pct);
}
/* pass 2: normalize a u8 chunk to [0,1] float using the GLOBAL lo/hi */
void fy_norm_apply_u8(const unsigned char *in, float *out, size_t n,
                      unsigned char lo, unsigned char hi) {
    float flo = lo, inv = (hi > lo) ? 1.0f / (float)(hi - lo) : 1.0f;
    for (size_t i = 0; i < n; i++) {
        float v = (in[i] - flo) * inv;
        out[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
}

/* ---- segmentation-quality metrics (math in C; Python is glue) ----
 * Computed from a 256-bin histogram so they're O(256), not O(N*thresholds).
 * Build the histogram with fy_hist_accumulate_u8 (or pass counts directly). */

/* Haralick-Shapiro bias-guarded Fisher discriminant J over all thresholds in [lo,hi):
 *   J(t) = (mu_hi - mu_lo)^2 / (var_lo + var_hi) * (min(n_lo,n_hi)/N / 0.5)
 * The balance term penalizes degenerate (mode-collapsed) splits. Returns best J; writes
 * the argmax threshold to *best_t. Intensities normalized to [0,1] (u8/255). */
double fy_valley_depth(const long hist[256], int *dark_out, int *light_out, int *valley_out) {
    double hf[256];
    for (int i = 0; i < 256; i++) {
        double s = 0; int cnt = 0;
        for (int k = -2; k <= 2; k++) { int j = i + k; if (j >= 0 && j < 256) { s += hist[j]; cnt++; } }
        hf[i] = s / cnt;
    }
    hf[0] = hf[254] = hf[255] = 0;
    double mx = 0; for (int i = 0; i < 256; i++) if (hf[i] > mx) mx = hf[i];
    if (mx <= 0) return -1;
    /* collect peaks > 5% of max, merge those within 10 bins (keep the taller) */
    int peaks[256], np = 0;
    for (int u = 3; u < 253; u++)
        if (hf[u] >= hf[u-1] && hf[u] >= hf[u+1] && hf[u] > 0.05 * mx) {
            if (np && u - peaks[np-1] <= 10) { if (hf[u] > hf[peaks[np-1]]) peaks[np-1] = u; }
            else peaks[np++] = u;
        }
    if (np < 2) return -1;
    /* two tallest peaks */
    int p1 = peaks[0], p2 = peaks[1];
    for (int i = 0; i < np; i++) {
        if (hf[peaks[i]] > hf[p1]) { p2 = p1; p1 = peaks[i]; }
        else if (hf[peaks[i]] > hf[p2] && peaks[i] != p1) p2 = peaks[i];
    }
    int a = p1 < p2 ? p1 : p2, b = p1 < p2 ? p2 : p1;
    int v = a; double vmin = hf[a];
    for (int u = a; u <= b; u++) if (hf[u] < vmin) { vmin = hf[u]; v = u; }
    if (dark_out) *dark_out = a; if (light_out) *light_out = b; if (valley_out) *valley_out = v;
    double mn = hf[a] < hf[b] ? hf[a] : hf[b];
    return 1.0 - vmin / (mn + 1e-12);
}


/* Flat-region noise: median local std over flat (low-gradient) blocks -- the noise floor.
 * blk=block size, samples nonoverlapping blocks, keeps flat ones (low mean-gradient), returns
 * the median of their stds. Lower = less noise. */
double fy_flat_noise(const float *v, int nz, int ny, int nx, int blk) {
    if (blk < 4) blk = 8;
    int nb = (nz/blk)*(ny/blk)*(nx/blk); if (nb <= 0) return 0;
    double *stds = (double*)malloc(sizeof(double)*nb); int m = 0;
    for (int z = 0; z+blk <= nz; z += blk)
        for (int y = 0; y+blk <= ny; y += blk)
            for (int x = 0; x+blk <= nx; x += blk) {
                double s = 0, s2 = 0; int c = 0; double mn = 1e9, mx = -1e9;
                for (int dz = 0; dz < blk; dz++)
                    for (int dy = 0; dy < blk; dy++)
                        for (int dx = 0; dx < blk; dx++) {
                            float val = v[(((size_t)(z+dz)*ny+(y+dy))*nx)+(x+dx)];
                            s += val; s2 += val*val; c++;
                            if (val < mn) mn = val; if (val > mx) mx = val;
                        }
                double mean = s/c, var = s2/c - mean*mean;
                (void)mx; (void)mn;
                /* keep all bright (papyrus, not air) blocks; the noise floor is the LOW
                 * percentile of their stds -- the least-textured papyrus blocks are ~pure
                 * noise. (Requiring genuinely "flat" blocks fails: papyrus is textured.) */
                if (mean > 0.1) stds[m++] = sqrt(var > 0 ? var : 0);
            }
    if (m == 0) { free(stds); return 0; }
    /* sort; return the 10th percentile (the noise floor among bright blocks) */
    for (int i = 0; i < m; i++) for (int j = i+1; j < m; j++) if (stds[j] < stds[i]) { double t = stds[i]; stds[i] = stds[j]; stds[j] = t; }
    double nf = stds[m/10]; free(stds); return nf;
}
