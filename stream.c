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

/* ---- GLOBAL GLCAE mapping (the global stage, computed once) ----
 * The global GLCAE stage is exactly a histogram -> lambda-blend -> CDF mapping.
 * We compute it from the WHOLE-VOLUME histogram in finalize, then apply the same
 * 256-entry lookup to every chunk in pass 2. (Defined in glcae.c via the shared
 * global_mapping logic exposed below.) */
extern void fy_global_glcae_mapping_from_hist(const long *hist, long total, int *t_out);

void fy_glcae_global_finalize(const fy_hist_state *s, int *mapping_out) {
    fy_global_glcae_mapping_from_hist(s->hist, s->total, mapping_out);
}
/* pass 2: apply the global GLCAE lookup to a u8 chunk -> float [0,1] */
void fy_glcae_global_apply_u8(const unsigned char *in, float *out, size_t n,
                              const int *mapping) {
    float inv = 1.0f / (float)(L - 1);
    for (size_t i = 0; i < n; i++) out[i] = mapping[in[i]] * inv;
}

/* ---- auto air/papyrus threshold (Otsu) from the global histogram ----
 * Replaces a hardcoded threshold: finds the intensity that best separates the two
 * modes (air vs papyrus) by maximizing between-class variance (Otsu's method) over
 * the volume's actual histogram. Returns the threshold as a [0,1] fraction (u8/255),
 * matching the air_thresh convention. */
float fy_auto_air_thresh(const fy_hist_state *s) {
    long total = s->total;
    if (total <= 0) return 0.15f;
    /* total intensity */
    double sumAll = 0;
    for (int i = 0; i < 256; i++) sumAll += (double)i * s->hist[i];
    double wB = 0, sumB = 0, maxVar = -1;
    int best = 38; /* ~0.15*255 fallback */
    for (int t = 0; t < 256; t++) {
        wB += s->hist[t];
        if (wB == 0) continue;
        double wF = total - wB;
        if (wF == 0) break;
        sumB += (double)t * s->hist[t];
        double mB = sumB / wB;
        double mF = (sumAll - sumB) / wF;
        double var = wB * wF * (mB - mF) * (mB - mF);
        if (var > maxVar) { maxVar = var; best = t; }
    }
    return (float)best / 255.0f;
}


/* ---- segmentation-quality metrics (math in C; Python is glue) ----
 * Computed from a 256-bin histogram so they're O(256), not O(N*thresholds).
 * Build the histogram with fy_hist_accumulate_u8 (or pass counts directly). */

/* Haralick-Shapiro bias-guarded Fisher discriminant J over all thresholds in [lo,hi):
 *   J(t) = (mu_hi - mu_lo)^2 / (var_lo + var_hi) * (min(n_lo,n_hi)/N / 0.5)
 * The balance term penalizes degenerate (mode-collapsed) splits. Returns best J; writes
 * the argmax threshold to *best_t. Intensities normalized to [0,1] (u8/255). */
double fy_haralick_shapiro(const long hist[256], int lo, int hi, int min_count, int *best_t) {
    double n = 0, cx[256], cx2[256], c[256];
    double acc = 0, accx = 0, accx2 = 0;
    for (int i = 0; i < 256; i++) {
        double xi = (double)i / 255.0;
        acc += hist[i]; accx += hist[i] * xi; accx2 += hist[i] * xi * xi;
        c[i] = acc; cx[i] = accx; cx2[i] = accx2;
    }
    n = c[255];
    double totx = cx[255];
    if (lo < 1) lo = 1; if (hi > 256) hi = 256;
    double best = -1; int bt = lo;
    for (int t = lo; t < hi; t++) {
        double na = c[t-1], nb = n - na;
        if (na < min_count || nb < min_count) continue;
        double ma = cx[t-1] / na, mb = (totx - cx[t-1]) / nb;
        double va = cx2[t-1] / na - ma * ma;
        double vb = (cx2[255] - cx2[t-1]) / nb - mb * mb;
        double bal = (na < nb ? na : nb) / n / 0.5;
        double J = (mb - ma) * (mb - ma) / (va + vb + 1e-9) * bal;
        if (J > best) { best = J; bt = t; }
    }
    if (best_t) *best_t = bt;
    return best;
}

/* Bimodal valley depth from a histogram: smooth (box-5), exclude rails (0,254,255), find the
 * two dominant modes and the valley between them. depth = 1 - h_valley/min(h_dark,h_light) in
 * [0,1] (higher=deeper=better separated). Writes dark/light/valley bins if non-NULL. Returns
 * -1 if not bimodal. */
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


/* ---- whole-pipeline quality metrics (math in C; Python glue) ----
 * Computed in one pass over a float volume in [0,1]. */

/* Edge/sheet sharpness: mean gradient magnitude (central differences). Higher = sharper
 * boundaries (papyrus sheet edges). Skips a 1-voxel border. */
double fy_edge_sharpness(const float *v, int nz, int ny, int nx) {
    double s = 0; long cnt = 0;
    for (int z = 1; z < nz-1; z++)
        for (int y = 1; y < ny-1; y++)
            for (int x = 1; x < nx-1; x++) {
                size_t i = ((size_t)z*ny + y)*nx + x;
                double gz = v[i+(size_t)ny*nx] - v[i-(size_t)ny*nx];
                double gy = v[i+nx] - v[i-nx];
                double gx = v[i+1] - v[i-1];
                s += sqrt(gx*gx + gy*gy + gz*gz); cnt++;
            }
    return cnt ? s/cnt*0.5 : 0;
}

/* Dynamic-range usage: fraction of the [0,1] range actually occupied by the bulk (1-99
 * percentile span) -- how well the volume uses its bit depth. From a histogram. */
double fy_dynamic_range_usage(const long hist[256]) {
    double n = 0; for (int i = 0; i < 256; i++) n += hist[i];
    if (n <= 0) return 0;
    double acc = 0; int lo = 0, hi = 255;
    for (int i = 0; i < 256; i++) { acc += hist[i]; if (acc >= 0.01*n) { lo = i; break; } }
    acc = 0; for (int i = 255; i >= 0; i--) { acc += hist[i]; if (acc >= 0.01*n) { hi = i; break; } }
    return (hi - lo) / 255.0;
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
