/* glcae.c -- Global and Local Contrast Adaptive Enhancement (grayscale).
 *
 * Grayscale adaptation of Tian & Cohen, "Global and Local Contrast Adaptive
 * Enhancement for Non-uniform Illumination Color Images" (ICCV Workshops 2017).
 * Color/hue-preservation parts are dropped; the contrast machinery is kept.
 *
 * Pipeline:
 *   1. linear stretch to [0, L-1]
 *   2. GLOBAL: find lambda that blends the image histogram with a uniform one to
 *      minimize histogram "collisions" (adaptive between identity and full
 *      equalization), map via the blended CDF. This handles non-uniform
 *      illumination / shading globally.
 *   3. LOCAL: CLAHE (clip-limited adaptive histogram equalization) for local detail.
 *   4. FUSION: per-pixel blend of global & local, weighted by local contrast
 *      (Laplacian) x a brightness term (Gaussian around mid-gray).
 *
 * Works on a normalized [0,1] grayscale image/volume slice. L = 256 levels.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define L 256

/* histogram-collision cost for a given lambda (eq. in the paper's f()).
 * Lower = fewer gray levels merged. h_i normalized image histogram. */
static double collision_cost(double lam, const double *h_i, const double *h_u) {
    double h_tilde[L];
    for (int i = 0; i < L; i++)
        h_tilde[i] = (1.0 / (1.0 + lam)) * h_i[i] + (lam / (1.0 + lam)) * h_u[i];
    /* mapping t[i] = ceil((L-1)*cumsum + 0.5) */
    int t[L]; double cum = 0;
    for (int i = 0; i < L; i++) { cum += h_tilde[i]; t[i] = (int)ceil((L - 1) * cum + 0.5); }
    /* d = max distance between two occupied levels mapped to the same output */
    int d = 0;
    for (int i = 0; i < L; i++) {
        if (h_tilde[i] <= 0) continue;
        for (int j = 0; j <= i; j++)
            if (h_tilde[j] > 0 && t[i] == t[j] && (i - j) > d) d = i - j;
    }
    return (double)d;
}

/* golden-section minimization of collision_cost over lambda in [0, lam_max] */
static double optimize_lambda(const double *h_i, const double *h_u) {
    double a = 0.0, b = 8.0;          /* lambda range */
    const double gr = (sqrt(5.0) - 1.0) / 2.0;
    double c = b - gr * (b - a), d = a + gr * (b - a);
    double fc = collision_cost(c, h_i, h_u), fd = collision_cost(d, h_i, h_u);
    for (int it = 0; it < 40; it++) {
        if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = collision_cost(c, h_i, h_u); }
        else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = collision_cost(d, h_i, h_u); }
        if (fabs(b - a) < 1e-3) break;
    }
    return 0.5 * (a + b);
}

/* GLCAE global mapping from a normalized histogram h_i (sums to 1). Shared by the
 * per-slice path and the whole-volume streaming path. */
static void glcae_mapping_from_normhist(const double *h_i, int *t_out) {
    double h_u[L];
    for (int i = 0; i < L; i++) h_u[i] = 1.0 / L;
    double lam = optimize_lambda(h_i, h_u);
    double h_tilde[L], cum = 0;
    for (int i = 0; i < L; i++)
        h_tilde[i] = (1.0 / (1.0 + lam)) * h_i[i] + (lam / (1.0 + lam)) * h_u[i];
    for (int i = 0; i < L; i++) { cum += h_tilde[i]; int v = (int)ceil((L - 1) * cum + 0.5); t_out[i] = v < 0 ? 0 : (v > L - 1 ? L - 1 : v); }
}

/* build the global mapping CDF from a normalized [0,1] image (per-slice path) */
static void global_mapping(const float *img01, size_t n, int *t_out) {
    double h_i[L];
    for (int i = 0; i < L; i++) h_i[i] = 0;
    for (size_t k = 0; k < n; k++) {
        int b = (int)(img01[k] * (L - 1) + 0.5f);
        if (b < 0) b = 0; if (b >= L) b = L - 1;
        h_i[b] += 1.0;
    }
    for (int i = 0; i < L; i++) h_i[i] /= (double)n;
    glcae_mapping_from_normhist(h_i, t_out);
}

/* WHOLE-VOLUME streaming entry: build the GLCAE global mapping from an integer
 * histogram (256 bins) accumulated over the entire volume. Called once in the
 * two-pass streaming pipeline (stream.c). */
void fy_global_glcae_mapping_from_hist(const long *hist, long total, int *t_out) {
    double h_i[L];
    double inv = total > 0 ? 1.0 / (double)total : 0.0;
    for (int i = 0; i < L; i++) h_i[i] = hist[i] * inv;
    glcae_mapping_from_normhist(h_i, t_out);
}

/* fusion weight: local contrast (|Laplacian|) x brightness gaussian around mid */
static void fusion_weight(const float *g01, int ny, int nx, float *w) {
    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++) {
            size_t i = (size_t)y * nx + x;
            /* 4-neighbour Laplacian magnitude */
            float c = g01[i];
            float l = (x > 0 ? g01[i-1] : c) + (x < nx-1 ? g01[i+1] : c)
                    + (y > 0 ? g01[i-nx] : c) + (y < ny-1 ? g01[i+nx] : c) - 4*c;
            float cd = fabsf(l) + 1e-5f;
            float bd = expf(-(c - 0.5f) * (c - 0.5f) / (2.0f * 0.2f * 0.2f));
            w[i] = cd < bd ? cd : bd;   /* min, per the paper */
        }
}

/* 2D CLAHE on a normalized [0,1] slice (uses the 3D CLAHE with tiles_z=1 idea,
 * but implemented standalone 2D here for the per-slice fusion). */
extern int fy_clahe2d(const float *in, float *out, int ny, int nx,
                      int tiles_y, int tiles_x, int nbins, float clip_limit);

/* GLCAE on a single grayscale slice (normalized [0,1]) -> out (normalized [0,1]). */
int fy_glcae2d(const float *in, float *out, int ny, int nx,
               int clahe_tiles, float clahe_clip) {
    size_t n = (size_t)ny * nx;
    /* linear stretch to [0,1] */
    float vmin = 1e30f, vmax = -1e30f;
    for (size_t i = 0; i < n; i++) { if (in[i] < vmin) vmin = in[i]; if (in[i] > vmax) vmax = in[i]; }
    float inv = (vmax > vmin) ? 1.0f / (vmax - vmin) : 1.0f;
    float *s = malloc(sizeof(float) * n);
    if (!s) return 1;
    for (size_t i = 0; i < n; i++) s[i] = (in[i] - vmin) * inv;

    /* GLOBAL enhancement */
    int t[L];
    global_mapping(s, n, t);
    float *g = malloc(sizeof(float) * n);
    if (!g) { free(s); return 1; }
    for (size_t i = 0; i < n; i++) {
        int b = (int)(s[i] * (L - 1) + 0.5f); if (b < 0) b = 0; if (b >= L) b = L - 1;
        g[i] = t[b] / (float)(L - 1);
    }

    /* LOCAL enhancement (CLAHE) */
    float *lo = malloc(sizeof(float) * n);
    if (!lo) { free(s); free(g); return 1; }
    if (clahe_tiles < 1) clahe_tiles = 8;
    if (clahe_clip <= 0) clahe_clip = 2.0f;
    fy_clahe2d(s, lo, ny, nx, clahe_tiles, clahe_tiles, L, clahe_clip);

    /* FUSION: weight by each result's (local-contrast x brightness) */
    float *wg = malloc(sizeof(float) * n), *wl = malloc(sizeof(float) * n);
    if (!wg || !wl) { free(s); free(g); free(lo); free(wg); free(wl); return 1; }
    fusion_weight(g, ny, nx, wg);
    fusion_weight(lo, ny, nx, wl);
    for (size_t i = 0; i < n; i++) {
        float den = wg[i] + wl[i] + 1e-8f;
        out[i] = (wg[i] * g[i] + wl[i] * lo[i]) / den;
    }
    free(s); free(g); free(lo); free(wg); free(wl);
    return 0;
}
