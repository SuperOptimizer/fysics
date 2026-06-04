/* guided.c -- 3D guided filter (He, Sun, Tang). Fast edge-preserving denoise.
 *
 * O(N) regardless of window size -- built entirely from box filters (running
 * sums), so it's far faster than bilateral (~100x at large windows) and has NO
 * gradient-reversal artifacts. Recommended as the default fast denoiser for
 * streaming 20TB volumes (research: guided filter ranked #1 on speed x simplicity).
 *
 * Self-guided (guide == input): edge-preserving smoothing.
 *   a = var/(var+eps),  b = mean*(1-a),  out = a*I + b   (with a,b box-averaged)
 * eps = range parameter (smaller = preserve more edges/texture). r = window radius.
 *
 * Box filter = three separable 1D moving-sum passes (vectorizable). For streaming,
 * tile with an r-voxel halo.
 */
#include "fysics.h"
#include <stdlib.h>
#include <string.h>

/* separable 3D box filter (mean over (2r+1)^3) via 1D moving sums */
static void box_mean(const float *restrict in, float *restrict out, float *restrict tmp,
                     int nz, int ny, int nx, int r) {
    size_t n = (size_t)nz * ny * nx;
    /* X */
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) {
        size_t row = ((size_t)z * ny + y) * nx;
        double s = 0; int cnt = 0;
        for (int x = 0; x <= r && x < nx; x++) { s += in[row + x]; cnt++; }
        for (int x = 0; x < nx; x++) {
            out[row + x] = (float)(s / cnt);
            int xa = x - r, xb = x + r + 1;
            if (xb < nx) { s += in[row + xb]; cnt++; }
            if (xa >= 0) { s -= in[row + xa]; cnt--; }
        }
    }
    /* Y (out->tmp) */
    for (int z = 0; z < nz; z++) for (int x = 0; x < nx; x++) {
        double s = 0; int cnt = 0;
        for (int y = 0; y <= r && y < ny; y++) { s += out[((size_t)z*ny+y)*nx+x]; cnt++; }
        for (int y = 0; y < ny; y++) {
            tmp[((size_t)z*ny+y)*nx+x] = (float)(s / cnt);
            int ya = y - r, yb = y + r + 1;
            if (yb < ny) { s += out[((size_t)z*ny+yb)*nx+x]; cnt++; }
            if (ya >= 0) { s -= out[((size_t)z*ny+ya)*nx+x]; cnt--; }
        }
    }
    /* Z (tmp->out) */
    for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        double s = 0; int cnt = 0;
        for (int z = 0; z <= r && z < nz; z++) { s += tmp[((size_t)z*ny+y)*nx+x]; cnt++; }
        for (int z = 0; z < nz; z++) {
            out[((size_t)z*ny+y)*nx+x] = (float)(s / cnt);
            int za = z - r, zb = z + r + 1;
            if (zb < nz) { s += tmp[((size_t)zb*ny+y)*nx+x]; cnt++; }
            if (za >= 0) { s -= tmp[((size_t)za*ny+y)*nx+x]; cnt--; }
        }
    }
    (void)n;
}

/* self-guided guided filter. eps in (normalized intensity)^2, e.g. 0.01^2..0.1^2.
 * radius r voxels. in/out (may differ). Returns 0 on success. */
int fy_guided_denoise(const float *in, float *out, int nz, int ny, int nx,
                      int radius, double eps) {
    if (radius < 1) radius = 2;
    size_t n = (size_t)nz * ny * nx;
    float *mean = malloc(sizeof(float) * n);
    float *corr = malloc(sizeof(float) * n);   /* mean of I*I */
    float *isq  = malloc(sizeof(float) * n);
    float *a    = malloc(sizeof(float) * n);
    float *b    = malloc(sizeof(float) * n);
    float *tmp  = malloc(sizeof(float) * n);
    if (!mean || !corr || !isq || !a || !b || !tmp) {
        free(mean); free(corr); free(isq); free(a); free(b); free(tmp); return 1;
    }
    for (size_t i = 0; i < n; i++) isq[i] = in[i] * in[i];
    box_mean(in, mean, tmp, nz, ny, nx, radius);
    box_mean(isq, corr, tmp, nz, ny, nx, radius);
    /* a = var/(var+eps); b = mean*(1-a) */
    for (size_t i = 0; i < n; i++) {
        float var = corr[i] - mean[i] * mean[i];
        if (var < 0) var = 0;
        float ai = var / (var + (float)eps);
        a[i] = ai;
        b[i] = mean[i] * (1.0f - ai);
    }
    /* average a,b over the window, then out = a*I + b */
    box_mean(a, corr, tmp, nz, ny, nx, radius);   /* corr = mean_a */
    box_mean(b, isq,  tmp, nz, ny, nx, radius);    /* isq  = mean_b */
    for (size_t i = 0; i < n; i++) out[i] = corr[i] * in[i] + isq[i];
    free(mean); free(corr); free(isq); free(a); free(b); free(tmp);
    return 0;
}
