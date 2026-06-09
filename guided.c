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
/* precompute 1/window_count for each position along a dimension of length L (count varies only
 * within r of the two edges; interior is the full 2r+1). Caller provides a buffer of length L. */
static void box_recip(float *recip, int L, int r) {
    for (int i = 0; i < L; i++) {
        int lo = i - r; if (lo < 0) lo = 0;
        int hi = i + r; if (hi > L - 1) hi = L - 1;
        recip[i] = 1.0f / (float)(hi - lo + 1);
    }
}

/* Separable 3D box mean via sliding window. Float accumulator + PRECOMPUTED reciprocals
 * (no per-voxel int->double convert, no per-voxel divide) -- this is the pipeline's hottest
 * kernel (~78% of runtime), so the scalar double accumulator + divide it replaced were the
 * bottleneck. The Y/Z passes are made cache-friendly by processing a contiguous X-row at a
 * time (sliding the whole row), so the inner loop is unit-stride over x. */
static void box_mean(const float *restrict in, float *restrict out, float *restrict tmp,
                     int nz, int ny, int nx, int r) {
    /* per-dimension reciprocal tables (small; reused across all rows/cols) */
    int Lmax = nx > ny ? (nx > nz ? nx : nz) : (ny > nz ? ny : nz);
    float *rx = malloc(sizeof(float) * (size_t)Lmax * 3);
    float *ry = rx + Lmax, *rz = ry + Lmax;
    box_recip(rx, nx, r); box_recip(ry, ny, r); box_recip(rz, nz, r);

    /* X: unit-stride sliding sum along each row. */
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) {
        const float *irow = in + ((size_t)z * ny + y) * nx;
        float *orow = out + ((size_t)z * ny + y) * nx;
        float s = 0;
        for (int x = 0; x <= r && x < nx; x++) s += irow[x];
        for (int x = 0; x < nx; x++) {
            orow[x] = s * rx[x];
            int xb = x + r + 1, xa = x - r;
            if (xb < nx) s += irow[xb];
            if (xa >= 0) s -= irow[xa];
        }
    }
    /* Y: slide whole X-ROWS (unit-stride vector adds over x). out->tmp. */
    for (int z = 0; z < nz; z++) {
        const float *plane = out + (size_t)z * ny * nx;
        float *oplane = tmp + (size_t)z * ny * nx;
        float *acc = oplane;                      /* reuse first row region as running sum? no -- need scratch */
        /* running sum row in a small buffer */
        static __thread float *accbuf = NULL; static __thread int acccap = 0;
        if (acccap < nx) { free(accbuf); accbuf = malloc(sizeof(float) * nx); acccap = nx; }
        acc = accbuf;
        for (int x = 0; x < nx; x++) acc[x] = 0;
        for (int y = 0; y <= r && y < ny; y++) { const float *row = plane + (size_t)y*nx; for (int x=0;x<nx;x++) acc[x]+=row[x]; }
        for (int y = 0; y < ny; y++) {
            float *orow = oplane + (size_t)y * nx; float ryv = ry[y];
            for (int x = 0; x < nx; x++) orow[x] = acc[x] * ryv;
            int yb = y + r + 1, ya = y - r;
            if (yb < ny) { const float *row = plane + (size_t)yb*nx; for (int x=0;x<nx;x++) acc[x]+=row[x]; }
            if (ya >= 0) { const float *row = plane + (size_t)ya*nx; for (int x=0;x<nx;x++) acc[x]-=row[x]; }
        }
    }
    /* Z: slide whole X-ROWS across planes. tmp->out. */
    {
        static __thread float *zacc = NULL; static __thread int zcap = 0;
        if (zcap < nx) { free(zacc); zacc = malloc(sizeof(float) * nx); zcap = nx; }
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) zacc[x] = 0;
            for (int z = 0; z <= r && z < nz; z++) { const float *row = tmp + ((size_t)z*ny+y)*nx; for (int x=0;x<nx;x++) zacc[x]+=row[x]; }
            for (int z = 0; z < nz; z++) {
                float *orow = out + ((size_t)z*ny+y)*nx; float rzv = rz[z];
                for (int x = 0; x < nx; x++) orow[x] = zacc[x] * rzv;
                int zb = z + r + 1, za = z - r;
                if (zb < nz) { const float *row = tmp + ((size_t)zb*ny+y)*nx; for (int x=0;x<nx;x++) zacc[x]+=row[x]; }
                if (za >= 0) { const float *row = tmp + ((size_t)za*ny+y)*nx; for (int x=0;x<nx;x++) zacc[x]-=row[x]; }
            }
        }
    }
    free(rx);
}

/* Plain 3D box smooth (mean over (2r+1)^3) -- ONE separable box pass, no variance/edge math.
 * Used for the air-cut SCRATCH (which only needs to tighten the histogram modes for a valley
 * decision, not preserve edges): a single box r=5 matches a 5-pass guided eps=0.01 scratch
 * (valley within 1 u8) at ~1/4 the cost. `tmp` = n-float scratch. in/out may alias. */
void fy_box_smooth(const float *in, float *out, float *tmp, int nz, int ny, int nx, int r) {
    box_mean(in, out, tmp, nz, ny, nx, r);
}

/* self-guided guided filter. eps in (normalized intensity)^2, e.g. 0.01^2..0.1^2.
 * radius r voxels. in/out (may differ). Returns 0 on success. */
/* workspace version: caller provides `ws` = 6*n contiguous floats (see fy_guided_ws_floats).
 * Avoids the 6 malloc/free per call -- critical in the per-tile pass-2 loop (thousands of calls)
 * where the alloc churn page-faults. in/out may alias each other but NOT ws. */
int fy_guided_denoise_ws(const float *in, float *out, int nz, int ny, int nx,
                         int radius, double eps, float *ws) {
    if (radius < 1) radius = 2;
    size_t n = (size_t)nz * ny * nx;
    /* Only 4 work buffers (was 6) by reusing dead lifetimes -> 33% smaller working set so more of
     * the per-tile data stays L3-resident across the 12 box sweeps (the kernel is RAM-bandwidth
     * bound; less footprint = fewer RAM round-trips). buf0..buf3 + tmp share `ws`. */
    float *mean = ws, *corr = ws + n, *ab = ws + 2*n, *tmp = ws + 3*n;
    /* mean = boxmean(I);  corr = boxmean(I^2)  (compute I^2 into corr in-place via tmp) */
    box_mean(in, mean, tmp, nz, ny, nx, radius);
    for (size_t i = 0; i < n; i++) corr[i] = in[i] * in[i];   /* corr := I^2 */
    box_mean(corr, ab, tmp, nz, ny, nx, radius);              /* ab := boxmean(I^2) */
    /* a := var/(var+eps) into `corr`; b := mean*(1-a) into `ab` (overwrites mean(I^2)). */
    for (size_t i = 0; i < n; i++) {
        float var = ab[i] - mean[i] * mean[i];
        if (var < 0) var = 0;
        float ai = var / (var + (float)eps);
        corr[i] = ai;                       /* corr := a */
        ab[i]   = mean[i] * (1.0f - ai);    /* ab   := b */
    }
    box_mean(corr, mean, tmp, nz, ny, nx, radius);   /* mean := mean_a */
    box_mean(ab,   corr, tmp, nz, ny, nx, radius);   /* corr := mean_b */
    for (size_t i = 0; i < n; i++) out[i] = mean[i] * in[i] + corr[i];
    return 0;
}

/* number of float scratch elements fy_guided_denoise_ws needs for an nz*ny*nx volume. */
size_t fy_guided_ws_floats(int nz, int ny, int nx) { return (size_t)4 * nz * ny * nx; }

int fy_guided_denoise(const float *in, float *out, int nz, int ny, int nx,
                      int radius, double eps) {
    size_t n = (size_t)nz * ny * nx;
    float *ws = malloc(sizeof(float) * 4 * n);
    if (!ws) return 1;
    int rc = fy_guided_denoise_ws(in, out, nz, ny, nx, radius, eps, ws);
    free(ws);
    return rc;
}
