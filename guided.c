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

    /* X+Y FUSED per z-plane: smooth X into a plane-local L2-resident buffer (xp, ~81KB), then
     * slide that buffer in Y into `tmp`. The X intermediate never round-trips to RAM (it was a
     * full 11.9MB write+read of `out` before) -- big bandwidth saving on this RAM-bound kernel. */
    static __thread float *xp = NULL, *accbuf = NULL; static __thread int xcap = 0;
    int planesz = nx * ny;
    if (xcap < planesz) { free(xp); free(accbuf); xp = malloc(sizeof(float)*planesz); accbuf = malloc(sizeof(float)*nx); xcap = planesz; }
    for (int z = 0; z < nz; z++) {
        const float *iplane = in + (size_t)z * planesz;
        /* X-smooth this plane into xp (cache-resident) */
        for (int y = 0; y < ny; y++) {
            const float *irow = iplane + (size_t)y * nx;
            float *orow = xp + (size_t)y * nx;
            float s = 0;
            for (int x = 0; x <= r && x < nx; x++) s += irow[x];
            for (int x = 0; x < nx; x++) {
                orow[x] = s * rx[x];
                int xb = x + r + 1, xa = x - r;
                if (xb < nx) s += irow[xb];
                if (xa >= 0) s -= irow[xa];
            }
        }
        /* Y-smooth xp (in L2) -> tmp plane */
        float *oplane = tmp + (size_t)z * planesz;
        float *acc = accbuf;
        for (int x = 0; x < nx; x++) acc[x] = 0;
        for (int y = 0; y <= r && y < ny; y++) { const float *row = xp + (size_t)y*nx; for (int x=0;x<nx;x++) acc[x]+=row[x]; }
        for (int y = 0; y < ny; y++) {
            float *orow = oplane + (size_t)y * nx; float ryv = ry[y];
            for (int x = 0; x < nx; x++) orow[x] = acc[x] * ryv;
            int yb = y + r + 1, ya = y - r;
            if (yb < ny) { const float *row = xp + (size_t)yb*nx; for (int x=0;x<nx;x++) acc[x]+=row[x]; }
            if (ya >= 0) { const float *row = xp + (size_t)ya*nx; for (int x=0;x<nx;x++) acc[x]-=row[x]; }
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

/* ---- FAST guided filter (He & Sun 2015, arXiv:1505.00996) ----
 * Compute the filter coefficients (a,b) on an s-times-decimated grid with radius
 * radius/s, then trilinearly upsample mean_a/mean_b and apply out = a*I + b at
 * full resolution. The box statistics are low-pass at the window scale, so the
 * coefficient fields are smooth and the subsampling changes the result
 * negligibly -- while cutting the box-filter work (the pipeline's dominant
 * cost, ~78% of runtime) by ~s^3. Decimation (not block-mean) is used for the
 * downsample so the per-sample noise VARIANCE is preserved and the calibrated
 * eps (fy_guided_eps_for_noise) keeps its meaning. s=2 recommended; s<=1 or a
 * tile too small to decimate falls back to the exact path. Same ws as exact. */
int fy_guided_denoise_fast_ws(const float *in, float *out, int nz, int ny, int nx,
                              int radius, double eps, int s, float *ws) {
    if (radius < 1) radius = 2;
    if (s <= 1) return fy_guided_denoise_ws(in, out, nz, ny, nx, radius, eps, ws);
    int lz = (nz + s - 1) / s, ly = (ny + s - 1) / s, lx = (nx + s - 1) / s;
    if (lz < 2 || ly < 2 || lx < 2)
        return fy_guided_denoise_ws(in, out, nz, ny, nx, radius, eps, ws);
    size_t nl = (size_t)lz * ly * lx;
    /* low-res buffers carved from ws: 5*nl <= 5n/8 < 4n for s=2 */
    float *Il = ws, *mean = ws + nl, *corr = ws + 2 * nl, *ab = ws + 3 * nl, *tmp = ws + 4 * nl;

    for (int z = 0; z < lz; z++)
        for (int y = 0; y < ly; y++) {
            const float *irow = in + ((size_t)(z * s) * ny + (size_t)(y * s)) * nx;
            float *orow = Il + ((size_t)z * ly + y) * lx;
            for (int x = 0; x < lx; x++) orow[x] = irow[(size_t)x * s];
        }

    int rl = radius / s; if (rl < 1) rl = 1;
    box_mean(Il, mean, tmp, lz, ly, lx, rl);
    for (size_t i = 0; i < nl; i++) corr[i] = Il[i] * Il[i];
    box_mean(corr, ab, tmp, lz, ly, lx, rl);
    for (size_t i = 0; i < nl; i++) {
        float var = ab[i] - mean[i] * mean[i];
        if (var < 0) var = 0;
        float ai = var / (var + (float)eps);
        corr[i] = ai;                       /* corr := a */
        ab[i]   = mean[i] * (1.0f - ai);    /* ab   := b */
    }
    box_mean(corr, mean, tmp, lz, ly, lx, rl);   /* mean := mean_a */
    box_mean(ab,   corr, tmp, lz, ly, lx, rl);   /* corr := mean_b */

    /* trilinear upsample of (mean_a, mean_b); apply at full res. Low sample i
     * sits at full coordinate i*s (decimation), so u = x/s exactly. */
    int *xi = (int *)malloc(sizeof(int) * (size_t)nx);
    float *xf = (float *)malloc(sizeof(float) * (size_t)nx);
    if (!xi || !xf) { free(xi); free(xf); return 1; }
    for (int x = 0; x < nx; x++) {
        float u = (float)x / s;
        int x0 = (int)u; if (x0 > lx - 2) x0 = lx - 2;
        float f = u - x0; if (f > 1.0f) f = 1.0f;
        xi[x] = x0; xf[x] = f;
    }
    for (int z = 0; z < nz; z++) {
        float uz = (float)z / s;
        int z0 = (int)uz; if (z0 > lz - 2) z0 = lz - 2;
        float zf = uz - z0; if (zf > 1.0f) zf = 1.0f;
        for (int y = 0; y < ny; y++) {
            float uy = (float)y / s;
            int y0 = (int)uy; if (y0 > ly - 2) y0 = ly - 2;
            float yfr = uy - y0; if (yfr > 1.0f) yfr = 1.0f;
            size_t r00 = ((size_t)z0 * ly + y0) * lx;
            size_t r01 = r00 + lx;                  /* y0+1 */
            size_t r10 = r00 + (size_t)ly * lx;     /* z0+1 */
            size_t r11 = r10 + lx;
            float w00 = (1 - zf) * (1 - yfr), w01 = (1 - zf) * yfr;
            float w10 = zf * (1 - yfr),       w11 = zf * yfr;
            const float *irow = in + ((size_t)z * ny + y) * nx;
            float *orow = out + ((size_t)z * ny + y) * nx;
            for (int x = 0; x < nx; x++) {
                int x0 = xi[x]; float fx = xf[x], gx = 1 - fx;
                float a = gx * (w00 * mean[r00 + x0] + w01 * mean[r01 + x0] +
                                w10 * mean[r10 + x0] + w11 * mean[r11 + x0]) +
                          fx * (w00 * mean[r00 + x0 + 1] + w01 * mean[r01 + x0 + 1] +
                                w10 * mean[r10 + x0 + 1] + w11 * mean[r11 + x0 + 1]);
                float b = gx * (w00 * corr[r00 + x0] + w01 * corr[r01 + x0] +
                                w10 * corr[r10 + x0] + w11 * corr[r11 + x0]) +
                          fx * (w00 * corr[r00 + x0 + 1] + w01 * corr[r01 + x0 + 1] +
                                w10 * corr[r10 + x0 + 1] + w11 * corr[r11 + x0 + 1]);
                orow[x] = a * irow[x] + b;
            }
        }
    }
    free(xi); free(xf);
    return 0;
}

int fy_guided_denoise(const float *in, float *out, int nz, int ny, int nx,
                      int radius, double eps) {
    size_t n = (size_t)nz * ny * nx;
    float *ws = malloc(sizeof(float) * 4 * n);
    if (!ws) return 1;
    int rc = fy_guided_denoise_ws(in, out, nz, ny, nx, radius, eps, ws);
    free(ws);
    return rc;
}
