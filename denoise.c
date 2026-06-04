/* denoise.c -- 3D non-local means denoising (pure C, vectorizable).
 *
 * NLM is a natural complement to deconvolution: deconvolution sharpens but
 * amplifies noise; NLM removes noise while preserving structure by averaging over
 * SIMILAR neighborhoods. Papyrus texture is self-similar (repeating fibers/sheets),
 * which is exactly the regime where NLM excels.
 *
 * For each voxel we average voxels in a search window, weighted by the similarity
 * of their surrounding patches: w = exp(-||patch_i - patch_j||^2 / h^2). The patch
 * L2 distance inner loop is the vectorization target.
 *
 * Cost ~ N * (2S+1)^3 * (2P+1)^3. Keep S,P small (S=5,P=1 typical) for speed.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* squared L2 distance between patches centered at (az,ay,ax) and (bz,by,bx),
 * patch half-width P. Boundary voxels are clamped. This is the hot loop. */
static double patch_dist2(const float *restrict v, int nz, int ny, int nx,
                          int az, int ay, int ax, int bz, int by, int bx, int P) {
    double acc = 0.0;
    for (int dz = -P; dz <= P; dz++) {
        int az1 = clampi(az + dz, 0, nz - 1), bz1 = clampi(bz + dz, 0, nz - 1);
        for (int dy = -P; dy <= P; dy++) {
            int ay1 = clampi(ay + dy, 0, ny - 1), by1 = clampi(by + dy, 0, ny - 1);
            size_t arow = ((size_t)az1 * ny + ay1) * nx;
            size_t brow = ((size_t)bz1 * ny + by1) * nx;
            for (int dx = -P; dx <= P; dx++) {
                int ax1 = clampi(ax + dx, 0, nx - 1), bx1 = clampi(bx + dx, 0, nx - 1);
                float d = v[arow + ax1] - v[brow + bx1];
                acc += (double)d * d;
            }
        }
    }
    return acc;
}

/* 3D non-local means. h = filter strength (~ noise sigma). search_radius S,
 * patch_radius P. in/out may differ; out must be allocated (nz*ny*nx). */
int fy_nlm_denoise(const float *in, float *out, int nz, int ny, int nx,
                   double h, int search_radius, int patch_radius) {
    if (search_radius < 1) search_radius = 5;
    if (patch_radius < 1) patch_radius = 1;
    int S = search_radius, P = patch_radius;
    int npatch = (2 * P + 1) * (2 * P + 1) * (2 * P + 1);
    double inv_h2 = 1.0 / (h * h * npatch + 1e-12);

    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                double wsum = 0.0, vsum = 0.0, wmax = 0.0;
                int z0 = z - S, z1 = z + S, y0 = y - S, y1 = y + S, x0 = x - S, x1 = x + S;
                for (int sz = z0; sz <= z1; sz++) {
                    int szc = clampi(sz, 0, nz - 1);
                    for (int sy = y0; sy <= y1; sy++) {
                        int syc = clampi(sy, 0, ny - 1);
                        for (int sx = x0; sx <= x1; sx++) {
                            int sxc = clampi(sx, 0, nx - 1);
                            if (szc == z && syc == y && sxc == x) continue; /* skip self for now */
                            double d2 = patch_dist2(in, nz, ny, nx, z, y, x, szc, syc, sxc, P);
                            double w = exp(-d2 * inv_h2);
                            if (w > wmax) wmax = w;
                            wsum += w;
                            vsum += w * in[((size_t)szc * ny + syc) * nx + sxc];
                        }
                    }
                }
                /* give the center voxel the max neighbor weight (standard NLM trick) */
                if (wmax <= 0.0) wmax = 1.0;
                wsum += wmax;
                vsum += wmax * in[((size_t)z * ny + y) * nx + x];
                out[((size_t)z * ny + y) * nx + x] = (float)(vsum / wsum);
            }
        }
    }
    return 0;
}

/* Bilateral filter: cheaper edge-preserving denoise (range + spatial gaussian).
 * Good fast alternative to NLM when speed matters. */
int fy_bilateral_denoise(const float *in, float *out, int nz, int ny, int nx,
                         double sigma_spatial, double sigma_range, int radius) {
    if (radius < 1) radius = (int)(2.0 * sigma_spatial + 0.5);
    if (radius < 1) radius = 1;
    int R = radius;
    double inv_s2 = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
    double inv_r2 = 1.0 / (2.0 * sigma_range * sigma_range);

    /* precompute spatial weights */
    int W = 2 * R + 1;
    double *sw = (double *)malloc(sizeof(double) * W * W * W);
    if (!sw) return 1;
    for (int dz = -R; dz <= R; dz++)
        for (int dy = -R; dy <= R; dy++)
            for (int dx = -R; dx <= R; dx++) {
                double r2 = (double)(dz*dz + dy*dy + dx*dx);
                sw[((dz+R)*W + (dy+R))*W + (dx+R)] = exp(-r2 * inv_s2);
            }

    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                float center = in[((size_t)z * ny + y) * nx + x];
                double wsum = 0.0, vsum = 0.0;
                for (int dz = -R; dz <= R; dz++) {
                    int zc = clampi(z + dz, 0, nz - 1);
                    for (int dy = -R; dy <= R; dy++) {
                        int yc = clampi(y + dy, 0, ny - 1);
                        size_t row = ((size_t)zc * ny + yc) * nx;
                        const double *swrow = &sw[((dz+R)*W + (dy+R))*W + R];
                        for (int dx = -R; dx <= R; dx++) {
                            int xc = clampi(x + dx, 0, nx - 1);
                            float val = in[row + xc];
                            float diff = val - center;
                            double w = swrow[dx] * exp(-(double)diff * diff * inv_r2);
                            wsum += w; vsum += w * val;
                        }
                    }
                }
                out[((size_t)z * ny + y) * nx + x] = (float)(vsum / wsum);
            }
    free(sw);
    return 0;
}
