/* fsc.c -- Fourier Shell Correlation (FSC): the synchrotron-native resolution metric.
 *
 * Measures the spatial resolution actually present in a volume by correlating two
 * independent half-data-sets shell-by-shell in Fourier space; resolution = the
 * frequency where correlation drops below a threshold (commonly 0.143 or 0.5).
 * This is how to PROVE a processing step (deconv, sharpening) genuinely recovered
 * resolution rather than just looking sharper -- if FSC extends to higher frequency
 * after processing, resolution improved; if not, it just amplified noise.
 *
 * Practical reduced-reference use (no second acquisition): split the volume into
 * two interleaved half-sets (even/odd voxels along an axis, or a checkerboard),
 * FSC them, read the cutoff. Returns the per-shell correlation curve; the caller
 * finds where it crosses the threshold.
 *
 *   FSC(shell) = Re[ sum F1*conj(F2) ] / sqrt( sum|F1|^2 * sum|F2|^2 )   over the shell
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Compute FSC between two equal-shaped volumes. nbins shells from 0..Nyquist.
 * Writes freqs[nbins] (cycles/voxel) and fsc[nbins]. Returns 0 on success. */
int fy_fsc(const float *vol1, const float *vol2, int nz, int ny, int nx,
           int nbins, float *freqs, float *fsc) {
    int pz = fy_next_pow2(nz), py = fy_next_pow2(ny), px = fy_next_pow2(nx);
    size_t np = (size_t)pz * py * px;
    float *r1 = calloc(np, sizeof(float)), *i1 = calloc(np, sizeof(float));
    float *r2 = calloc(np, sizeof(float)), *i2 = calloc(np, sizeof(float));
    if (!r1 || !i1 || !r2 || !i2) { free(r1); free(i1); free(r2); free(i2); return 1; }
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) {
        size_t s = ((size_t)z*ny+y)*nx, d = ((size_t)z*py+y)*px;
        for (int x = 0; x < nx; x++) { r1[d+x] = vol1[s+x]; r2[d+x] = vol2[s+x]; }
    }
    fy_fft3d(r1, i1, pz, py, px, -1);
    fy_fft3d(r2, i2, pz, py, px, -1);

    double *num = calloc(nbins, sizeof(double));
    double *d1 = calloc(nbins, sizeof(double)), *d2 = calloc(nbins, sizeof(double));
    if (!num || !d1 || !d2) { free(r1);free(i1);free(r2);free(i2);free(num);free(d1);free(d2); return 1; }
    for (int z = 0; z < pz; z++) {
        double fz = (z <= pz/2) ? (double)z/pz : (double)(z-pz)/pz;
        for (int y = 0; y < py; y++) {
            double fy_ = (y <= py/2) ? (double)y/py : (double)(y-py)/py;
            size_t row = ((size_t)z*py+y)*px;
            for (int x = 0; x < px; x++) {
                double fx = (x <= px/2) ? (double)x/px : (double)(x-px)/px;
                double fr = sqrt(fz*fz + fy_*fy_ + fx*fx);
                int b = (int)(fr / 0.5 * nbins);     /* 0..Nyquist(0.5) -> nbins */
                if (b < 0 || b >= nbins) continue;
                double a1 = r1[row+x], b1 = i1[row+x], a2 = r2[row+x], b2 = i2[row+x];
                num[b] += a1*a2 + b1*b2;             /* Re[F1 conj(F2)] */
                d1[b]  += a1*a1 + b1*b1;
                d2[b]  += a2*a2 + b2*b2;
            }
        }
    }
    for (int b = 0; b < nbins; b++) {
        freqs[b] = (float)((b + 0.5) * 0.5 / nbins);
        double den = sqrt(d1[b] * d2[b]);
        fsc[b] = (den > 1e-12) ? (float)(num[b] / den) : 0.0f;
    }
    free(r1);free(i1);free(r2);free(i2);free(num);free(d1);free(d2);
    return 0;
}

/* Reduced-reference FSC: split ONE volume into two interleaved half-sets
 * (checkerboard by (x+y+z) parity, each upsampled by nearest), FSC them. Gives a
 * resolution estimate from a single volume. Returns the shell at which FSC first
 * drops below `threshold` (e.g. 0.143) as a fraction of Nyquist in *res_frac;
 * also fills freqs/fsc if non-NULL. */
int fy_fsc_self(const float *vol, int nz, int ny, int nx, int nbins,
                float threshold, float *res_frac, float *freqs, float *fsc) {
    /* Split by EVEN/ODD x-slices into two independent half-volumes of width nx/2.
     * Two genuine sub-samples of the same signal: their FSC measures the resolution
     * actually shared between them. (Checkerboard self-fill is unreliable; the
     * even/odd decimation is the standard, correct split.) */
    int hx = nx / 2;
    if (hx < 4) { return 2; }  /* too small to split meaningfully */
    size_t hn = (size_t)nz * ny * hx;
    float *h1 = malloc(sizeof(float)*hn), *h2 = malloc(sizeof(float)*hn);
    if (!h1 || !h2) { free(h1); free(h2); return 1; }
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < hx; x++) {
        size_t d = ((size_t)z*ny+y)*hx + x;
        h1[d] = vol[((size_t)z*ny+y)*nx + 2*x];      /* even x */
        h2[d] = vol[((size_t)z*ny+y)*nx + 2*x + 1];  /* odd x  */
    }
    nx = hx;  /* FSC on the half-width volumes */
    float *fr = freqs ? freqs : malloc(sizeof(float)*nbins);
    float *fc = fsc ? fsc : malloc(sizeof(float)*nbins);
    int rc = fy_fsc(h1, h2, nz, ny, nx, nbins, fr, fc);
    if (rc == 0 && res_frac) {
        *res_frac = 1.0f;  /* default: resolved to Nyquist */
        for (int b = 1; b < nbins; b++)
            if (fc[b] < threshold) { *res_frac = fr[b] / 0.5f; break; }
    }
    if (!freqs) free(fr); if (!fsc) free(fc);
    free(h1); free(h2);
    return rc;
}
