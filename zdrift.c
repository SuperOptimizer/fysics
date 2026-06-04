/* zdrift.c -- z-axis intensity-drift / shading correction (whole-volume, streaming).
 *
 * The synchrotron beam current drifts during a scan (BM18 metadata shows ~13% drop,
 * ~87->76 mA start->stop), producing a slow brightness gradient along z: slices near
 * the scan's end are dimmer. This poisons any global intensity threshold or
 * normalization and biases downstream models. We remove it the post-recon way:
 * estimate the per-slice mean of the PAPYRUS (above a threshold, ignoring air),
 * smooth that z-profile (drift is slow), and divide each slice by it so brightness
 * is uniform in z.
 *
 * Two-pass streaming (the z-profile needs the whole volume, but it's one scalar per
 * slice -> tiny state):
 *   pass 1: fy_zdrift_accumulate() per z-slab  -> per-slice papyrus sum/count
 *   finalize: fy_zdrift_finalize()             -> smoothed correction factor per z
 *   pass 2: fy_zdrift_apply() per slice        -> multiply by its factor
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* pass 1: accumulate papyrus sum & count for slices [z0, z0+nz_slab) of a chunk.
 * sums[z]/counts[z] are indexed by GLOBAL z; caller sizes them to the full nz and
 * zeroes before the first call. papyrus_thresh in the same units as the data. */
void fy_zdrift_accumulate(const float *chunk, int nz_slab, int ny, int nx,
                          int z0, double *sums, long *counts, float papyrus_thresh) {
    for (int z = 0; z < nz_slab; z++) {
        double s = 0; long c = 0;
        const float *sl = chunk + (size_t)z * ny * nx;
        for (size_t i = 0; i < (size_t)ny * nx; i++)
            if (sl[i] > papyrus_thresh) { s += sl[i]; c++; }
        sums[z0 + z] += s;
        counts[z0 + z] += c;
    }
}

/* finalize: from per-slice sums/counts, produce a per-slice multiplicative
 * correction `factor[z]` that flattens the drift to the global papyrus mean.
 * The raw per-slice mean is smoothed (drift is slow; window ~ nz/20) so we remove
 * only the slow trend, not real slice-to-slice structure. */
void fy_zdrift_finalize(const double *sums, const long *counts, int nz,
                        float *factor) {
    double *mean = malloc(sizeof(double) * nz);
    double gsum = 0; long gcount = 0;
    for (int z = 0; z < nz; z++) {
        mean[z] = counts[z] ? sums[z] / counts[z] : 0.0;
        gsum += sums[z]; gcount += counts[z];
    }
    double gmean = gcount ? gsum / gcount : 1.0;
    int win = nz / 20 + 1;
    for (int z = 0; z < nz; z++) {
        double s = 0; int c = 0;
        for (int dz = -win; dz <= win; dz++) {
            int zz = z + dz;
            if (zz >= 0 && zz < nz && mean[zz] > 0) { s += mean[zz]; c++; }
        }
        double sm = c ? s / c : gmean;
        factor[z] = (sm > 1e-6) ? (float)(gmean / sm) : 1.0f;
    }
    free(mean);
}

/* pass 2: apply the per-slice correction to a chunk (slices [z0, z0+nz_slab)). */
void fy_zdrift_apply(float *chunk, int nz_slab, int ny, int nx, int z0,
                     const float *factor) {
    for (int z = 0; z < nz_slab; z++) {
        float f = factor[z0 + z];
        float *sl = chunk + (size_t)z * ny * nx;
        for (size_t i = 0; i < (size_t)ny * nx; i++) sl[i] *= f;
    }
}

/* convenience: correct a whole in-RAM volume in one call (small enough to fit). */
int fy_correct_zdrift(float *vol, int nz, int ny, int nx, float papyrus_thresh) {
    double *sums = calloc(nz, sizeof(double));
    long *counts = calloc(nz, sizeof(long));
    float *factor = malloc(sizeof(float) * nz);
    if (!sums || !counts || !factor) { free(sums); free(counts); free(factor); return 1; }
    fy_zdrift_accumulate(vol, nz, ny, nx, 0, sums, counts, papyrus_thresh);
    fy_zdrift_finalize(sums, counts, nz, factor);
    fy_zdrift_apply(vol, nz, ny, nx, 0, factor);
    free(sums); free(counts); free(factor);
    return 0;
}
