/* rings.c -- ring-artifact removal for tomographic slices (pure C).
 *
 * Ring artifacts (concentric circles centered on the rotation axis) come from
 * detector pixel defects / miscalibration that survive flatfield + the recon's
 * own ring correction. They're a real, visible nuisance in scroll volumes.
 *
 * This is a HEURISTIC cleanup, not a physics inverse: in polar coordinates
 * (centered on the rotation axis) a concentric ring becomes a straight line at
 * constant radius. Averaging each slice over angle gives a 1-D radial profile;
 * its high-frequency component is the ring signal (real structure varies with
 * angle, rings don't). We high-pass that radial profile and subtract it.
 *
 * Operates slice-by-slice in the XY plane (rings are in-plane, per z). The center
 * defaults to the slice center; pass it explicitly if the rotation axis differs.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 1-D moving-average smooth (for the high-pass of the radial profile) */
static void smooth1d(const double *in, double *out, int n, int win) {
    if (win < 1) win = 1;
    for (int i = 0; i < n; i++) {
        double acc = 0; int cnt = 0;
        for (int j = i - win; j <= i + win; j++) {
            if (j >= 0 && j < n) { acc += in[j]; cnt++; }
        }
        out[i] = acc / cnt;
    }
}

/* Remove rings from a single XY slice in place.
 *   cx,cy: rotation center (use nx/2,ny/2 if unknown)
 *   nr: number of radial bins (default ~ max radius)
 *   strength: 0..1, fraction of the ring estimate to subtract (1 = full)
 *   smooth_win: radial smoothing window for the high-pass (larger = only sharper rings)
 */
static void dering_slice(float *slice, int ny, int nx, double cx, double cy,
                         double strength, int smooth_win) {
    double rmax = hypot((nx > cx ? nx - cx : cx), (ny > cy ? ny - cy : cy));
    int nr = (int)(rmax + 1.5);
    if (nr < 4) return;
    double *profile = (double *)calloc(nr, sizeof(double));
    double *count = (double *)calloc(nr, sizeof(double));
    if (!profile || !count) { free(profile); free(count); return; }

    /* angular-average -> radial profile (rings show up here; real structure averages out) */
    for (int y = 0; y < ny; y++) {
        double dy = y - cy;
        for (int x = 0; x < nx; x++) {
            double dx = x - cx;
            double r = sqrt(dx * dx + dy * dy);
            int ri = (int)(r + 0.5);
            if (ri >= 0 && ri < nr) {
                profile[ri] += slice[(size_t)y * nx + x];
                count[ri] += 1.0;
            }
        }
    }
    for (int i = 0; i < nr; i++) if (count[i] > 0) profile[i] /= count[i];

    /* high-pass: ring component = profile - smooth(profile) */
    double *smooth = (double *)malloc(sizeof(double) * nr);
    if (!smooth) { free(profile); free(count); return; }
    smooth1d(profile, smooth, nr, smooth_win);
    for (int i = 0; i < nr; i++) profile[i] = (profile[i] - smooth[i]) * strength;

    /* subtract the radial ring estimate back from each voxel */
    for (int y = 0; y < ny; y++) {
        double dy = y - cy;
        for (int x = 0; x < nx; x++) {
            double dx = x - cx;
            int ri = (int)(sqrt(dx * dx + dy * dy) + 0.5);
            if (ri >= 0 && ri < nr)
                slice[(size_t)y * nx + x] -= (float)profile[ri];
        }
    }
    free(profile); free(count); free(smooth);
}

/* Remove rings from a 3-D volume (each XY slice). center<0 -> use slice center.
 * strength in [0,1], smooth_win = radial high-pass window (e.g. 30). */
int fy_remove_rings(const float *in, float *out, int nz, int ny, int nx,
                    double center_x, double center_y, double strength, int smooth_win) {
    if (in != out) memcpy(out, in, sizeof(float) * (size_t)nz * ny * nx);
    double cx = center_x >= 0 ? center_x : nx / 2.0;
    double cy = center_y >= 0 ? center_y : ny / 2.0;
    if (smooth_win < 1) smooth_win = 30;
    if (strength <= 0) strength = 1.0;
    for (int z = 0; z < nz; z++)
        dering_slice(out + (size_t)z * ny * nx, ny, nx, cx, cy, strength, smooth_win);
    return 0;
}
