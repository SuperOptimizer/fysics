/* mask.c -- papyrus / air segmentation and masked processing.
 *
 * The big practical problem with sharpening scroll volumes: the AIR GAPS between
 * papyrus sheets have no signal, so deconvolution there just amplifies pure noise
 * ("air noise"). The fix is to detect papyrus vs air and only sharpen the papyrus,
 * leaving air clean.
 *
 * Papyrus is BRIGHT (high intensity) and STRUCTURED (high local variance); air is
 * DARK and FLAT. We build a soft mask in [0,1] from intensity + local variance,
 * then blend: out = mask*sharpened + (1-mask)*air_value. Soft (not hard) so there
 * are no mask-edge artifacts.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* local mean and variance in a (2r+1)^3 box, via a simple separable-ish pass.
 * Cheap and vectorizable; r small (1-2). */
static void local_stats(const float *restrict v, int nz, int ny, int nx, int r,
                        float *restrict mean, float *restrict var) {
    size_t n = (size_t)nz * ny * nx;
    for (size_t i = 0; i < n; i++) { mean[i] = 0; var[i] = 0; }
    /* brute box stats (r tiny) -- clear and auto-vectorizable inner loop */
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                double s = 0, s2 = 0; int cnt = 0;
                for (int dz = -r; dz <= r; dz++) {
                    int zz = z + dz; if (zz < 0 || zz >= nz) continue;
                    for (int dy = -r; dy <= r; dy++) {
                        int yy = y + dy; if (yy < 0 || yy >= ny) continue;
                        size_t row = ((size_t)zz * ny + yy) * nx;
                        for (int dx = -r; dx <= r; dx++) {
                            int xx = x + dx; if (xx < 0 || xx >= nx) continue;
                            float val = v[row + xx];
                            s += val; s2 += (double)val * val; cnt++;
                        }
                    }
                }
                double m = s / cnt;
                size_t idx = ((size_t)z * ny + y) * nx + x;
                mean[idx] = (float)m;
                var[idx] = (float)(s2 / cnt - m * m);
            }
}

/* smoothstep for soft thresholding */
static inline float smoothstep(float lo, float hi, float x) {
    if (hi <= lo) return x >= hi ? 1.0f : 0.0f;
    float t = (x - lo) / (hi - lo);
    if (t < 0) t = 0; if (t > 1) t = 1;
    return t * t * (3.0f - 2.0f * t);
}

/* Build a soft papyrus mask in [0,1] from a (normalized ~[0,1]) volume.
 *   intensity_lo/hi: smoothstep band on local mean intensity (air is dark)
 *   var_lo/hi: smoothstep band on local variance (air is flat)
 * mask = intensity_term * variance_term. Larger => more "papyrus".
 */
int fy_papyrus_mask(const float *in, float *mask, int nz, int ny, int nx,
                    float intensity_lo, float intensity_hi,
                    float var_lo, float var_hi, int radius) {
    if (radius < 1) radius = 1;
    size_t n = (size_t)nz * ny * nx;
    float *mean = malloc(sizeof(float) * n);
    float *var = malloc(sizeof(float) * n);
    if (!mean || !var) { free(mean); free(var); return 1; }
    local_stats(in, nz, ny, nx, radius, mean, var);
    for (size_t i = 0; i < n; i++) {
        float mi = smoothstep(intensity_lo, intensity_hi, mean[i]);
        float mv = smoothstep(var_lo, var_hi, var[i]);
        mask[i] = mi * mv;
    }
    free(mean); free(var);
    return 0;
}

/* Blend processed and air: out = mask*processed + (1-mask)*air_fill.
 * air_fill<0 -> use the original input's local mean (keeps air looking natural);
 * otherwise a constant (e.g. 0 to zero out air). */
int fy_apply_mask(const float *processed, const float *original, const float *mask,
                  float *out, int nz, int ny, int nx, float air_fill) {
    size_t n = (size_t)nz * ny * nx;
    if (air_fill >= 0.0f) {
        for (size_t i = 0; i < n; i++) {
            float m = mask[i];
            out[i] = m * processed[i] + (1.0f - m) * air_fill;
        }
    } else {
        /* fill air with the (smooth) original so gaps stay natural, not sharpened */
        for (size_t i = 0; i < n; i++) {
            float m = mask[i];
            out[i] = m * processed[i] + (1.0f - m) * original[i];
        }
    }
    return 0;
}

/* Convenience: deconvolve, then keep the sharpening only on papyrus (air stays
 * as the smooth original). One call for the common "sharpen without air noise". */
int fy_deconvolve_masked(const float *in, float *out, int nz, int ny, int nx,
                         const fy_physics *p, double reg,
                         float intensity_thresh, float var_thresh) {
    size_t n = (size_t)nz * ny * nx;
    float *dec = malloc(sizeof(float) * n);
    float *mask = malloc(sizeof(float) * n);
    if (!dec || !mask) { free(dec); free(mask); return 1; }
    fy_deconvolve(in, dec, nz, ny, nx, p, reg);
    /* soft bands centered on the thresholds */
    fy_papyrus_mask(in, mask, nz, ny, nx,
                    intensity_thresh * 0.5f, intensity_thresh * 1.5f,
                    var_thresh * 0.3f, var_thresh * 1.5f, 1);
    /* air filled with the original (smooth) so gaps don't get the deconv noise */
    fy_apply_mask(dec, in, mask, out, nz, ny, nx, -1.0f);
    free(dec); free(mask);
    return 0;
}
