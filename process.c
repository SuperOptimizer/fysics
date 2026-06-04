/* process.c -- the one-call validated recipe (+ task profiles).
 *
 * Recipe order (validated visually on real PHerc data):
 *   1. mask air on the RAW data (raw has the cleanest air/papyrus histogram valley;
 *      deconv ringing muddies it, so we detect air BEFORE deconv)
 *   2. deconvolve (Paganin inverse)
 *   3. re-apply the raw mask -> air gaps stay clean (kills deconv ring halos there)
 *   4. optional bilateral denoise
 *   5. optional GLCAE contrast (per XY slice)
 *
 * Task profiles encode that different downstream tasks want different processing:
 *   - INK: aggressive contrast, KEEP near-noise texture (ink is a faint cue) -> low
 *     reg, NO denoise, GLCAE on.
 *   - SEGMENT: clean + sharp boundaries, low noise -> moderate reg, denoise on,
 *     GLCAE off (contrast not the goal).
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ONE pipeline. Punchy default: aggressive deconv to recover resolution, air mask
 * to keep gaps clean, light guided denoise to tame the deconv noise. GLCAE contrast
 * is OFF by default (it's the most opinionated step and may fight downstream model
 * normalization) -- flip do_glcae=1 to add it for human viewing. We deliberately do
 * NOT pre-split into task "grades" yet; settle one good pipeline first, specialize
 * later only if ablation shows it's needed. */
fy_recipe fy_recipe_default(void) {
    fy_recipe r;
    r.deconv_reg = 0.015;          /* punchy -- recover resolution */
    r.air_thresh = 0.25f;          /* mask air (histogram-valley threshold) */
    r.denoise_bilateral = 0.05;    /* light guided denoise (eps), tames deconv noise */
    r.do_glcae = 0;                /* contrast enhancement off by default */
    r.glcae_clip = 2.0f;
    return r;
}

int fy_process(const float *in, float *out, int nz, int ny, int nx,
               const fy_physics *p, const fy_recipe *r) {
    size_t n = (size_t)nz * ny * nx;
    float *buf = malloc(sizeof(float) * n);
    float *mask = NULL;
    if (!buf) return 1;

    const float *cur = in;

    /* 1. mask on raw */
    int use_mask = (r->air_thresh > 0.0f);
    if (use_mask) {
        mask = malloc(sizeof(float) * n);
        if (!mask) { free(buf); return 1; }
        float t = r->air_thresh;
        fy_papyrus_mask(in, mask, nz, ny, nx, t * 0.6f, t * 1.4f, 0, 0, 1);
    }

    /* 2. deconvolve */
    if (r->deconv_reg > 0.0) {
        fy_deconvolve(cur, buf, nz, ny, nx, p, r->deconv_reg);
        /* 3. re-apply mask: air = smooth original (in), papyrus = deconv (buf) */
        if (use_mask) { fy_apply_mask(buf, in, mask, out, nz, ny, nx, -1.0f); }
        else { memcpy(out, buf, sizeof(float) * n); }
        cur = out;
    } else {
        memcpy(out, in, sizeof(float) * n);
        cur = out;
    }

    /* 4. optional denoise -- GUIDED filter (fast, O(N), no gradient reversal).
     * denoise_bilateral is reused as the guided eps (range); ~7x faster than
     * bilateral and O(1) in radius, the right default for streaming volumes. */
    if (r->denoise_bilateral > 0.0) {
        double eps = r->denoise_bilateral * r->denoise_bilateral;  /* range^2 */
        fy_guided_denoise(cur, buf, nz, ny, nx, 2, eps);
        memcpy(out, buf, sizeof(float) * n);
        cur = out;
    }

    /* 5. optional GLCAE contrast, per XY slice */
    if (r->do_glcae) {
        for (int z = 0; z < nz; z++) {
            const float *sin = cur + (size_t)z * ny * nx;
            float *sout = buf + (size_t)z * ny * nx;
            fy_glcae2d(sin, sout, ny, nx, 8, r->glcae_clip);
        }
        memcpy(out, buf, sizeof(float) * n);
    }

    free(buf); if (mask) free(mask);
    return 0;
}
