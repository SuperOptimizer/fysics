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

fy_recipe fy_recipe_default(void) {
    fy_recipe r;
    r.deconv_reg = 0.02; r.air_thresh = 0.25f; r.denoise_bilateral = 0.0;
    r.do_glcae = 1; r.glcae_clip = 2.0f;
    return r;
}

fy_recipe fy_recipe_ink(void) {
    /* INK-GRADE: ink is a subtle, near-threshold signal (carbon-on-carbon has low
     * attenuation contrast), so we bias toward PRESERVING fine high-frequency
     * detail rather than smoothing it away -- deconv to recover resolution, no
     * heavy denoise, no aggressive contrast stretch (let the downstream ink model
     * do its own per-fragment normalization). Denoise is left OFF by default but
     * the field is available if ablation shows it helps.
     *
     * NOTE: the precise ink-grade recipe for BM18/ESRF data is UNVALIDATED -- the
     * "crackle morphology" guidance comes from earlier non-BM18 (Diamond/2023)
     * scans, and there is no published controlled ablation of denoise/contrast vs
     * ink-model F0.5 on BM18 volumes. Treat these as sensible defaults to be tuned
     * empirically against labeled BM18 ink data, not as hard rules. */
    fy_recipe r;
    r.deconv_reg = 0.02;            /* recover resolution */
    r.air_thresh = 0.25f;          /* mask air (cleaning gaps shouldn't hurt ink) */
    r.denoise_bilateral = 0.0;     /* off by default -- ablate before enabling */
    r.do_glcae = 0;                /* no contrast stretch -- model normalizes itself */
    r.glcae_clip = 0.0f;
    return r;
}

fy_recipe fy_recipe_segment(void) {
    /* clean + sharp boundaries, denoise on, no contrast stretch */
    fy_recipe r;
    r.deconv_reg = 0.025; r.air_thresh = 0.25f; r.denoise_bilateral = 0.12;
    r.do_glcae = 0; r.glcae_clip = 2.0f;
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
