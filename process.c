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
    r.deconv_reg = 0.015;          /* measured knee (reg sweep, 90 cubes) */
    r.auto_deltabeta = 1;          /* partial inversion on fine volumes (measured) */
    r.air_thresh = 0.15f;          /* mask air (lower = keep more dim papyrus) */
    r.air_fill = 0.0f;             /* ZERO out the air (black gaps), before contrast */
    r.denoise_bilateral = 0.05;    /* fallback guided eps if auto_denoise is off */
    r.auto_denoise = 1;            /* set eps from measured per-volume noise (recommended) */
    r.do_musica = 1;               /* MUSICA contrast ON, gentle (see musica_p) */
    r.musica_p = 0.9f;             /* gentle (closer to 1 = milder boost) */
    r.do_glcae = 0;                /* legacy GLCAE off; prefer MUSICA */
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

    /* 1. mask computed on the RAW data (cleanest air/papyrus histogram valley) */
    int use_mask = (r->air_thresh > 0.0f);
    if (use_mask) {
        mask = malloc(sizeof(float) * n);
        if (!mask) { free(buf); return 1; }
        float t = r->air_thresh;
        fy_papyrus_mask(in, mask, nz, ny, nx, t * 0.6f, t * 1.4f, 0, 0, 1);
    }

    /* 2. deconvolve. This RESTORES high-frequency CONTRAST that the Paganin low-pass
     * suppressed (a sharpness/contrast restoration -- it does NOT recover resolution;
     * FRC confirms no SNR-limited resolution gain). auto_deltabeta uses partial
     * inversion on fine volumes (full delta_beta over-inverts them; measured). */
    if (r->deconv_reg > 0.0) {
        fy_physics pp = *p;
        if (r->auto_deltabeta) pp.delta_beta *= fy_auto_deltabeta_scale(p);
        fy_deconvolve(cur, buf, nz, ny, nx, &pp, r->deconv_reg);
        memcpy(out, buf, sizeof(float) * n);
        cur = out;
    } else {
        memcpy(out, in, sizeof(float) * n);
        cur = out;
    }

    /* 3. optional denoise (guided filter -- fast O(N), no gradient reversal).
     * auto_denoise sets eps from the volume's MEASURED noise (the level varies
     * 1.5-3.3x scroll-to-scroll, so it must be estimated per-volume, not hardcoded).
     * ORDER NOTE (measured): deblur->denoise vs denoise->deblur are ~equivalent with
     * the self-calibrated eps, EXCEPT on high-amplification fine volumes (~1.1um, large
     * unsharp_sigma) where denoise-FIRST wins ~12-13%. Not worth special-casing the
     * default; revisit if such scans dominate. fy_denoise_quality is the cleaner tier. */
    {
        double eps = r->denoise_bilateral * r->denoise_bilateral;
        int do_denoise = (r->denoise_bilateral > 0.0);
        if (r->auto_denoise) {
            fy_noise_model nm;
            if (fy_estimate_noise(cur, nz, ny, nx, 5, 10.0, 0.4, &nm) == 0) {
                eps = fy_guided_eps_for_noise(nm.noise_ref);
                do_denoise = 1;
            }
        }
        if (do_denoise) {
            fy_guided_denoise(cur, buf, nz, ny, nx, 2, eps);
            memcpy(out, buf, sizeof(float) * n);
            cur = out;
        }
    }

    /* 4. ZERO OUT THE AIR (before contrast) -- we don't want to contrast-enhance
     * empty gaps. air_fill: 0 = black (default), <0 = keep smooth original. */
    if (use_mask) {
        float fill = r->air_fill;
        fy_apply_mask(cur, in, mask, buf, nz, ny, nx, fill < 0 ? -1.0f : fill);
        memcpy(out, buf, sizeof(float) * n);
        cur = out;
    }

    /* 5. CONTRAST ENHANCE (after air is zeroed -- operates on papyrus + black gaps).
     * MUSICA preferred (multiscale, no tile/halo, better for faint detail). The
     * zeroed air is flat -> no detail to amplify (coring leaves it alone). */
    if (r->do_musica) {
        float pexp = r->musica_p > 0 ? r->musica_p : 0.8f;
        for (int z = 0; z < nz; z++)
            fy_musica2d(cur + (size_t)z*ny*nx, buf + (size_t)z*ny*nx, ny, nx, 4, pexp, 0.0f);
        memcpy(out, buf, sizeof(float) * n);
        cur = out;
    } else if (r->do_glcae) {
        for (int z = 0; z < nz; z++)
            fy_glcae2d(cur + (size_t)z*ny*nx, buf + (size_t)z*ny*nx, ny, nx, 8, r->glcae_clip);
        memcpy(out, buf, sizeof(float) * n);
        cur = out;
    }

    /* 6. re-zero air after contrast (contrast may have lifted the black slightly) */
    if (use_mask && (r->air_fill >= 0)) {
        float fill = r->air_fill;
        fy_apply_mask(cur, in, mask, buf, nz, ny, nx, fill);
        memcpy(out, buf, sizeof(float) * n);
    }

    free(buf); if (mask) free(mask);
    return 0;
}
