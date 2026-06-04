/* musica.c -- MUSICA multiscale contrast enhancement (Vuylsteke & Schoeters, Agfa).
 *
 * The medical-radiography workhorse for lifting faint low-contrast detail in noisy
 * grayscale transmission X-ray -- exactly the scroll problem. Better than CLAHE/
 * GLCAE here: multiscale (not tile-based -> no tile artifacts, no halos), and it
 * doesn't penalize RARE faint features the way histogram-equalization does.
 *
 * Method: build a Laplacian pyramid; apply a sublinear gain to each level's detail
 * coefficients (amplifies small/low-contrast coeffs more than large ones); rebuild.
 *   y = a * sign(x) * |x/a|^p,   p < 1   (per-level, on Laplacian coeffs)
 * with optional per-level soft "coring" (suppress |x| below k*noise to avoid
 * amplifying noise). Operates on a 2D slice in [0,1]; cheap, pure-C, streamable
 * (apron ~2^levels).
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* 5-tap binomial (Gaussian) kernel: 1 4 6 4 1 / 16 */
static void blur5(const float *restrict in, float *restrict out, int ny, int nx) {
    const float k0 = 6.f/16, k1 = 4.f/16, k2 = 1.f/16;
    float *tmp = malloc(sizeof(float) * (size_t)ny * nx);
    /* horizontal */
    for (int y = 0; y < ny; y++) {
        const float *r = in + (size_t)y * nx; float *o = tmp + (size_t)y * nx;
        for (int x = 0; x < nx; x++) {
            int xm2=x-2<0?0:x-2, xm1=x-1<0?0:x-1, xp1=x+1>=nx?nx-1:x+1, xp2=x+2>=nx?nx-1:x+2;
            o[x] = k2*r[xm2]+k1*r[xm1]+k0*r[x]+k1*r[xp1]+k2*r[xp2];
        }
    }
    /* vertical */
    for (int y = 0; y < ny; y++) {
        int ym2=y-2<0?0:y-2, ym1=y-1<0?0:y-1, yp1=y+1>=ny?ny-1:y+1, yp2=y+2>=ny?ny-1:y+2;
        for (int x = 0; x < nx; x++)
            out[(size_t)y*nx+x] = k2*tmp[(size_t)ym2*nx+x]+k1*tmp[(size_t)ym1*nx+x]
                                + k0*tmp[(size_t)y*nx+x]+k1*tmp[(size_t)yp1*nx+x]+k2*tmp[(size_t)yp2*nx+x];
    }
    free(tmp);
}

/* sublinear gain on a detail coefficient */
static inline float musica_gain(float x, float a, float p, float core) {
    float ax = fabsf(x);
    if (ax < core) return x * (ax / (core + 1e-8f));  /* soft-core small (noise) coeffs */
    float s = x < 0 ? -1.f : 1.f;
    return a * s * powf(ax / a, p);
}

/* MUSICA on a single [0,1] slice. levels = pyramid depth, p = gain exponent (<1
 * boosts faint detail; ~0.7), core = noise coring threshold (0 disables). */
int fy_musica2d(const float *in, float *out, int ny, int nx,
                int levels, float p, float core) {
    if (levels < 1) levels = 4;
    if (p <= 0) p = 0.7f;
    size_t n = (size_t)ny * nx;
    /* We do a same-size (non-decimated) Laplacian pyramid for simplicity and to
     * avoid resampling artifacts: residual at each scale = img - blur(img), then
     * blur becomes the next coarser base. Gain each residual, sum back. */
    float *cur  = malloc(sizeof(float) * n);   /* current (finer) base */
    float *next = malloc(sizeof(float) * n);   /* blurred (coarser) base */
    float *acc  = malloc(sizeof(float) * n);   /* sum of gained details */
    if (!cur || !next || !acc) { free(cur); free(next); free(acc); return 1; }
    memcpy(cur, in, sizeof(float) * n);
    memset(acc, 0, sizeof(float) * n);
    float a = 0.5f;  /* gain normalization point */
    for (int l = 0; l < levels; l++) {
        /* coarser base = cur blurred 2^l times (octave-ish scale increase) */
        memcpy(next, cur, sizeof(float) * n);
        float *tmp = malloc(sizeof(float) * n);
        int iters = 1 << l;                     /* 1,2,4,8,... blur passes */
        for (int it = 0; it < iters; it++) { blur5(next, tmp, ny, nx); memcpy(next, tmp, sizeof(float)*n); }
        free(tmp);
        /* detail at this scale = cur - next; gain and accumulate */
        for (size_t i = 0; i < n; i++)
            acc[i] += musica_gain(cur[i] - next[i], a, p, core);
        /* descend: coarser base becomes the next level's input */
        memcpy(cur, next, sizeof(float) * n);
    }
    /* result = coarsest residual base (low-freq, untouched) + gained details.
     * ZERO-AWARE: pixels that were exactly 0 in the input (masked air) stay 0 --
     * we never contrast-enhance the air, and the black gaps are preserved. */
    for (size_t i = 0; i < n; i++) {
        if (in[i] == 0.0f) { out[i] = 0.0f; continue; }
        float v = cur[i] + acc[i];
        out[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
    free(cur); free(next); free(acc);
    return 0;
}
