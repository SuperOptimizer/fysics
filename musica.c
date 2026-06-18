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
 * boosts faint detail; ~0.7), core = noise coring threshold (0 disables).
 *
 * MASK/CLIP AWARE: only voxels strictly in (0,1) are "real". Masked (==0) and
 * clipped (==1, i.e. u8 255) voxels are EXCLUDED from the pyramid blur via
 * NORMALIZED CONVOLUTION -- each blurred base is sum(value*weight)/sum(weight)
 * over valid neighbours only -- so black/white never bleeds into real pixels near
 * a mask/clip boundary (no edge halos). Those voxels pass through UNMODIFIED.
 * Net: the enhancement counts and modifies only the 1..254 range. */
int fy_musica2d(const float *in, float *out, int ny, int nx,
                int levels, float p, float core) {
    if (levels < 1) levels = 4;
    if (p <= 0) p = 0.7f;
    size_t n = (size_t)ny * nx;
    float *V    = malloc(sizeof(float) * n);   /* value  (0 where invalid)        */
    float *W    = malloc(sizeof(float) * n);   /* weight (1 valid, 0 masked/clip) */
    float *base = malloc(sizeof(float) * n);   /* current normalized base         */
    float *Vn   = malloc(sizeof(float) * n);
    float *Wn   = malloc(sizeof(float) * n);
    float *tmp  = malloc(sizeof(float) * n);
    float *acc  = malloc(sizeof(float) * n);   /* sum of gained details           */
    if (!V||!W||!base||!Vn||!Wn||!tmp||!acc) { free(V);free(W);free(base);free(Vn);free(Wn);free(tmp);free(acc); return 1; }
    for (size_t i = 0; i < n; i++) {
        int valid = (in[i] > 0.0f && in[i] < 1.0f);
        W[i] = valid ? 1.0f : 0.0f;
        V[i] = valid ? in[i] : 0.0f;
        base[i] = in[i];
        acc[i] = 0.0f;
    }
    memcpy(Vn, V, sizeof(float) * n);
    memcpy(Wn, W, sizeof(float) * n);
    float a = 0.5f;  /* gain normalization point */
    for (int l = 0; l < levels; l++) {
        int iters = 1 << l;                     /* cumulative octave blur (1,3,7,15 passes) */
        for (int it = 0; it < iters; it++) {
            blur5(Vn, tmp, ny, nx); memcpy(Vn, tmp, sizeof(float) * n);
            blur5(Wn, tmp, ny, nx); memcpy(Wn, tmp, sizeof(float) * n);
        }
        /* coarser base = blur(value)/blur(weight): mask-aware, 0/255 contribute nothing */
        for (size_t i = 0; i < n; i++) {
            float nb = (Wn[i] > 1e-6f) ? Vn[i] / Wn[i] : base[i];
            acc[i] += musica_gain(base[i] - nb, a, p, core);
            base[i] = nb;
        }
    }
    /* masked (0) and clipped (1) pass through unmodified; real pixels rebuilt */
    for (size_t i = 0; i < n; i++) {
        if (!(in[i] > 0.0f && in[i] < 1.0f)) { out[i] = in[i]; continue; }
        float v = base[i] + acc[i];
        out[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
    free(V); free(W); free(base); free(Vn); free(Wn); free(tmp); free(acc);
    return 0;
}
