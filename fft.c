/* fft.c -- self-contained, vectorizable complex FFT (pure C, no deps).
 *
 * Iterative in-place Cooley-Tukey radix-2, plus a top-level radix-3 split so
 * sizes 3*2^k are supported directly (a 176-voxel halo'd tile pads to 192, not
 * 256 -- ~2.4x less FFT work). Written for auto-vectorization: split real/imag
 * arrays (SoA), flat loops, no aliasing (restrict), no intrinsics.
 *
 * Twiddle factors are PRECOMPUTED into per-size thread-local tables: the old
 * in-loop rotation recurrence (wr,wi updated every butterfly) was a loop-carried
 * dependency that blocked vectorization of the hot butterfly loop, and it
 * accumulated rounding drift. Tables make the butterfly a clean independent-
 * iteration loop (contiguous loads) and are exact per entry.
 *
 * Supported sizes: 2^k and 3*2^k. Use fy_next_fft_size() to pad.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int fy_is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

int fy_next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* smallest supported FFT size >= n (2^k or 3*2^k) */
int fy_next_fft_size(int n) {
    int p2 = fy_next_pow2(n);
    int p3 = 3 * fy_next_pow2((n + 2) / 3);
    return (p3 >= n && p3 < p2) ? p3 : p2;
}

/* ---- per-size twiddle tables (thread-local, built on demand) ----
 * For a pow2 size n the stages len=2,4,..,n need half=len/2 twiddles each,
 * stored contiguously per stage: total n-1 complex entries. Tables hold the
 * POSITIVE-angle values cos(2pi k/len), sin(2pi k/len); the transform sign is
 * applied as a scalar multiply on the imaginary part in the butterfly. */
static __thread float *tw_re[32] = {0}, *tw_im[32] = {0};

static void twiddle_build(int log2n, int n) {
    float *re = (float *)malloc(sizeof(float) * (size_t)(n - 1));
    float *im = (float *)malloc(sizeof(float) * (size_t)(n - 1));
    size_t off = 0;
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        double step = 2.0 * M_PI / (double)len;
        for (int k = 0; k < half; k++) {
            re[off + k] = (float)cos(step * k);
            im[off + k] = (float)sin(step * k);
        }
        off += half;
    }
    tw_re[log2n] = re; tw_im[log2n] = im;
}

/* bit-reversal permutation (scalar; tiny cost vs butterflies) */
static void bit_reverse(float *restrict re, float *restrict im, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
}

/* in-place 1-D pow2 FFT. sign=-1 forward, +1 inverse (no normalization). */
static void fft1d_pow2(float *restrict re, float *restrict im, int n, int sign) {
    int log2n = 0; while ((1 << log2n) < n) log2n++;
    if (!tw_re[log2n]) twiddle_build(log2n, n);
    const float *restrict tre = tw_re[log2n];
    const float *restrict tim = tw_im[log2n];
    const float sgn = (float)sign;
    bit_reverse(re, im, n);
    size_t off = 0;
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        for (int i = 0; i < n; i += len) {
            float *restrict rea = re + i, *restrict ima = im + i;
            float *restrict reb = rea + half, *restrict imb = ima + half;
            /* butterfly: independent iterations, contiguous loads -> vectorizes */
            for (int k = 0; k < half; k++) {
                float wr = tre[off + k], wi = sgn * tim[off + k];
                float ur = rea[k], ui = ima[k];
                float vr = reb[k] * wr - imb[k] * wi;
                float vi = reb[k] * wi + imb[k] * wr;
                rea[k] = ur + vr; ima[k] = ui + vi;
                reb[k] = ur - vr; imb[k] = ui - vi;
            }
        }
        off += half;
    }
}

/* ---- radix-3 top-level split: n = 3*m with m a power of two ----
 * DIT by residue: X[k], X[k+m], X[k+2m] from the three m-point FFTs of
 * x[3j], x[3j+1], x[3j+2], combined with w_n^k and the cube roots of unity.
 * Combine twiddles w_n^k (k<m) are cached per (n) in a thread-local table. */
static __thread float *tw3_re = NULL, *tw3_im = NULL;
static __thread int tw3_n = 0;
static __thread float *r3_scratch = NULL;
static __thread int r3_cap = 0;

static void fft1d_radix3(float *restrict re, float *restrict im, int n, int sign) {
    int m = n / 3;
    if (tw3_n != n) {
        free(tw3_re); free(tw3_im);
        tw3_re = (float *)malloc(sizeof(float) * (size_t)m * 2);
        tw3_im = (float *)malloc(sizeof(float) * (size_t)m * 2);
        double step = 2.0 * M_PI / (double)n;
        for (int k = 0; k < m; k++) {            /* w^k and w^2k, positive angle */
            tw3_re[k]     = (float)cos(step * k);
            tw3_im[k]     = (float)sin(step * k);
            tw3_re[m + k] = (float)cos(step * 2 * k);
            tw3_im[m + k] = (float)sin(step * 2 * k);
        }
        tw3_n = n;
    }
    if (r3_cap < 6 * m) {
        free(r3_scratch);
        r3_scratch = (float *)malloc(sizeof(float) * (size_t)6 * m);
        r3_cap = 6 * m;
    }
    float *ar = r3_scratch,        *ai = ar + m;
    float *br = ai + m,            *bi = br + m;
    float *cr = bi + m,            *ci = cr + m;
    for (int j = 0; j < m; j++) {
        ar[j] = re[3 * j];     ai[j] = im[3 * j];
        br[j] = re[3 * j + 1]; bi[j] = im[3 * j + 1];
        cr[j] = re[3 * j + 2]; ci[j] = im[3 * j + 2];
    }
    fft1d_pow2(ar, ai, m, sign);
    fft1d_pow2(br, bi, m, sign);
    fft1d_pow2(cr, ci, m, sign);
    const float sgn = (float)sign;
    const float u_re = -0.5f;                          /* e^(sign*2pi i/3) */
    const float u_im = sgn * 0.86602540378443864676f;
    for (int k = 0; k < m; k++) {
        float w1r = tw3_re[k],     w1i = sgn * tw3_im[k];
        float w2r = tw3_re[m + k], w2i = sgn * tw3_im[m + k];
        float t1r = br[k] * w1r - bi[k] * w1i, t1i = br[k] * w1i + bi[k] * w1r;
        float t2r = cr[k] * w2r - ci[k] * w2i, t2i = cr[k] * w2i + ci[k] * w2r;
        float a0r = ar[k], a0i = ai[k];
        re[k] = a0r + t1r + t2r;
        im[k] = a0i + t1i + t2i;
        /* u*t1 + u^2*t2 and u^2*t1 + u*t2 (u^2 = conj(u) for unit cube root) */
        float ut1r = u_re * t1r - u_im * t1i, ut1i = u_re * t1i + u_im * t1r;
        float ut2r = u_re * t2r + u_im * t2i, ut2i = u_re * t2i - u_im * t2r;
        re[k + m] = a0r + ut1r + ut2r;
        im[k + m] = a0i + ut1i + ut2i;
        float vt1r = u_re * t1r + u_im * t1i, vt1i = u_re * t1i - u_im * t1r;
        float vt2r = u_re * t2r - u_im * t2i, vt2i = u_re * t2i + u_im * t2r;
        re[k + 2 * m] = a0r + vt1r + vt2r;
        im[k + 2 * m] = a0i + vt1i + vt2i;
    }
}

/* in-place 1-D FFT. sign=-1 forward, +1 inverse (no normalization).
 * n must be 2^k or 3*2^k (see fy_next_fft_size). */
void fy_fft1d(float *restrict re, float *restrict im, int n, int sign) {
    if (n % 3 == 0 && fy_is_pow2(n / 3)) fft1d_radix3(re, im, n, sign);
    else fft1d_pow2(re, im, n, sign);
}

/* 3-D FFT over a (nz,ny,nx) split-complex volume, in place. sign as above.
 * Transforms each axis in turn. Strided 1-D transforms gathered into a scratch
 * line so the FFT inner loop stays contiguous (vectorizable). */
void fy_fft3d(float *restrict re, float *restrict im,
              int nz, int ny, int nx, int sign) {
    int maxn = nx > ny ? (nx > nz ? nx : nz) : (ny > nz ? ny : nz);
    float *lr = (float *)malloc(sizeof(float) * (size_t)maxn);
    float *li = (float *)malloc(sizeof(float) * (size_t)maxn);

    /* X axis: lines are contiguous -- transform directly */
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++) {
            size_t off = ((size_t)z * ny + y) * nx;
            fy_fft1d(re + off, im + off, nx, sign);
        }

    /* Y axis: stride = nx */
    for (int z = 0; z < nz; z++)
        for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) {
                size_t off = ((size_t)z * ny + y) * nx + x;
                lr[y] = re[off]; li[y] = im[off];
            }
            fy_fft1d(lr, li, ny, sign);
            for (int y = 0; y < ny; y++) {
                size_t off = ((size_t)z * ny + y) * nx + x;
                re[off] = lr[y]; im[off] = li[y];
            }
        }

    /* Z axis: stride = ny*nx */
    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++) {
            for (int z = 0; z < nz; z++) {
                size_t off = ((size_t)z * ny + y) * nx + x;
                lr[z] = re[off]; li[z] = im[off];
            }
            fy_fft1d(lr, li, nz, sign);
            for (int z = 0; z < nz; z++) {
                size_t off = ((size_t)z * ny + y) * nx + x;
                re[off] = lr[z]; im[off] = li[z];
            }
        }

    free(lr); free(li);
}

/* normalize after an inverse 3-D FFT (divide by total N) */
void fy_fft3d_normalize(float *restrict re, float *restrict im,
                        int nz, int ny, int nx) {
    size_t n = (size_t)nz * ny * nx;
    float inv = 1.0f / (float)n;
    for (size_t i = 0; i < n; i++) { re[i] *= inv; im[i] *= inv; }
}
