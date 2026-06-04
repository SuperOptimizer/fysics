/* fft.c -- self-contained, vectorizable radix-2 complex FFT (pure C, no deps).
 *
 * Iterative in-place Cooley-Tukey. Written for auto-vectorization: split real/imag
 * arrays (SoA), flat loops, no aliasing (restrict), no intrinsics. Compile with
 * -O3 -march=native -ffast-math to let the compiler vectorize the butterflies.
 *
 * Sizes must be powers of two. For 3-D we apply 1-D transforms along each axis.
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

/* in-place 1-D FFT. sign=-1 forward, +1 inverse (no normalization). */
void fy_fft1d(float *restrict re, float *restrict im, int n, int sign) {
    bit_reverse(re, im, n);
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        double ang = sign * 2.0 * M_PI / (double)len;
        float wr_step = (float)cos(ang);
        float wi_step = (float)sin(ang);
        for (int i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (int k = 0; k < half; k++) {
                int a = i + k;
                int b = a + half;
                /* butterfly: this inner loop is the vectorization target */
                float ur = re[a], ui = im[a];
                float vr = re[b] * wr - im[b] * wi;
                float vi = re[b] * wi + im[b] * wr;
                re[a] = ur + vr; im[a] = ui + vi;
                re[b] = ur - vr; im[b] = ui - vi;
                /* advance twiddle */
                float nwr = wr * wr_step - wi * wi_step;
                wi = wr * wi_step + wi * wr_step;
                wr = nwr;
            }
        }
    }
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
