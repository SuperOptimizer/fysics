/* test_fysics.c -- correctness tests for the C kernels. */
#include "fysics.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
                              else { printf("ok:   %s\n", msg); } } while (0)

/* naive DFT for cross-checking the FFT on a small size */
static void naive_dft(const float *re, const float *im, float *ore, float *oim, int n, int sign) {
    for (int k = 0; k < n; k++) {
        double sr = 0, si = 0;
        for (int t = 0; t < n; t++) {
            double a = sign * 2.0 * M_PI * k * t / n;
            double c = cos(a), s = sin(a);
            sr += re[t] * c - im[t] * s;
            si += re[t] * s + im[t] * c;
        }
        ore[k] = (float)sr; oim[k] = (float)si;
    }
}

static void test_fft_vs_dft(void) {
    int n = 16;
    float re[16], im[16], dre[16], dim[16];
    for (int i = 0; i < n; i++) { re[i] = sinf(i * 0.7f) + 0.3f * i; im[i] = 0; }
    float re2[16], im2[16];
    for (int i = 0; i < n; i++) { re2[i] = re[i]; im2[i] = im[i]; }
    naive_dft(re, im, dre, dim, n, -1);
    fy_fft1d(re2, im2, n, -1);
    double maxerr = 0;
    for (int i = 0; i < n; i++) {
        double e = fabs(re2[i] - dre[i]) + fabs(im2[i] - dim[i]);
        if (e > maxerr) maxerr = e;
    }
    CHECK(maxerr < 1e-3, "fft1d matches naive DFT");
}

static void test_fft_roundtrip(void) {
    int nz = 8, ny = 8, nx = 8, n = nz*ny*nx;
    float *re = malloc(sizeof(float)*n), *im = calloc(n, sizeof(float));
    float *orig = malloc(sizeof(float)*n);
    for (int i = 0; i < n; i++) { re[i] = (float)((i*2654435761u) % 1000) / 1000.0f; orig[i] = re[i]; }
    fy_fft3d(re, im, nz, ny, nx, -1);
    fy_fft3d(re, im, nz, ny, nx, +1);
    fy_fft3d_normalize(re, im, nz, ny, nx);
    double maxerr = 0;
    for (int i = 0; i < n; i++) { double e = fabs(re[i] - orig[i]); if (e > maxerr) maxerr = e; }
    CHECK(maxerr < 1e-4, "fft3d forward+inverse round-trips");
    free(re); free(im); free(orig);
}

static void test_paganin_transfer(void) {
    fy_physics p = {1000.0, 78.0, 220.0, 2.4, 1.2, 4.0};
    /* nabu reference values (cycles/voxel grid, px=2.4): computed in python */
    double T0  = fy_paganin_transfer(0.0, &p);
    double Tmid = fy_paganin_transfer(0.1, &p);
    double Thi = fy_paganin_transfer(0.4, &p);
    CHECK(fabs(T0 - 1.0) < 1e-9, "paganin T(0) == 1");
    CHECK(Tmid < T0 && Thi < Tmid, "paganin T decreasing (low-pass)");
    /* exact nabu value at f=0.1 cyc/voxel: 1/(1+1000*L*D*pi*(0.1/2.4)^2) */
    double L = 1.23984199e-3/78.0, D = 220000.0, f = 0.1/2.4;
    double ref = 1.0/(1.0 + 1000.0*L*D*M_PI*f*f);
    CHECK(fabs(Tmid - ref) < 1e-9, "paganin matches nabu formula exactly");
}

static void test_deconvolve_sharpens(void) {
    fy_physics p = {1000.0, 78.0, 220.0, 2.4, 1.2, 4.0};
    int nz = 32, ny = 32, nx = 32, n = nz*ny*nx;
    float *in = malloc(sizeof(float)*n), *out = malloc(sizeof(float)*n);
    /* a smooth ramp + bump (low-freq dominated) */
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        double r = (x-16.0)*(x-16.0)+(y-16.0)*(y-16.0)+(z-16.0)*(z-16.0);
        in[(z*ny+y)*nx+x] = (float)exp(-r/50.0);
    }
    int rc = fy_deconvolve(in, out, nz, ny, nx, &p, 0.05);
    CHECK(rc == 0, "deconvolve returns success");
    /* output finite and shape preserved (values change) */
    int finite = 1, changed = 0;
    for (int i = 0; i < n; i++) { if (!isfinite(out[i])) finite = 0; if (fabsf(out[i]-in[i])>1e-4f) changed = 1; }
    CHECK(finite, "deconvolve output is finite");
    CHECK(changed, "deconvolve modifies the volume");
    free(in); free(out);
}

static void test_halo_reasonable(void) {
    fy_physics p = {1000.0, 78.0, 220.0, 2.4, 1.2, 4.0};
    int h = fy_kernel_halo(&p);
    CHECK(h >= 8 && h <= 96, "kernel halo in [8,96] (local neighborhood)");
    printf("      (halo = %d voxels)\n", h);
}

int main(void) {
    test_fft_vs_dft();
    test_fft_roundtrip();
    test_paganin_transfer();
    test_deconvolve_sharpens();
    test_halo_reasonable();
    printf("\n%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
