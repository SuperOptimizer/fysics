/* bench_fysics.c -- timing for the deconvolution kernel at viewer-relevant sizes. */
#include "fysics.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static void bench(int nz, int ny, int nx) {
    fy_physics p = {1000.0, 78.0, 220.0, 2.4, 1.2, 4.0};
    size_t n = (size_t)nz * ny * nx;
    float *in = malloc(sizeof(float) * n), *out = malloc(sizeof(float) * n);
    for (size_t i = 0; i < n; i++) in[i] = (float)((i * 2654435761u) % 1000) / 1000.0f;
    fy_deconvolve(in, out, nz, ny, nx, &p, 0.05);   /* warm */
    double t0 = now_ms();
    int reps = 5;
    for (int r = 0; r < reps; r++) fy_deconvolve(in, out, nz, ny, nx, &p, 0.05);
    double dt = (now_ms() - t0) / reps;
    printf("  deconv %3dx%3dx%3d : %7.1f ms  (%.1f Mvox/s)\n",
           nz, ny, nx, dt, n / 1e6 / (dt / 1000.0));
    free(in); free(out);
}

int main(void) {
    printf("fysics deconvolution benchmark (CPU, -Ofast -march=native)\n");
    bench(8, 512, 512);     /* a viewer slab */
    bench(64, 256, 256);    /* a viewer sub-volume */
    bench(128, 128, 128);
    bench(256, 256, 256);   /* a batch tile */
    return 0;
}
