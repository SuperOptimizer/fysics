/* fy_stats.c -- fast intensity-distribution report for a raw-u8 zarr (level 0/).
 * Single O(N) histogram pass over the chunk files; all stats from the 256-bin
 * histogram. Masked (0) and clipped (255) are reported separately and EXCLUDED
 * from the "material" stats. Optionally previews a recenter+stretch remap.
 *
 *   fy_stats <zarr_root> [lo_pct hi_pct]
 * With lo_pct/hi_pct it previews mapping material [p_lo,p_hi] -> [0,255]
 * (a recenter+stretch): reports where the median lands and the new clip %.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static long jint(const char *s, const char *key, int idx) {
    const char *p = strstr(s, key); if (!p) return -1; p += strlen(key);
    for (int i = 0; i <= idx; i++) {
        while (*p && (*p < '0' || *p > '9') && *p != '-') p++;
        if (i < idx) while (*p >= '0' && *p <= '9') p++;
    }
    return strtol(p, NULL, 10);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <zarr_root> [lo_pct hi_pct]\n", argv[0]); return 2; }
    const char *root = argv[1];
    char path[2048]; snprintf(path, sizeof path, "%s/0/.zarray", root);
    FILE *zf = fopen(path, "rb"); if (!zf) { fprintf(stderr, "no %s\n", path); return 1; }
    char buf[4096]; size_t zn = fread(buf, 1, sizeof buf - 1, zf); buf[zn] = 0; fclose(zf);
    long SZ = jint(buf, "\"shape\"", 0), SY = jint(buf, "\"shape\"", 1), SX = jint(buf, "\"shape\"", 2);
    long CZ = jint(buf, "\"chunks\"", 0), CY = jint(buf, "\"chunks\"", 1), CX = jint(buf, "\"chunks\"", 2);
    if (SZ <= 0 || CZ <= 0) { fprintf(stderr, "bad .zarray\n"); return 1; }

    long hist[256] = {0};
    long ncz = (SZ + CZ - 1) / CZ, ncy = (SY + CY - 1) / CY, ncx = (SX + CX - 1) / CX;
    unsigned char *cb = malloc((size_t)CZ * CY * CX);
    for (long cz = 0; cz < ncz; cz++)
        for (long cy = 0; cy < ncy; cy++)
            for (long cx = 0; cx < ncx; cx++) {
                long ez = CZ < SZ - cz*CZ ? CZ : SZ - cz*CZ;
                long ey = CY < SY - cy*CY ? CY : SY - cy*CY;
                long ex = CX < SX - cx*CX ? CX : SX - cx*CX;
                size_t nvox = (size_t)ez * ey * ex;
                snprintf(path, sizeof path, "%s/0/%ld/%ld/%ld", root, cz, cy, cx);
                FILE *f = fopen(path, "rb");
                if (!f) { hist[0] += (long)nvox; continue; }   /* missing chunk = fill 0 */
                size_t got = fread(cb, 1, nvox, f); fclose(f);
                for (size_t i = 0; i < got; i++) hist[cb[i]]++;
                if (got < nvox) hist[0] += (long)(nvox - got);
            }
    free(cb);

    double N = 0; for (int i = 0; i < 256; i++) N += hist[i];
    long n0 = hist[0], n255 = hist[255];
    /* material = bins 1..254 */
    double M = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 1; i <= 254; i++) { M += hist[i]; s1 += (double)i*hist[i]; s2 += (double)i*i*hist[i]; s3 += (double)i*i*i*hist[i]; }
    double mean = s1 / M, var = s2 / M - mean*mean, sd = sqrt(var > 0 ? var : 0);
    double skew = (s3/M - 3*mean*var - mean*mean*mean) / (sd*sd*sd + 1e-12);
    /* percentile over material */
    long pct_bin(double q) {
        double t = q/100.0 * M, a = 0;
        for (int i = 1; i <= 254; i++) { a += hist[i]; if (a >= t) return i; }
        return 254;
    }
    long p1 = pct_bin(1), p50 = pct_bin(50), p99 = pct_bin(99), p995 = pct_bin(99.5), p05 = pct_bin(0.5);

    printf("zarr %s  (%ldx%ldx%ld, %.3g voxels)\n", root, SZ, SY, SX, N);
    printf("  masked(==0): %.1f%%   clipped(==255): %.2f%%   material(1..254): %.1f%%\n",
           100*n0/N, 100*n255/N, 100*M/N);
    printf("  material: mean %.1f  median %ld  std %.1f  skew %+.2f\n", mean, p50, sd, skew);
    printf("  percentiles: p0.5 %ld  p1 %ld  p50 %ld  p99 %ld  p99.5 %ld\n", p05, p1, p50, p99, p995);

    if (argc >= 4) {
        double lo = pct_bin(atof(argv[2])), hi = pct_bin(atof(argv[3]));
        double sc = 255.0 / (hi - lo);
        /* preview clip + median after mapping material [lo,hi] -> [0,255] */
        long lo_clip = 0, hi_clip = 0; double newmean = 0;
        for (int i = 1; i <= 254; i++) {
            double o = (i - lo) * sc;
            if (o <= 0) lo_clip += hist[i]; else if (o >= 255) hi_clip += hist[i];
            newmean += (o < 0 ? 0 : o > 255 ? 255 : o) * hist[i];
        }
        double newmed = (p50 - lo) * sc; if (newmed < 0) newmed = 0; if (newmed > 255) newmed = 255;
        printf("  REMAP material [%.0f,%.0f]->[0,255] (recenter+stretch):\n", lo, hi);
        printf("    new median %.0f  new mean %.0f  -> low-clip %.2f%%  high-clip %.2f%%\n",
               newmed, newmean/M, 100.0*lo_clip/N, 100.0*hi_clip/N);
    }
    return 0;
}
