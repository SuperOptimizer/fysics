/* fy_restretch.c -- GLOBAL final tone stretch for a raw-u8 zarr (level 0/), in place.
 * Two passes over the chunk files: (1) accumulate the INNER histogram (excl masked-0
 * and clipped-255), find [p_lo, p_hi]; (2) map material [p_lo,p_hi] -> [1,255] (air 0
 * stays 0). Clip is controlled by construction (~ (100-hi_pct)% high, lo_pct% low).
 *
 *   fy_restretch <zarr_root> [lo_pct hi_pct]    (default 0.5 99.5)
 *
 * This is the "stretch to full range" step that follows recenter+MUSICA: it uses the
 * MUSICA-OUTPUT distribution, so it reclaims the range without re-amplifying clip.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    double lo_pct = argc >= 4 ? atof(argv[2]) : 0.5;
    double hi_pct = argc >= 4 ? atof(argv[3]) : 99.5;
    int out_lo = argc >= 6 ? atoi(argv[4]) : 1;     /* output floor for material */
    int out_hi = argc >= 6 ? atoi(argv[5]) : 255;   /* output ceil  (leave <255 for highlight headroom) */
    char path[2048]; snprintf(path, sizeof path, "%s/0/.zarray", root);
    FILE *zf = fopen(path, "rb"); if (!zf) { fprintf(stderr, "no %s\n", path); return 1; }
    char zb[4096]; size_t zn = fread(zb, 1, sizeof zb - 1, zf); zb[zn] = 0; fclose(zf);
    long SZ = jint(zb, "\"shape\"", 0), SY = jint(zb, "\"shape\"", 1), SX = jint(zb, "\"shape\"", 2);
    long CZ = jint(zb, "\"chunks\"", 0), CY = jint(zb, "\"chunks\"", 1), CX = jint(zb, "\"chunks\"", 2);
    if (SZ <= 0 || CZ <= 0) { fprintf(stderr, "bad .zarray\n"); return 1; }
    long ncz = (SZ+CZ-1)/CZ, ncy = (SY+CY-1)/CY, ncx = (SX+CX-1)/CX;
    unsigned char *cb = malloc((size_t)CZ*CY*CX);

    /* pass 1: inner histogram */
    long hist[256] = {0};
    for (long cz = 0; cz < ncz; cz++) for (long cy = 0; cy < ncy; cy++) for (long cx = 0; cx < ncx; cx++) {
        long ez=CZ<SZ-cz*CZ?CZ:SZ-cz*CZ, ey=CY<SY-cy*CY?CY:SY-cy*CY, ex=CX<SX-cx*CX?CX:SX-cx*CX;
        size_t nv=(size_t)ez*ey*ex;
        snprintf(path, sizeof path, "%s/0/%ld/%ld/%ld", root, cz, cy, cx);
        FILE *f = fopen(path, "rb"); if (!f) continue;
        size_t got = fread(cb, 1, nv, f); fclose(f);
        for (size_t i = 0; i < got; i++) hist[cb[i]]++;
    }
    long tot = 0; for (int i = 1; i <= 254; i++) tot += hist[i];
    if (tot <= 0) { fprintf(stderr, "no material\n"); return 1; }
    int lo = 1, hi = 254; long a = 0, tl = (long)(lo_pct/100.0*tot), th = (long)(hi_pct/100.0*tot);
    for (int i = 1; i <= 254; i++) { a += hist[i]; if (a >= tl) { lo = i; break; } }
    a = 0; for (int i = 1; i <= 254; i++) { a += hist[i]; if (a >= th) { hi = i; break; } }
    if (hi <= lo) hi = lo + 1;
    printf("fy_restretch %s: material [p%.1f=%d, p%.1f=%d] -> [%d,%d]  (air 0 -> 0)\n",
           root, lo_pct, lo, hi_pct, hi, out_lo, out_hi);

    /* build LUT u8->u8: 0->0; [lo,hi]->[out_lo,out_hi] linear, clamped */
    unsigned char lut[256];
    lut[0] = 0;
    for (int v = 1; v < 256; v++) {
        double o = out_lo + (double)(v - lo) / (double)(hi - lo) * (double)(out_hi - out_lo);
        lut[v] = (unsigned char)(o < out_lo ? out_lo : o > out_hi ? out_hi : o + 0.5);
    }

    /* pass 2: apply LUT in place */
    long nwrite = 0;
    for (long cz = 0; cz < ncz; cz++) for (long cy = 0; cy < ncy; cy++) for (long cx = 0; cx < ncx; cx++) {
        long ez=CZ<SZ-cz*CZ?CZ:SZ-cz*CZ, ey=CY<SY-cy*CY?CY:SY-cy*CY, ex=CX<SX-cx*CX?CX:SX-cx*CX;
        size_t nv=(size_t)ez*ey*ex;
        snprintf(path, sizeof path, "%s/0/%ld/%ld/%ld", root, cz, cy, cx);
        FILE *f = fopen(path, "rb+"); if (!f) continue;
        size_t got = fread(cb, 1, nv, f);
        for (size_t i = 0; i < got; i++) cb[i] = lut[cb[i]];
        fseek(f, 0, SEEK_SET); fwrite(cb, 1, got, f); fclose(f);
        nwrite++;
    }
    free(cb);
    printf("  rewrote %ld chunks\n", nwrite);
    return 0;
}
