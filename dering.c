/* dering.c -- DETECT-then-subtract residual ring removal (streaming, 2-pass).
 *
 * Residual ring artifacts (detector-defect stripes that survive nabu's sinogram
 * dering) are concentric circles centered on the ROTATION AXIS. Measured on real
 * BM18 scroll volumes (PHerc0139 2.4um, 2026-06): rings of 2-4 u8 amplitude,
 * persistent over >200um of z, sitting 200x above the angular-average noise
 * floor. The confounder is the papyrus winding itself, which is locally
 * concentric -- so blind radial high-pass subtraction would eat real structure.
 * This module instead DETECTS rings with an angular-sector
 * consistency vote and subtracts ONLY the detected component:
 *
 *   - accumulate per (z-slab, sector, radius-bin) intensity sums (pass 1)
 *   - per slab+sector: angular-mean radial profile, box high-pass
 *   - a radius is a RING only if every valid sector agrees in sign
 *     (a true ring sits at the SAME radius in all sectors; a spiral wrap
 *     drifts in radius with angle by its pitch and fails the vote --
 *     verified on real data: 0 px sector-to-sector drift for rings)
 *   - ring estimate = median over sectors, clamped; everything else is 0
 *   - subtract ring[slab][rbin(y,x)] per voxel (pass 2), a LOCAL op
 *
 * Center: pass the metadata rotation axis if known; the BM18 recon places it at
 * the slice center, so cy/cx < 0 defaults to (Y-1)/2, (X-1)/2.
 * Memory: nslab * ns * nr * 8 bytes (e.g. 26511^2 x 77k-z volume, slab_z=512,
 * ns=8: ~180 MB) -- the price of per-slab profiles on a 50 TB volume.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int fy_dering_init(fy_dering *d, long Z, long Y, long X,
                   double cy, double cx, int slab_z, int ns) {
    memset(d, 0, sizeof(*d));
    d->Z = Z; d->Y = Y; d->X = X;
    d->cy = cy >= 0 ? cy : (Y - 1) / 2.0;
    d->cx = cx >= 0 ? cx : (X - 1) / 2.0;
    d->slab_z = slab_z > 0 ? slab_z : 512;
    d->nslab = (int)((Z + d->slab_z - 1) / d->slab_z);
    d->ns = ns > 0 ? ns : 8;
    double ry = d->cy > Y - 1 - d->cy ? d->cy : Y - 1 - d->cy;
    double rx = d->cx > X - 1 - d->cx ? d->cx : X - 1 - d->cx;
    d->nr = (int)(sqrt(ry * ry + rx * rx)) + 2;
    size_t nb = (size_t)d->nslab * d->ns * d->nr;
    d->sum = (float *)calloc(nb, sizeof(float));
    d->cnt = (unsigned int *)calloc(nb, sizeof(unsigned int));
    d->ring = (float *)calloc((size_t)d->nslab * d->nr, sizeof(float));
    if (!d->sum || !d->cnt || !d->ring) { fy_dering_free(d); return 1; }
    return 0;
}

void fy_dering_free(fy_dering *d) {
    free(d->sum); free(d->cnt); free(d->ring);
    d->sum = NULL; d->cnt = NULL; d->ring = NULL;
}

/* single-slab scratch clone for per-thread accumulation (merge with
 * fy_dering_merge_tile under the caller's lock). */
int fy_dering_tile_init(fy_dering *t, const fy_dering *d) {
    *t = *d;
    t->nslab = 1;
    size_t nb = (size_t)t->ns * t->nr;
    t->sum = (float *)calloc(nb, sizeof(float));
    t->cnt = (unsigned int *)calloc(nb, sizeof(unsigned int));
    t->ring = NULL;
    if (!t->sum || !t->cnt) { free(t->sum); free(t->cnt); return 1; }
    return 0;
}

void fy_dering_tile_reset(fy_dering *t) {
    size_t nb = (size_t)t->ns * t->nr;
    memset(t->sum, 0, nb * sizeof(float));
    memset(t->cnt, 0, nb * sizeof(unsigned int));
}

/* accumulate a raw-u8 region at global offset (z0,y0,x0) into a SINGLE-slab
 * state (zero/masked voxels skipped). ss = spatial subsample stride in y,x
 * (>=1; the profile only needs a fraction of the voxels to converge). */
void fy_dering_accumulate_u8(fy_dering *t, const unsigned char *buf,
                             long y0, long x0, long dz, long dy, long dx, int ss) {
    if (ss < 1) ss = 1;
    int ns = t->ns, nr = t->nr;
    double inv_sect = ns / (2.0 * M_PI);
    /* radius+sector depend only on (y,x): build the (sector*nr + rbin) plane map ONCE
     * and reuse it across all dz slices (the per-voxel sqrt+atan2 were a profiled
     * pass-1b hotspot). -1 = out of range. */
    long sy = (dy + ss - 1) / ss, sx = (dx + ss - 1) / ss;
    int *map = (int *)malloc(sizeof(int) * (size_t)sy * sx);
    if (!map) return;   /* accumulation is statistical; the min_cnt gates handle a miss */
    for (long yi = 0; yi < sy; yi++) {
        double gy = (double)(y0 + yi * ss) - t->cy, gy2 = gy * gy;
        int *mrow = map + yi * sx;
        for (long xi = 0; xi < sx; xi++) {
            double gx = (double)(x0 + xi * ss) - t->cx;
            int ri = (int)(sqrt(gy2 + gx * gx) + 0.5);
            if (ri >= nr) { mrow[xi] = -1; continue; }
            int q = (int)((atan2(gy, gx) + M_PI) * inv_sect); if (q >= ns) q = ns - 1;
            mrow[xi] = q * nr + ri;
        }
    }
    for (long z = 0; z < dz; z++) {
        const unsigned char *sl = buf + (size_t)z * dy * dx;
        for (long yi = 0; yi < sy; yi++) {
            const unsigned char *row = sl + (size_t)(yi * ss) * dx;
            const int *mrow = map + yi * sx;
            for (long xi = 0; xi < sx; xi++) {
                unsigned char v = row[(size_t)xi * ss];
                if (!v) continue;
                int b = mrow[xi];
                if (b < 0) continue;
                t->sum[b] += v;
                t->cnt[b]++;
            }
        }
    }
    free(map);
}

/* merge a single-slab scratch into global slab `slab` */
void fy_dering_merge_tile(fy_dering *d, int slab, const fy_dering *t) {
    if (slab < 0 || slab >= d->nslab) return;
    size_t nb = (size_t)d->ns * d->nr, off = (size_t)slab * nb;
    for (size_t i = 0; i < nb; i++) { d->sum[off + i] += t->sum[i]; d->cnt[off + i] += t->cnt[i]; }
}

static int cmpf(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

/* finalize: per slab, per-sector angular-mean profiles -> box high-pass ->
 * sector sign-consistency vote -> ring[slab][r] (u8 units).
 *   hp_win: radial high-pass window (vox; ~15 matches measured ring widths)
 *   min_cnt: minimum samples for a (sector,radius) bin to vote
 *   min_amp/max_amp: detection floor / safety clamp (u8 units)
 * Returns the number of detected ring radii (all slabs); sets d->have_rings. */
long fy_dering_finalize(fy_dering *d, int hp_win, unsigned int min_cnt,
                        double min_amp, double max_amp) {
    if (hp_win < 3) hp_win = 15;
    if (min_amp <= 0) min_amp = 0.5;
    if (max_amp <= 0) max_amp = 6.0;
    int ns = d->ns, nr = d->nr, hw = hp_win / 2;
    /* NOTE: explicit validity masks, NOT NaN sentinels -- the library is built with
     * -ffast-math, under which isnan() is compiled away. */
    float *p = (float *)malloc(sizeof(float) * (size_t)ns * nr);          /* mean profiles */
    float *hp = (float *)malloc(sizeof(float) * (size_t)ns * nr);         /* high-passed  */
    unsigned char *vd = (unsigned char *)malloc((size_t)ns * nr);         /* valid masks  */
    float med[64];
    long detected = 0;
    if (!p || !hp || !vd) { free(p); free(hp); free(vd); return 0; }
    for (int s = 0; s < d->nslab; s++) {
        size_t soff = (size_t)s * ns * nr;
        for (int q = 0; q < ns; q++) {
            const float *sum = d->sum + soff + (size_t)q * nr;
            const unsigned int *cnt = d->cnt + soff + (size_t)q * nr;
            float *pq = p + (size_t)q * nr, *hq = hp + (size_t)q * nr;
            unsigned char *vq = vd + (size_t)q * nr;
            for (int r = 0; r < nr; r++) {
                vq[r] = cnt[r] >= min_cnt;
                pq[r] = vq[r] ? sum[r] / cnt[r] : 0.0f;
            }
            /* box smooth over VALID bins only; hp = p - smooth */
            for (int r = 0; r < nr; r++) {
                if (!vq[r]) { hq[r] = 0.0f; continue; }
                double acc = 0; int c = 0;
                int lo = r - hw < 0 ? 0 : r - hw, hi = r + hw >= nr ? nr - 1 : r + hw;
                for (int j = lo; j <= hi; j++) if (vq[j]) { acc += pq[j]; c++; }
                if (c >= hw) hq[r] = pq[r] - (float)(acc / c);
                else { hq[r] = 0.0f; vq[r] = 0; }
            }
        }
        float *ring = d->ring + (size_t)s * nr;
        for (int r = 0; r < nr; r++) {
            int nv = 0, pos = 0, neg = 0;
            for (int q = 0; q < ns; q++) {
                if (!vd[(size_t)q * nr + r]) continue;
                float v = hp[(size_t)q * nr + r];
                med[nv++] = v;
                if (v > 0) pos++; else if (v < 0) neg++;
            }
            /* vote: enough sectors covered, ALL agreeing in sign */
            if (nv < ns - 2 || (pos && neg)) { ring[r] = 0; continue; }
            qsort(med, nv, sizeof(float), cmpf);
            float m = (nv & 1) ? med[nv / 2] : 0.5f * (med[nv / 2 - 1] + med[nv / 2]);
            if (fabsf(m) < (float)min_amp) { ring[r] = 0; continue; }
            if (m > (float)max_amp) m = (float)max_amp;
            if (m < (float)-max_amp) m = (float)-max_amp;
            ring[r] = m;
            detected++;
        }
    }
    free(p); free(hp); free(vd);
    d->have_rings = detected >= 2;
    return detected;
}

/* subtract the detected rings from a float tile at global offset (z0,y0,x0).
 * scale converts the ring's u8 units to the tile's intensity units
 * (1/255 for plain u8->f01; 1/(hi-lo) after fy_norm_apply_u8).
 * Masked voxels (f <= 0) are left untouched; output clamped at 0. */
void fy_dering_apply(const fy_dering *d, float *f, long z0, long y0, long x0,
                     long dz, long dy, long dx, double scale) {
    if (!d->have_rings) return;
    int nr = d->nr;
    /* per-(y,x) radius-bin map computed once, reused across all dz slices */
    int *map = (int *)malloc(sizeof(int) * (size_t)dy * dx);
    if (map) {
        for (long y = 0; y < dy; y++) {
            double gy = (double)(y0 + y) - d->cy, gy2 = gy * gy;
            int *mrow = map + y * dx;
            for (long x = 0; x < dx; x++) {
                double gx = (double)(x0 + x) - d->cx;
                int ri = (int)(sqrt(gy2 + gx * gx) + 0.5);
                mrow[x] = ri < nr ? ri : -1;
            }
        }
    }
    for (long z = 0; z < dz; z++) {
        int slab = (int)((z0 + z) / d->slab_z);
        if (slab < 0) slab = 0;
        if (slab >= d->nslab) slab = d->nslab - 1;
        const float *ring = d->ring + (size_t)slab * nr;
        float sc = (float)scale;
        float *sl = f + (size_t)z * dy * dx;
        for (long y = 0; y < dy; y++) {
            float *row = sl + (size_t)y * dx;
            if (map) {
                const int *mrow = map + y * dx;
                for (long x = 0; x < dx; x++) {
                    int ri = mrow[x];
                    if (ri < 0 || row[x] <= 0.0f) continue;
                    float v = row[x] - sc * ring[ri];
                    row[x] = v > 0.0f ? v : 0.0f;
                }
            } else {  /* alloc-failure fallback: per-voxel radius */
                double gy = (double)(y0 + y) - d->cy, gy2 = gy * gy;
                for (long x = 0; x < dx; x++) {
                    if (row[x] <= 0.0f) continue;
                    double gx = (double)(x0 + x) - d->cx;
                    int ri = (int)(sqrt(gy2 + gx * gx) + 0.5);
                    if (ri >= nr) continue;
                    float v = row[x] - sc * ring[ri];
                    row[x] = v > 0.0f ? v : 0.0f;
                }
            }
        }
    }
    free(map);
}
