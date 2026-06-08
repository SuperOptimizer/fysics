/* pipeline.c -- the 2-PASS whole-volume preprocessing ORCHESTRATION in pure C + OpenMP.
 *
 * C port of superres/fysics_pipeline.py (preprocess_volume + the two-pass stat
 * accumulation). fysics' per-chunk kernels do the MATH; this streams a (possibly 20TB)
 * local raw-u8 zarr tile-by-tile, halo-padded, applies the validated, auto-calibrated
 * chain, and writes the inner tile back -- seam-free because every output voxel was
 * processed with real neighbour context (the halo).
 *
 * THE WHOLE POINT (vs the python orchestrator): the pass-2 loop is `#pragma omp parallel
 * for` over INPUT tiles -- no GIL -- so all cores process tiles concurrently.
 *
 * PASS1 (cheap, streaming, constant RAM): GLOBAL stats from SAMPLE tiles + a streaming
 *   sweep for norm/zdrift:
 *     - guided_eps = (3 * median flat_noise(blk=8))^2   [BM18 clean-noise-floor driver]
 *     - air_cut_u8 = global histogram valley of the SCRATCH-denoised samples;
 *       air_cut_band = max(4,(valley-dark)/4)
 *     - norm lo/hi percentiles (GATE: only if span < 0.40)
 *     - z-drift per-z profile (fy_zdrift_accumulate -> finalize), GATED on
 *       coherence>=0.5 AND slope_frac>=0.05 AND metadata beam_drift>=0.05
 *     - downsample factor from the measured PSF sigma map (p5..median blend)
 *     - GLOBAL deconv-output rescale range (seam-safe), if cfg->do_deconv
 * PASS2 (parallel): per input tile, halo-padded:
 *     (a) norm_apply_u8 + zdrift_apply   (intensity corrections FIRST, gated)
 *     (b) deconv  (cfg->do_deconv; matched Wiener if use_matched_deconv else plain)
 *     (c) guided_denoise(guided_eps)
 *     (d) air-zero (scratch guided-denoise -> local valley clamped to global+-band -> zero)
 *     (e) optional MUSICA (per-slice, clip-aware)
 *   then take the inner tile and either write it directly (tile==chunk grid) OR, when
 *   downsampling, anti-alias gaussian(0.5*factor) + decimate on the GLOBAL f-grid and
 *   scatter the decimated atom into a CHUNK ACCUMULATOR (decouples input tiles from
 *   output chunks; flush-when-full -> bounded RAM, seam-correct on the global grid).
 */
#include "fysics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------------------------------------------------------------- small helpers */
static long ceil_div(long a, long b) { return (a + b - 1) / b; }
static long lmin(long a, long b) { return a < b ? a : b; }
static long lmax(long a, long b) { return a > b ? a : b; }

static void u8_to_f01(const unsigned char *in, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = in[i] * (1.0f / 255.0f);
}
static int any_nonzero(const unsigned char *b, size_t n) {
    for (size_t i = 0; i < n; i++) if (b[i]) return 1;
    return 0;
}
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median_sorted(double *v, int n) {
    if (n <= 0) return 0.0;
    qsort(v, n, sizeof(double), cmp_double);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static double percentile_sorted(const double *v, int n, double pct) { /* v MUST be sorted */
    if (n <= 0) return 0.0;
    double idx = (pct / 100.0) * (n - 1);
    int i = (int)idx; double fr = idx - i;
    if (i >= n - 1) return v[n - 1];
    return v[i] * (1.0 - fr) + v[i + 1] * fr;
}

/* separable 3D gaussian blur (nearest-edge), matched to the anti-alias decimate cutoff. */
static void gauss3d(const float *in, float *out, int nz, int ny, int nx, double sigma) {
    if (sigma <= 0.0) { memcpy(out, in, sizeof(float) * (size_t)nz * ny * nx); return; }
    int r = (int)(3.0 * sigma + 0.5); if (r < 1) r = 1;
    int kn = 2 * r + 1;
    float *k = (float *)malloc(sizeof(float) * kn);
    double s = 0;
    for (int i = -r; i <= r; i++) { k[i + r] = (float)exp(-(double)i * i / (2.0 * sigma * sigma)); s += k[i + r]; }
    for (int i = 0; i < kn; i++) k[i] /= (float)s;
    size_t N = (size_t)nz * ny * nx;
    float *t1 = (float *)malloc(sizeof(float) * N);
    float *t2 = (float *)malloc(sizeof(float) * N);
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        double a = 0; for (int i = -r; i <= r; i++) { int xx = x + i; if (xx < 0) xx = 0; if (xx >= nx) xx = nx - 1;
            a += k[i + r] * in[((size_t)z * ny + y) * nx + xx]; }
        t1[((size_t)z * ny + y) * nx + x] = (float)a;
    }
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        double a = 0; for (int i = -r; i <= r; i++) { int yy = y + i; if (yy < 0) yy = 0; if (yy >= ny) yy = ny - 1;
            a += k[i + r] * t1[((size_t)z * ny + yy) * nx + x]; }
        t2[((size_t)z * ny + y) * nx + x] = (float)a;
    }
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        double a = 0; for (int i = -r; i <= r; i++) { int zz = z + i; if (zz < 0) zz = 0; if (zz >= nz) zz = nz - 1;
            a += k[i + r] * t2[((size_t)zz * ny + y) * nx + x]; }
        out[((size_t)z * ny + y) * nx + x] = (float)a;
    }
    free(k); free(t1); free(t2);
}

/* physics from cfg; scaled variant applies the regime delta_beta partial-inversion. */
static fy_physics phys_from_cfg(const fy_pipeline_cfg *c) {
    fy_physics p;
    p.delta_beta = c->delta_beta; p.energy_kev = c->energy_kev; p.distance_mm = c->distance_mm;
    p.pixel_um = c->pixel_um; p.unsharp_sigma = c->unsharp_sigma; p.unsharp_coeff = c->unsharp_coeff;
    p.psf_sigma_vox = c->psf_sigma_vox;
    return p;
}
static fy_physics scaled_phys_from_cfg(const fy_pipeline_cfg *c) {
    fy_physics p = phys_from_cfg(c);
    p.delta_beta *= fy_auto_deltabeta_scale(&p);
    return p;
}

/* ============================================================================
 * CHUNK ACCUMULATOR -- decouples the INPUT tile grid from the OUTPUT chunk grid
 * for downsampling. Decimated atoms (in DOWNSAMPLED coords, on the GLOBAL f-grid)
 * are SCATTERED into output chunks of `C` voxels; a chunk is FLUSHED (written +
 * freed) as soon as every voxel it covers has been filled. Only partially-filled
 * boundary chunks live in RAM -> streaming, bounded memory.
 *
 * SEAM-CORRECTNESS: every tile's decimation samples lie on ONE shared global grid
 * (input coord % factor == 0), keyed by the tile's absolute origin, so adjacent
 * tiles contribute DISJOINT, non-overlapping atoms that tile the output exactly --
 * no double-count, no gap, no seam. A chunk's expected fill count is its true voxel
 * volume; it flushes only when that exact count is reached, so each chunk is written
 * ONCE, fully assembled (never a partial write a later atom must read-modify-write).
 *
 * BOUNDED RAM: a chunk is held only between its first touch and the moment it is
 * complete. With atoms arriving in raster tile order, at most O(one output-chunk
 * plane) of partial chunks are ever live -- never the whole volume.
 *
 * THREAD SAFETY: one omp lock guards the live-chunk hash map; the heavy I/O
 * (fy_zarr_write_chunk) runs OUTSIDE the lock.
 * ========================================================================== */
typedef struct chunk_node {
    long cz, cy, cx;          /* chunk index (output-chunk units) */
    long ez, ey, ex;          /* this chunk's true extent (edge chunks are smaller) */
    long filled;              /* voxels written so far */
    unsigned char *buf;       /* ez*ey*ex u8 (calloc'd to fill_value 0 = air) */
    struct chunk_node *next;  /* hash-bucket chain */
} chunk_node;

typedef struct {
    long Sz, Sy, Sx;          /* output shape (downsampled voxels) */
    long C;                   /* output chunk edge (cubic) */
    long nbuckets;
    chunk_node **buckets;
    const fy_zarr *out;
#ifdef _OPENMP
    omp_lock_t lock;
#endif
} chunk_accum;

static unsigned long acc_hash(long cz, long cy, long cx, long nb) {
    unsigned long h = (unsigned long)(cz * 73856093L ^ cy * 19349663L ^ cx * 83492791L);
    return h % (unsigned long)nb;
}
static void acc_init(chunk_accum *A, const fy_zarr *out, long Sz, long Sy, long Sx, long C) {
    A->out = out; A->Sz = Sz; A->Sy = Sy; A->Sx = Sx; A->C = C;
    A->nbuckets = 4096;
    A->buckets = (chunk_node **)calloc(A->nbuckets, sizeof(chunk_node *));
#ifdef _OPENMP
    omp_init_lock(&A->lock);
#endif
}
static void acc_extent(const chunk_accum *A, long cz, long cy, long cx, long *ez, long *ey, long *ex) {
    *ez = lmin(A->C, A->Sz - cz * A->C);
    *ey = lmin(A->C, A->Sy - cy * A->C);
    *ex = lmin(A->C, A->Sx - cx * A->C);
}

/* place `atom` (az*ay*ax u8) at OUTPUT-space origin (oz,oy,ox); flush full chunks. */
static void acc_add(chunk_accum *A, long oz, long oy, long ox,
                    const unsigned char *atom, long az, long ay, long ax) {
    if (az <= 0 || ay <= 0 || ax <= 0) return;
    long C = A->C;
    long cz0 = oz / C, cy0 = oy / C, cx0 = ox / C;
    long cz1 = (oz + az - 1) / C, cy1 = (oy + ay - 1) / C, cx1 = (ox + ax - 1) / C;
    for (long cz = cz0; cz <= cz1; cz++)
    for (long cy = cy0; cy <= cy1; cy++)
    for (long cx = cx0; cx <= cx1; cx++) {
        long ez, ey, ex; acc_extent(A, cz, cy, cx, &ez, &ey, &ex);
        long exp_vox = ez * ey * ex;
        long bz0 = cz * C, by0 = cy * C, bx0 = cx * C;
        long lz0 = lmax(oz, bz0), ly0 = lmax(oy, by0), lx0 = lmax(ox, bx0);
        long lz1 = lmin(oz + az, bz0 + ez), ly1 = lmin(oy + ay, by0 + ey), lx1 = lmin(ox + ax, bx0 + ex);
        if (lz1 <= lz0 || ly1 <= ly0 || lx1 <= lx0) continue;
        unsigned char *flush_buf = NULL; long fcz = 0, fcy = 0, fcx = 0, fez = 0, fey = 0, fex = 0;
#ifdef _OPENMP
        omp_set_lock(&A->lock);
#endif
        unsigned long h = acc_hash(cz, cy, cx, A->nbuckets);
        chunk_node *nd = A->buckets[h];
        while (nd && !(nd->cz == cz && nd->cy == cy && nd->cx == cx)) nd = nd->next;
        if (!nd) {
            nd = (chunk_node *)malloc(sizeof(chunk_node));
            nd->cz = cz; nd->cy = cy; nd->cx = cx; nd->ez = ez; nd->ey = ey; nd->ex = ex;
            nd->filled = 0;
            nd->buf = (unsigned char *)calloc((size_t)exp_vox, 1);
            nd->next = A->buckets[h]; A->buckets[h] = nd;
        }
        for (long z = lz0; z < lz1; z++)
        for (long y = ly0; y < ly1; y++) {
            const unsigned char *src = atom + (((z - oz) * ay) + (y - oy)) * ax + (lx0 - ox);
            unsigned char *dst = nd->buf + (((z - bz0) * nd->ey) + (y - by0)) * nd->ex + (lx0 - bx0);
            memcpy(dst, src, (size_t)(lx1 - lx0));
        }
        nd->filled += (lz1 - lz0) * (ly1 - ly0) * (lx1 - lx0);
        if (nd->filled >= exp_vox) {
            chunk_node **pp = &A->buckets[h];
            while (*pp != nd) pp = &(*pp)->next;
            *pp = nd->next;
            flush_buf = nd->buf; fcz = nd->cz; fcy = nd->cy; fcx = nd->cx;
            fez = nd->ez; fey = nd->ey; fex = nd->ex;
            free(nd);
        }
#ifdef _OPENMP
        omp_unset_lock(&A->lock);
#endif
        if (flush_buf) {
            fy_zarr_write_chunk(A->out, fcz, fcy, fcx, flush_buf, fez, fey, fex);
            free(flush_buf);
        }
    }
}

/* flush remaining partial chunks (volume-edge chunks that never fully filled). */
static void acc_finalize(chunk_accum *A) {
    for (long h = 0; h < A->nbuckets; h++) {
        chunk_node *nd = A->buckets[h];
        while (nd) {
            chunk_node *nx = nd->next;
            fy_zarr_write_chunk(A->out, nd->cz, nd->cy, nd->cx, nd->buf, nd->ez, nd->ey, nd->ex);
            free(nd->buf); free(nd);
            nd = nx;
        }
        A->buckets[h] = NULL;
    }
    free(A->buckets);
#ifdef _OPENMP
    omp_destroy_lock(&A->lock);
#endif
}

/* ---------------------------------------------------------------- PSF measurement
 * Effective system PSF sigma (voxels) from subpixel-aligned air<->papyrus edges in a
 * u8 tile -- direct port of _measure_tile_psf_sigma (half=7, sup=8, min_contrast=40 u8,
 * 50% subpixel crossing align, LSF=grad(ESF), sigma=2nd moment). -1 if too few edges. */
static unsigned int psf_rng;
static unsigned int psf_rand(void) { psf_rng = psf_rng * 1103515245u + 12345u; return (psf_rng >> 16) & 0x7fff; }

static double measure_tile_psf_sigma(const unsigned char *vol, int Z, int Y, int X) {
    const int half = 7, sup = 8, min_c = 40, max_edges = 400;
    int plen = 2 * half + 1;
    int glen = 2 * half * sup + 1;
    double *grid = (double *)malloc(sizeof(double) * glen);
    for (int i = 0; i < glen; i++) grid[i] = -half + (double)i * (2.0 * half) / (glen - 1);
    double *acc = (double *)calloc(glen, sizeof(double));
    double *prof_cross = (double *)malloc(sizeof(double) * max_edges);
    double *prof_esf = (double *)malloc(sizeof(double) * (size_t)max_edges * plen);
    int nprof = 0;
    psf_rng = 1u;
    for (int ax = 1; ax <= 2 && nprof < max_edges; ax++) {
        for (int it = 0; it < 2500 && nprof < max_edges; it++) {
            int z, line[64];
            if (ax == 1) {
                if (Y - 2 * half <= 0) break;
                z = psf_rand() % Z; int y = half + psf_rand() % (Y - 2 * half); int x = psf_rand() % X;
                for (int i = 0; i < plen; i++) line[i] = vol[((size_t)z * Y + (y - half + i)) * X + x];
            } else {
                if (X - 2 * half <= 0) break;
                z = psf_rand() % Z; int y = psf_rand() % Y; int x = half + psf_rand() % (X - 2 * half);
                for (int i = 0; i < plen; i++) line[i] = vol[((size_t)z * Y + y) * X + (x - half + i)];
            }
            double lo = 0.5 * (line[0] + line[1]);
            double hi = 0.5 * (line[plen - 1] + line[plen - 2]);
            if (fabs(hi - lo) < min_c) continue;
            int s[64];
            if (hi > lo) for (int i = 0; i < plen; i++) s[i] = line[i];
            else         for (int i = 0; i < plen; i++) s[i] = line[plen - 1 - i];
            double dmin = 1e9; int amax = 0; double dmax = -1e9;
            for (int i = 0; i < plen - 1; i++) {
                double d = (double)s[i + 1] - s[i];
                if (d < dmin) dmin = d;
                if (d > dmax) { dmax = d; amax = i; }
            }
            if (dmin < -8) continue;
            if (!(amax == half - 1 || amax == half || amax == half + 1)) continue;
            int smin = s[0]; for (int i = 1; i < plen; i++) if (s[i] < smin) smin = s[i];
            double e[64]; double emax = 0;
            for (int i = 0; i < plen; i++) { e[i] = s[i] - smin; if (e[i] > emax) emax = e[i]; }
            if (emax <= 0) continue;
            for (int i = 0; i < plen; i++) e[i] /= emax;
            int idx0 = -1; for (int i = 0; i < plen; i++) if (e[i] >= 0.5) { idx0 = i; break; }
            if (idx0 <= 0) continue;
            double frac = (0.5 - e[idx0 - 1]) / (e[idx0] - e[idx0 - 1] + 1e-9);
            prof_cross[nprof] = idx0 - 1 + frac;
            for (int i = 0; i < plen; i++) prof_esf[(size_t)nprof * plen + i] = e[i];
            nprof++;
        }
    }
    if (nprof < 30) { free(grid); free(acc); free(prof_cross); free(prof_esf); return -1.0; }
    for (int p = 0; p < nprof; p++) {
        double cross = prof_cross[p];
        const double *e = prof_esf + (size_t)p * plen;
        for (int gi = 0; gi < glen; gi++) {
            double xs = grid[gi] + cross; double v;
            if (xs <= 0) v = 0.0;
            else if (xs >= plen - 1) v = 1.0;
            else { int i = (int)xs; double f = xs - i; v = e[i] * (1.0 - f) + e[i + 1] * f; }
            acc[gi] += v;
        }
    }
    double *esf = (double *)malloc(sizeof(double) * glen);
    for (int gi = 0; gi < glen; gi++) esf[gi] = acc[gi] / nprof;
    double *lsf = (double *)malloc(sizeof(double) * glen); double lsum = 0;
    for (int gi = 0; gi < glen; gi++) {
        double d;
        if (gi == 0) d = esf[1] - esf[0];
        else if (gi == glen - 1) d = esf[glen - 1] - esf[glen - 2];
        else d = 0.5 * (esf[gi + 1] - esf[gi - 1]);
        if (d < 0) d = 0;
        lsf[gi] = d; lsum += d;
    }
    double sig = -1.0;
    if (lsum > 0) {
        double mu = 0; for (int gi = 0; gi < glen; gi++) { lsf[gi] /= lsum; mu += grid[gi] * lsf[gi]; }
        double var = 0; for (int gi = 0; gi < glen; gi++) var += (grid[gi] - mu) * (grid[gi] - mu) * lsf[gi];
        sig = sqrt(var);
        if (!(sig > 0.3 && sig < 5.0)) sig = -1.0;
    }
    free(grid); free(acc); free(prof_cross); free(prof_esf); free(esf); free(lsf);
    return sig;
}

/* integer safe downsample factor from the PSF sigma map (port of _downsample_factor_from_psf). */
static int downsample_factor_from_psf(double p5, double med, double aggr) {
    double sigma = p5 + aggr * (med - p5);
    if (sigma <= 0) return 1;
    double f0 = sqrt(-log(0.1) / (2.0 * M_PI * M_PI * sigma * sigma));
    double factor = 0.5 / f0;
    int fi = (int)factor;
    return fi < 1 ? 1 : fi;
}

/* iterated scratch denoise (guided eps=0.01, `passes` iterations); `buf` in/out, `tmp` scratch. */
static void scratch_denoise(float *buf, int nz, int ny, int nx, int passes, float *tmp) {
    size_t n = (size_t)nz * ny * nx;
    for (int p = 0; p < passes; p++) {
        if (fy_guided_denoise(buf, tmp, nz, ny, nx, 2, 0.01) != 0) break;
        memcpy(buf, tmp, sizeof(float) * n);
    }
}

/* ----------------------------------------------------------------- per-tile chain.
 * `f` is the (corrected) haloed tile in [0,1]; returns the processed [0,1] in `f`.
 * `orig` = the corrected pre-deconv tile (air-blend baseline). `b1`,`b2` are caller-
 * owned per-thread scratch. u8orig = the haloed source (MUSICA rails). */
static void process_tile(float *f, const float *orig, const unsigned char *u8orig,
                         int nz, int ny, int nx, const fy_pipeline_cfg *cfg,
                         int have_dec_range, double dec_lo, double dec_hi,
                         float *b1, float *b2) {
    size_t N = (size_t)nz * ny * nx;

    /* (b) DECONV (STORED off for BM18; matched Wiener when requested else plain auto-reg). */
    if (cfg->do_deconv) {
        fy_physics ph = scaled_phys_from_cfg(cfg);
        int rc = (cfg->use_matched_deconv && cfg->psf_sigma_vox > 0)
            ? fy_deconvolve_matched(f, b1, nz, ny, nx, &ph, cfg->deconv_tikhonov)
            : fy_deconvolve(f, b1, nz, ny, nx, &ph, -1.0);
        if (rc == 0) {
            if (have_dec_range && dec_hi - dec_lo > 1e-6) {
                float lo = (float)dec_lo, inv = 1.0f / (float)(dec_hi - dec_lo);
                for (size_t i = 0; i < N; i++) b1[i] = (b1[i] - lo) * inv;
            }
            memcpy(f, b1, sizeof(float) * N);
        }
    }

    /* (c) GUIDED DENOISE at the calibrated eps. */
    if (cfg->guided_eps > 0) {
        if (fy_guided_denoise(f, b1, nz, ny, nx, 2, cfg->guided_eps) == 0)
            memcpy(f, b1, sizeof(float) * N);
    }

    /* (d) AIR-ZERO: scratch denoise -> local valley clamped to global+-band -> zero. */
    if (cfg->do_air_zero) {
        memcpy(b2, orig, sizeof(float) * N);
        scratch_denoise(b2, nz, ny, nx, cfg->scratch_passes > 0 ? cfg->scratch_passes : 5, b1);
        long hist[256]; memset(hist, 0, sizeof(hist));
        for (size_t i = 0; i < N; i++) {
            int b = (int)(b2[i] * 255.0f + 0.5f); if (b < 0) b = 0; if (b > 255) b = 255; hist[b]++;
        }
        int dark = 0, light = 0, valley = 0;
        double d = fy_valley_depth(hist, &dark, &light, &valley);
        int cut;
        if (cfg->air_cut_u8 >= 0) {
            int gg = cfg->air_cut_u8, band = cfg->air_cut_band;
            if (d >= 0) {
                int local = (dark + valley) / 2;
                cut = local; if (cut < gg - band) cut = gg - band; if (cut > gg + band) cut = gg + band;
            } else cut = gg;
        } else if (d >= 0) {
            int mid = (dark + valley) / 2; cut = (dark + 8) < mid ? (dark + 8) : mid;
        } else {
            cut = (int)((cfg->air_thresh > 0 ? cfg->air_thresh : 0.05) * 255);
        }
        float cutf = cut / 255.0f;
        for (size_t i = 0; i < N; i++) if (b2[i] < cutf) f[i] = 0.0f;
    }

    /* (e) MUSICA viewing enhancement (per-slice, clip-aware via the export rails). */
    if (cfg->do_musica) {
        for (int z = 0; z < nz; z++) {
            float *sl = f + (size_t)z * ny * nx;
            float *o = b1 + (size_t)z * ny * nx;
            fy_musica2d(sl, o, ny, nx, cfg->musica_levels, (float)cfg->musica_p, (float)cfg->musica_core);
            for (size_t i = 0; i < (size_t)ny * nx; i++) {
                size_t gi = (size_t)z * ny * nx + i;
                unsigned char rr = u8orig[gi];
                if (rr >= 254 || rr == 0) o[i] = sl[i];
                else { float v = o[i]; o[i] = v < 0 ? 0 : (v > 1 ? 1 : v); }
            }
        }
        memcpy(f, b1, sizeof(float) * N);
    }
}

/* ============================================================================
 * fy_run_pipeline -- the public 2-pass entry point.
 * ========================================================================== */
int fy_run_pipeline(const char *in_root, const char *out_root, fy_pipeline_cfg *cfg,
                    int tile, int verbose) {
    fy_zarr zin;
    if (fy_zarr_open(&zin, in_root) != 0) { fprintf(stderr, "pipeline: cannot open %s\n", in_root); return 1; }
    long Z = zin.shape[0], Y = zin.shape[1], X = zin.shape[2];
    if (tile <= 0) tile = 128;
    if (verbose) fprintf(stderr, "pipeline: input %ldx%ldx%ld tile=%d\n", Z, Y, X, tile);

    /* halo: kernel halo from physics (deconv reach); air-zero/denoise are local r=2. */
    fy_physics base_ph = phys_from_cfg(cfg);
    int halo = cfg->do_deconv ? fy_kernel_halo(&base_ph) : 8;
    if (halo < 8) halo = 8;
    cfg->halo = halo;

    /* ===================== PASS 1: CALIBRATION (sample tiles) ===================== */
    long air_hist[256]; memset(air_hist, 0, sizeof(air_hist));
    double floors[64]; int nfloor = 0, nsamp = 0;
    double sigmas[64]; int nsig = 0;
    int g = 3;                          /* 3x3x3 candidate quasi-grid */
    long stile = tile;
    size_t scap = (size_t)stile * stile * stile;
    unsigned char *sbuf = (unsigned char *)malloc(scap);
    float *sf = (float *)malloc(sizeof(float) * scap);
    float *so = (float *)malloc(sizeof(float) * scap);
    float *stmp = (float *)malloc(sizeof(float) * scap);   /* scratch-denoise tmp */
    float *sdec = (cfg->do_deconv) ? (float *)malloc(sizeof(float) * scap) : NULL;
    double *dec_vals = NULL; size_t dec_n = 0, dec_cap = 0;

    for (int iz = 0; iz < g; iz++) for (int iy = 0; iy < g; iy++) for (int ix = 0; ix < g; ix++) {
        long z0 = (long)(((iz + 0.5) / g) * (Z - stile)); if (z0 < 0) z0 = 0; if (z0 > Z - stile) z0 = Z > stile ? Z - stile : 0;
        long y0 = (long)(((iy + 0.5) / g) * (Y - stile)); if (y0 < 0) y0 = 0; if (y0 > Y - stile) y0 = Y > stile ? Y - stile : 0;
        long x0 = (long)(((ix + 0.5) / g) * (X - stile)); if (x0 < 0) x0 = 0; if (x0 > X - stile) x0 = X > stile ? X - stile : 0;
        long dz = lmin(stile, Z - z0), dy = lmin(stile, Y - y0), dx = lmin(stile, X - x0);
        size_t n = (size_t)dz * dy * dx;
        if (fy_zarr_read(&zin, z0, y0, x0, dz, dy, dx, sbuf) != 0) continue;
        size_t occ = 0; for (size_t i = 0; i < n; i++) if (sbuf[i] > 0) occ++;
        if ((double)occ / n < 0.6) continue;
        u8_to_f01(sbuf, sf, n);
        double fn = fy_flat_noise(sf, (int)dz, (int)dy, (int)dx, 8);
        if (fn > 0 && nfloor < 64) floors[nfloor++] = fn;
        /* air histogram (scratch-denoised copy in `so`; keeps `sf` pristine for deconv below). */
        if (cfg->do_air_zero) {
            memcpy(so, sf, sizeof(float) * n);
            scratch_denoise(so, (int)dz, (int)dy, (int)dx, cfg->scratch_passes > 0 ? cfg->scratch_passes : 5, stmp);
            for (size_t i = 0; i < n; i++) {
                int b = (int)(so[i] * 255.0f + 0.5f); if (b < 0) b = 0; if (b > 255) b = 255; air_hist[b]++;
            }
        }
        /* PSF sigma for downsample */
        if (cfg->do_downsample && nsig < 64) {
            double sg = measure_tile_psf_sigma(sbuf, (int)dz, (int)dy, (int)dx);
            if (sg > 0) sigmas[nsig++] = sg;
        }
        /* deconv output range (seam-safe global rescale) */
        if (cfg->do_deconv && sdec) {
            fy_physics ph = scaled_phys_from_cfg(cfg);
            int rc = (cfg->use_matched_deconv && cfg->psf_sigma_vox > 0)
                ? fy_deconvolve_matched(sf, sdec, (int)dz, (int)dy, (int)dx, &ph, cfg->deconv_tikhonov)
                : fy_deconvolve(sf, sdec, (int)dz, (int)dy, (int)dx, &ph, -1.0);
            if (rc == 0) {
                if (dec_n + n > dec_cap) { dec_cap = (dec_n + n) * 2; dec_vals = realloc(dec_vals, sizeof(double) * dec_cap); }
                for (size_t i = 0; i < n; i++) dec_vals[dec_n++] = sdec[i];
            }
        }
        nsamp++;
    }
    free(sbuf); free(sf); free(so); free(stmp); if (sdec) free(sdec);

    /* commit guided_eps = (4.2 * median flat_nf)^2 ~0.004 (RETUNE consensus) unless pre-set. */
    if (cfg->guided_eps <= 0) {
        double flat_nf = nfloor ? median_sorted(floors, nfloor) : 0.015;
        cfg->guided_eps = (4.2 * flat_nf) * (4.2 * flat_nf);
    }
    /* commit air cut: PREFER the PHYSICS WINDOW-FLOOR (cfg->air_thresh), NOT the scratch-valley.
     * RETUNE finding: on non-bimodal volumes the scratch denoise manufactures a false dark mode
     * and the valley lands INSIDE real low-density papyrus (u8 62 destroyed 0.42% of confident
     * material, RMSE 90x worse than the floor u8 12). Physics floor == no-cut on RMSE. Valley =
     * fallback only when there is no export window. */
    if (cfg->do_air_zero && cfg->air_cut_u8 < 0) {
        if (cfg->air_thresh > 0) {
            cfg->air_cut_u8 = (int)(cfg->air_thresh * 255 + 0.5); cfg->air_cut_band = 4;
        } else {
        long sum = 0; for (int i = 0; i < 256; i++) sum += air_hist[i];
        if (sum > 0) {
            int dark = 0, light = 0, valley = 0;
            double d = fy_valley_depth(air_hist, &dark, &light, &valley);
            if (d >= 0) {
                cfg->air_cut_u8 = valley;
                int band = (valley - dark) / 4; cfg->air_cut_band = band > 4 ? band : 4;
            } else {
                cfg->air_cut_u8 = (int)((cfg->air_thresh > 0 ? cfg->air_thresh : 0.05) * 255);
                cfg->air_cut_band = 8;
            }
        }
        }
    }
    /* commit deconv global range. */
    int have_dec_range = 0; double dec_lo = 0.0, dec_hi = 1.0;
    if (cfg->do_deconv && dec_n > 0) {
        qsort(dec_vals, dec_n, sizeof(double), cmp_double);
        dec_lo = percentile_sorted(dec_vals, dec_n, 0.1);
        dec_hi = percentile_sorted(dec_vals, dec_n, 99.9);
        if (dec_hi - dec_lo < 1e-6) { dec_lo = 0.0; dec_hi = 1.0; }
        have_dec_range = 1;
    }
    free(dec_vals);
    /* commit downsample factor from the PSF sigma map (p5..median blend). */
    int fds = 1;
    if (cfg->do_downsample && nsig >= 3) {
        qsort(sigmas, nsig, sizeof(double), cmp_double);
        double p5 = percentile_sorted(sigmas, nsig, 5.0);
        double med = percentile_sorted(sigmas, nsig, 50.0);
        cfg->psf_p5 = p5; cfg->psf_med = med;
        fds = downsample_factor_from_psf(p5, med, cfg->downsample_aggr);
        cfg->downsample_factor = fds;
        cfg->do_downsample = fds > 1;
    } else {
        cfg->do_downsample = 0;
    }
    fds = cfg->do_downsample ? cfg->downsample_factor : 1;
    if (verbose) fprintf(stderr, "[calib] samples=%d flat_nf_n=%d guided_eps=%.5f air_cut=%d band=%d halo=%d fds=%d\n",
                         nsamp, nfloor, cfg->guided_eps, cfg->air_cut_u8, cfg->air_cut_band, halo, fds);

    /* ===================== PASS 1b: global norm + z-drift (streaming) ===================== */
    fy_hist_state nhist; fy_hist_init(&nhist);
    double *zsums = (double *)calloc(Z, sizeof(double));
    long *zcnts = (long *)calloc(Z, sizeof(long));
    float *zfactor = NULL;
    int have_norm = 0, have_zdrift = 0;

    if (cfg->do_normalize || cfg->do_zdrift) {
        long ptile = tile > 256 ? tile : 256;
        long pnz = ceil_div(Z, ptile), pny = ceil_div(Y, ptile), pnx = ceil_div(X, ptile);
        long pntiles = pnz * pny * pnx;
        #pragma omp parallel
        {
            fy_hist_state lh; fy_hist_init(&lh);
            double *ls = (double *)calloc(Z, sizeof(double));
            long *lc = (long *)calloc(Z, sizeof(long));
            unsigned char *pb = (unsigned char *)malloc((size_t)ptile * ptile * ptile);
            float *pf = (float *)malloc(sizeof(float) * (size_t)ptile * ptile * ptile);
            #pragma omp for schedule(dynamic)
            for (long t = 0; t < pntiles; t++) {
                long iz = t / (pny * pnx), r = t % (pny * pnx), iy = r / pnx, ix = r % pnx;
                long z0 = iz * ptile, y0 = iy * ptile, x0 = ix * ptile;
                long dz = lmin(ptile, Z - z0), dy = lmin(ptile, Y - y0), dx = lmin(ptile, X - x0);
                size_t n = (size_t)dz * dy * dx;
                if (fy_zarr_read(&zin, z0, y0, x0, dz, dy, dx, pb) != 0) continue;
                if (!any_nonzero(pb, n)) continue;
                if (cfg->do_normalize) fy_hist_accumulate_u8(&lh, pb, n);
                if (cfg->do_zdrift) {
                    u8_to_f01(pb, pf, n);
                    fy_zdrift_accumulate(pf, (int)dz, (int)dy, (int)dx, (int)z0, ls, lc, 0.10f);
                }
            }
            #pragma omp critical
            {
                if (cfg->do_normalize) fy_hist_merge(&nhist, &lh);
                for (long z = 0; z < Z; z++) { zsums[z] += ls[z]; zcnts[z] += lc[z]; }
            }
            free(ls); free(lc); free(pb); free(pf);
        }
        /* norm gate: only if (hi-lo)/255 < 0.40 */
        if (cfg->do_normalize && nhist.total > 0) {
            int lo = fy_hist_percentile_u8(&nhist, 0.5);
            int hi = fy_hist_percentile_u8(&nhist, 99.5);
            if ((hi - lo) / 255.0 < 0.40 && hi > lo) { cfg->norm_lo = lo; cfg->norm_hi = hi; have_norm = 1; }
        }
        /* z-drift gate: coherence>=0.5 AND slope_frac>=0.05 AND beam_drift>=0.05 */
        if (cfg->do_zdrift) {
            long valid = 0; for (long z = 0; z < Z; z++) if (zcnts[z] > 0) valid++;
            if (valid >= 8) {
                zfactor = (float *)malloc(sizeof(float) * Z);
                fy_zdrift_finalize(zsums, zcnts, (int)Z, zfactor);
                double sx = 0, sy = 0, sxx = 0, sxy = 0; long nv = 0;
                double *prof = (double *)malloc(sizeof(double) * Z);
                for (long z = 0; z < Z; z++) {
                    prof[z] = zcnts[z] ? zsums[z] / zcnts[z] : -1;
                    if (prof[z] >= 0) { sx += z; sy += prof[z]; sxx += (double)z * z; sxy += (double)z * prof[z]; nv++; }
                }
                double slope = (nv * sxy - sx * sy) / (nv * sxx - sx * sx + 1e-12);
                double icpt = (sy - slope * sx) / (nv + 1e-12);
                double lmn = 1e30, lmx = -1e30, totvar = 0, prev = -1;
                double *medacc = (double *)malloc(sizeof(double) * (nv > 0 ? nv : 1)); long nmed = 0;
                for (long z = 0; z < Z; z++) if (prof[z] >= 0) {
                    double lin = slope * z + icpt; if (lin < lmn) lmn = lin; if (lin > lmx) lmx = lin;
                    if (prev >= 0) totvar += fabs(prof[z] - prev); prev = prof[z];
                    medacc[nmed++] = prof[z];
                }
                double lin_span = lmx - lmn;
                double coherence = lin_span / (totvar + 1e-9);
                double med = median_sorted(medacc, (int)nmed);
                double slope_frac = lin_span / (med + 1e-9);
                double beam = (cfg->machine_current_start > 0)
                    ? fabs(cfg->machine_current_stop - cfg->machine_current_start) / cfg->machine_current_start : -1;
                int meta_ok = (beam < 0) || (beam >= 0.05);
                if (coherence >= 0.5 && slope_frac >= 0.05 && meta_ok) have_zdrift = 1;
                free(prof); free(medacc);
            }
        }
        if (!have_zdrift) { free(zfactor); zfactor = NULL; }
        if (verbose) fprintf(stderr, "[pass1] norm=%d(lo=%d hi=%d) zdrift=%d\n",
                             have_norm, cfg->norm_lo, cfg->norm_hi, have_zdrift);
    }

    /* ===================== OUTPUT zarr (shape = ceil(in/fds), chunk = tile) ===================== */
    long oZ = ceil_div(Z, fds), oY = ceil_div(Y, fds), oX = ceil_div(X, fds);
    long oshape[3] = { oZ, oY, oX };
    long ochunk[3] = { tile, tile, tile };
    fy_zarr zout;
    if (fy_zarr_create(&zout, out_root, oshape, ochunk) != 0) {
        fprintf(stderr, "pipeline: cannot create %s\n", out_root);
        free(zsums); free(zcnts); free(zfactor); return 1;
    }
    chunk_accum acc; int use_acc = (fds > 1);
    if (use_acc) acc_init(&acc, &zout, oZ, oY, oX, tile);

    /* ===================== PASS 2: parallel per-tile ===================== */
    long ntz = ceil_div(Z, tile), nty = ceil_div(Y, tile), ntx = ceil_div(X, tile);
    long ntiles = ntz * nty * ntx;
    long out_done = 0;
    if (verbose) fprintf(stderr, "[pass2] %ld tiles, halo=%d, downsample=%d(fds=%d)\n",
                         ntiles, halo, cfg->do_downsample, fds);

    #pragma omp parallel
    {
        long bz_max = tile + 2 * halo;
        size_t cap = (size_t)bz_max * bz_max * bz_max;
        unsigned char *u8 = (unsigned char *)malloc(cap);
        float *f = (float *)malloc(sizeof(float) * cap);
        float *orig = (float *)malloc(sizeof(float) * cap);
        float *b1 = (float *)malloc(sizeof(float) * cap);
        float *b2 = (float *)malloc(sizeof(float) * cap);
        float *lp = use_acc ? (float *)malloc(sizeof(float) * cap) : NULL;
        unsigned char *ob = (unsigned char *)malloc(cap);

        #pragma omp for schedule(dynamic)
        for (long t = 0; t < ntiles; t++) {
            long iz = t / (nty * ntx), r = t % (nty * ntx), iy = r / ntx, ix = r % ntx;
            long z0 = iz * tile, y0 = iy * tile, x0 = ix * tile;
            long tz = lmin(tile, Z - z0), ty = lmin(tile, Y - y0), tx = lmin(tile, X - x0);
            long rz0 = lmax(0, z0 - halo), ry0 = lmax(0, y0 - halo), rx0 = lmax(0, x0 - halo);
            long rz1 = lmin(Z, z0 + tz + halo), ry1 = lmin(Y, y0 + ty + halo), rx1 = lmin(X, x0 + tx + halo);
            long hz = rz1 - rz0, hy = ry1 - ry0, hx = rx1 - rx0;
            size_t hn = (size_t)hz * hy * hx;
            if (fy_zarr_read(&zin, rz0, ry0, rx0, hz, hy, hx, u8) != 0) continue;
            if (!any_nonzero(u8, hn)) continue;

            /* (a) INTENSITY CORRECTIONS FIRST: normalize then z-drift (gated). */
            if (have_norm) fy_norm_apply_u8(u8, f, hn, (unsigned char)cfg->norm_lo, (unsigned char)cfg->norm_hi);
            else u8_to_f01(u8, f, hn);
            if (have_zdrift) fy_zdrift_apply(f, (int)hz, (int)hy, (int)hx, (int)rz0, zfactor);
            memcpy(orig, f, sizeof(float) * hn);

            /* (b-e) per-tile chain. */
            process_tile(f, orig, u8, (int)hz, (int)hy, (int)hx, cfg,
                         have_dec_range, dec_lo, dec_hi, b1, b2);

            long iz0 = z0 - rz0, iy0 = y0 - ry0, ix0 = x0 - rx0;

            if (use_acc) {
                /* fused anti-alias + decimate on the haloed block (real neighbours), sampled on
                 * the GLOBAL f-grid keyed by the tile's absolute origin -> seam-free. */
                gauss3d(f, lp, (int)hz, (int)hy, (int)hx, 0.5 * fds);
                long sz = ((-z0) % fds + fds) % fds, sy = ((-y0) % fds + fds) % fds, sx = ((-x0) % fds + fds) % fds;
                long iz1 = iz0 + tz, iy1 = iy0 + ty, ix1 = ix0 + tx;
                long az = 0; for (long zz = iz0 + sz; zz < iz1; zz += fds) az++;
                long ay = 0; for (long yy = iy0 + sy; yy < iy1; yy += fds) ay++;
                long ax = 0; for (long xx = ix0 + sx; xx < ix1; xx += fds) ax++;
                if (az > 0 && ay > 0 && ax > 0) {
                    unsigned char *atom = (unsigned char *)malloc((size_t)az * ay * ax);
                    long ci = 0;
                    for (long zz = iz0 + sz; zz < iz1; zz += fds)
                    for (long yy = iy0 + sy; yy < iy1; yy += fds)
                    for (long xx = ix0 + sx; xx < ix1; xx += fds) {
                        float v = lp[((size_t)zz * hy + yy) * hx + xx] * 255.0f + 0.5f;
                        atom[ci++] = v < 0 ? 0 : (v > 255 ? 255 : (unsigned char)v);
                    }
                    /* output origin = ceil(z0/fds) = first global-grid index covered by this tile. */
                    acc_add(&acc, ceil_div(z0, fds), ceil_div(y0, fds), ceil_div(x0, fds), atom, az, ay, ax);
                    free(atom);
                }
            } else {
                /* tile == output chunk: convert the inner tile to u8 and write the chunk directly. */
                long oi = 0;
                for (long zz = iz0; zz < iz0 + tz; zz++)
                for (long yy = iy0; yy < iy0 + ty; yy++)
                for (long xx = ix0; xx < ix0 + tx; xx++) {
                    float v = f[((size_t)zz * hy + yy) * hx + xx] * 255.0f + 0.5f;
                    ob[oi++] = v < 0 ? 0 : (v > 255 ? 255 : (unsigned char)v);
                }
                fy_zarr_write_chunk(&zout, iz, iy, ix, ob, tz, ty, tx);
            }
            #pragma omp atomic
            out_done++;
            if (verbose && (out_done % 64 == 0)) {
                #pragma omp critical (prog)
                fprintf(stderr, "\r[pass2] %ld/%ld tiles", out_done, ntiles);
            }
        }
        free(u8); free(f); free(orig); free(b1); free(b2); if (lp) free(lp); free(ob);
    }

    if (use_acc) acc_finalize(&acc);
    if (verbose) fprintf(stderr, "\n[done] %ld tiles, out shape %ldx%ldx%ld\n", out_done, oZ, oY, oX);
    free(zsums); free(zcnts); free(zfactor);
    return 0;
}
