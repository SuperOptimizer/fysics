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
 *     - GLOBAL deconv-output rescale range (seam-safe), if cfg->do_deconv
 * PASS2 (parallel): per input tile, halo-padded:
 *     (a) norm_apply_u8 + zdrift_apply   (intensity corrections FIRST, gated)
 *     (b) deconv  (cfg->do_deconv; matched Wiener if use_matched_deconv else plain)
 *     (c) guided_denoise(guided_eps)
 *     (d) air-zero (scratch guided-denoise -> local valley clamped to global+-band -> zero)
 *     (e) optional MUSICA (per-slice, clip-aware)
 *   then take the inner tile and write it directly (tile == output chunk grid).
 * (Downsampling was removed: the volumes are not meaningfully oversampled and the auto-factor
 *  rested on an unreliable PSF sigma; resolution-preserving decimation would need a sharp
 *  anti-alias and a per-volume eyes-on check -- a manual tool, not an auto stage.)
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


/* iterated scratch denoise (guided eps=0.01, `passes` iterations); `buf` in/out, `tmp` scratch. */
/* SCRATCH smoothing for the air-cut valley decision. The scratch is thrown away after
 * thresholding (it only tightens the histogram modes), so it does NOT need the edge-preserving
 * guided filter -- a single PLAIN BOX SMOOTH at radius ~= passes*2+1 matches the multi-pass guided
 * scratch's valley within ~1 u8 at ~1/4 the cost (the guided filter does 4 box passes per call).
 * `passes` is interpreted as the equivalent guided-pass count -> box radius (passes>=1). */
static void scratch_denoise(float *buf, int nz, int ny, int nx, int passes, float *tmp, float *ws) {
    size_t n = (size_t)nz * ny * nx;
    int r = passes > 0 ? passes : 5;        /* 5 guided eps=0.01 passes ~ box r=5 (measured) */
    /* box_mean is not in==out safe (X pass reads ahead); smooth into ws, copy back. */
    fy_box_smooth(buf, ws, tmp, nz, ny, nx, r);
    memcpy(buf, ws, sizeof(float) * n);
}

/* ----------------------------------------------------------------- per-tile chain.
 * `f` is the (corrected) haloed tile in [0,1]; returns the processed [0,1] in `f`.
 * `orig` = the corrected pre-deconv tile (air-blend baseline). `b1`,`b2` are caller-
 * owned per-thread scratch. u8orig = the haloed source (MUSICA rails). */
static void process_tile(float *f, const float *orig, const unsigned char *u8orig,
                         int nz, int ny, int nx, const fy_pipeline_cfg *cfg,
                         int have_dec_range, double dec_lo, double dec_hi,
                         float *b1, float *b2, float *ws) {
    size_t N = (size_t)nz * ny * nx;

    /* (b) DECONV (STORED off for BM18; matched Wiener when requested else plain auto-reg).
     * The physics inverse must run on PHYSICAL ATTENUATION mu, not u8/255 -- our u8 is a clipped
     * linear window of mu (nabu conversion window window_lo/hi). So DE-WINDOW f (u8/255 -> mu)
     * before the inverse, run it, then RE-WINDOW back to [0,1]. (When no window is known, mu==f.) */
    if (cfg->do_deconv) {
        fy_physics ph = scaled_phys_from_cfg(cfg);
        int windowed = (cfg->window_hi > cfg->window_lo);
        float wlo = (float)cfg->window_lo, wspan = (float)(cfg->window_hi - cfg->window_lo);
        if (windowed) for (size_t i = 0; i < N; i++) f[i] = wlo + f[i] * wspan;   /* -> physical mu */
        int rc = (cfg->use_matched_deconv && cfg->psf_sigma_vox > 0)
            ? fy_deconvolve_matched(f, b1, nz, ny, nx, &ph, cfg->deconv_tikhonov)
            : fy_deconvolve(f, b1, nz, ny, nx, &ph, -1.0);
        if (rc == 0) {
            if (windowed) {                                 /* mu -> [0,1] window */
                float inv = 1.0f / wspan;
                for (size_t i = 0; i < N; i++) b1[i] = (b1[i] - wlo) * inv;
            }
            if (have_dec_range && dec_hi - dec_lo > 1e-6) {
                float lo = (float)dec_lo, inv = 1.0f / (float)(dec_hi - dec_lo);
                for (size_t i = 0; i < N; i++) b1[i] = (b1[i] - lo) * inv;
            }
            for (size_t i = 0; i < N; i++) {               /* clamp to [0,1] (rescale/window overshoot) */
                float v = b1[i]; f[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            }
        } else if (windowed) {                              /* deconv failed -> restore f to [0,1] */
            float inv = 1.0f / wspan;
            for (size_t i = 0; i < N; i++) f[i] = (f[i] - wlo) * inv;
        }
    }

    /* (c) GUIDED DENOISE at the calibrated eps. */
    if (cfg->guided_eps > 0) {
        if (fy_guided_denoise_ws(f, b1, nz, ny, nx, 2, cfg->guided_eps, ws) == 0)
            memcpy(f, b1, sizeof(float) * N);
    }

    /* (d) AIR-ZERO: scratch smooth (box) -> local valley clamped to global+-band -> zero.
     * Box-smooth ORIG directly into b2 (no pre-copy, no copy-back -- box_mean isn't in==out
     * safe but orig!=b2). The smoothed scratch lives in b2, read only for the histogram. */
    if (cfg->do_air_zero) {
        int sr = cfg->scratch_passes > 0 ? cfg->scratch_passes : 5;
        fy_box_smooth(orig, b2, b1, nz, ny, nx, sr);   /* b1 = tmp */
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
                /* local estimate: same PHYSICS-FLOOR -> valley interpolation as the global cut,
                 * clamped to the global +-band so it tracks the profile but can't run away. */
                int pf = (cfg->air_thresh > 0) ? (int)(cfg->air_thresh * 255 + 0.5) : 39;
                double a = cfg->air_cut_aggr; if (a < 0) a = 0; if (a > 1) a = 1;
                int local = (valley > pf) ? (int)(pf + a * (valley - pf) + 0.5) : pf;
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

    /* PRE-PASS: measure PSF sigma on a few sample tiles FIRST, so the AUTO-DECONV gate is decided
     * BEFORE halo/sdec/rescale setup (deconv reach drives the halo; the gate must precede it). */
    {
        long st = tile; size_t cap = (size_t)st * st * st;
        unsigned char *pb = (unsigned char *)malloc(cap);
        double psig[27]; int nps = 0;
        if (pb) {
            for (int iz = 0; iz < 3 && nps < 27; iz++)
              for (int iy = 0; iy < 3 && nps < 27; iy++)
                for (int ix = 0; ix < 3 && nps < 27; ix++) {
                    long z0 = lmin((long)iz * (Z / 3), Z - st);
                    long y0 = lmin((long)iy * (Y / 3), Y - st);
                    long x0 = lmin((long)ix * (X / 3), X - st);
                    if (z0 < 0 || y0 < 0 || x0 < 0) continue;
                    if (fy_zarr_read(&zin, z0, y0, x0, st, st, st, pb) != 0) continue;
                    if (!any_nonzero(pb, cap)) continue;
                    double sg = measure_tile_psf_sigma(pb, (int)st, (int)st, (int)st);
                    if (sg > 0) psig[nps++] = sg;
                }
            free(pb);
        }
        if (nps >= 3) {
            qsort(psig, nps, sizeof(double), cmp_double);
            double med = percentile_sorted(psig, nps, 50.0);
            cfg->psf_med = med; cfg->psf_p5 = percentile_sorted(psig, nps, 5.0);
            /* AUTO-DECONV GATE: matched-Wiener deconv recovers REAL signal only when sharp
             * (RMSE-to-truth: wins sigma<=1.0, loses >=1.3; boundary ~1.1). Above -> contrast-only
             * (cosmetic, amplifies noise) -> do NOT store. Caller --deconv forces it regardless. */
            if (!cfg->do_deconv && med <= 1.1) {
                cfg->do_deconv = 1; cfg->use_matched_deconv = 1; cfg->psf_sigma_vox = med;
                if (verbose) fprintf(stderr, "[calib] AUTO-DECONV ON: psf_med=%.2f<=1.1 (sharp->recovers signal)\n", med);
            } else if (verbose && !cfg->do_deconv) {
                fprintf(stderr, "[calib] auto-deconv OFF: psf_med=%.2f>1.1 (blurry->contrast-only, view-time)\n", med);
            }
            if (cfg->do_deconv && cfg->psf_sigma_vox <= 0) cfg->psf_sigma_vox = med;
        }
    }

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
    float *sws  = (float *)malloc(sizeof(float) * 6 * scap); /* guided workspace */
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
            scratch_denoise(so, (int)dz, (int)dy, (int)dx, cfg->scratch_passes > 0 ? cfg->scratch_passes : 5, stmp, sws);
            for (size_t i = 0; i < n; i++) {
                int b = (int)(so[i] * 255.0f + 0.5f); if (b < 0) b = 0; if (b > 255) b = 255; air_hist[b]++;
            }
        }
        /* PSF sigma -- measured for the auto-deconv
         * gate (deconv recovers real signal only when sharp, sigma<=~1.1). */
        if (nsig < 64) {
            double sg = measure_tile_psf_sigma(sbuf, (int)dz, (int)dy, (int)dx);
            if (sg > 0) sigmas[nsig++] = sg;
        }
        /* deconv output range (seam-safe global rescale) -- measured in the SAME windowed space
         * as pass-2: de-window sf (u8/255 -> mu), deconv, re-window, collect [0,1] values. */
        if (cfg->do_deconv && sdec) {
            fy_physics ph = scaled_phys_from_cfg(cfg);
            int windowed = (cfg->window_hi > cfg->window_lo);
            float wlo = (float)cfg->window_lo, wspan = (float)(cfg->window_hi - cfg->window_lo);
            if (windowed) for (size_t i = 0; i < n; i++) sf[i] = wlo + sf[i] * wspan;
            int rc = (cfg->use_matched_deconv && cfg->psf_sigma_vox > 0)
                ? fy_deconvolve_matched(sf, sdec, (int)dz, (int)dy, (int)dx, &ph, cfg->deconv_tikhonov)
                : fy_deconvolve(sf, sdec, (int)dz, (int)dy, (int)dx, &ph, -1.0);
            if (windowed) for (size_t i = 0; i < n; i++) sf[i] = (sf[i] - wlo) / wspan;  /* restore sf */
            if (rc == 0) {
                if (dec_n + n > dec_cap) { dec_cap = (dec_n + n) * 2; dec_vals = realloc(dec_vals, sizeof(double) * dec_cap); }
                for (size_t i = 0; i < n; i++)
                    dec_vals[dec_n++] = windowed ? (sdec[i] - wlo) / wspan : sdec[i];
            }
        }
        nsamp++;
    }
    free(sbuf); free(sf); free(so); free(stmp); free(sws); if (sdec) free(sdec);

    /* commit guided_eps = (k * median flat_nf)^2, k = cfg->denoise_k (profile: conservative 3.0
     * ~0.002 preserves detail, aggressive 4.2 ~0.004). Default 4.2 if unset. Unless pre-set. */
    if (cfg->guided_eps <= 0) {
        double flat_nf = nfloor ? median_sorted(floors, nfloor) : 0.015;
        double k = cfg->denoise_k > 0 ? cfg->denoise_k : 4.2;
        cfg->guided_eps = (k * flat_nf) * (k * flat_nf);
    }
    /* commit air cut: PHYSICS-FLOOR (conservative) -> VALLEY (aggressive) by cfg->air_cut_aggr.
     * The u8 is a clipped linear window of physical attenuation mu; the void/air level (mu~0) maps
     * to u8 = air_thresh*255 (the metadata WINDOW floor -- this is the PHYSICALLY CORRECT air level,
     * NOT a guess). The volume is not cleanly bimodal above that, so how much faint low-density
     * material to additionally remove is a USE-CASE choice up to the histogram valley.
     *   aggr 0 (conservative/FIDELITY): cut = physics window-floor (u8 ~39 here; zeros ~only true
     *           air -- matches the python reference, which preferred the window floor).
     *   aggr 1 (aggressive/READABILITY): cut = valley (~v7; also removes faint low-density material).
     * Both keep ~0% confident papyrus. Anchoring conservative to the PHYSICS floor (not dark+0.2span)
     * fixes the audit blocker where the C zeroed 36% the python kept. */
    if (cfg->do_air_zero && cfg->air_cut_u8 < 0) {
        long sum = 0; for (int i = 0; i < 256; i++) sum += air_hist[i];
        int dark = 0, light = 0, valley = 0;
        double d = (sum > 0) ? fy_valley_depth(air_hist, &dark, &light, &valley) : -1.0;
        int phys_floor = (cfg->air_thresh > 0) ? (int)(cfg->air_thresh * 255 + 0.5) : 39;
        if (d >= 0 && valley > phys_floor) {
            double a = cfg->air_cut_aggr; if (a < 0) a = 0; if (a > 1) a = 1;
            cfg->air_cut_u8 = (int)(phys_floor + a * (valley - phys_floor) + 0.5);
            int band = (valley - phys_floor) / 4; cfg->air_cut_band = band > 4 ? band : 4;
        } else {
            cfg->air_cut_u8 = phys_floor; cfg->air_cut_band = 4;
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
    /* (downsample removed -- the volume is not meaningfully oversampled; the auto-factor rested on
     * an unreliable PSF sigma. PSF is still measured above, but only for the auto-deconv gate.) */
    if (verbose) fprintf(stderr, "[calib] samples=%d flat_nf_n=%d guided_eps=%.5f air_cut=%d band=%d psf_med=%.2f deconv=%d halo=%d\n",
                         nsamp, nfloor, cfg->guided_eps, cfg->air_cut_u8, cfg->air_cut_band, cfg->psf_med, cfg->do_deconv, halo);

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
                /* SUBSAMPLE: read EVERY z-slab (the z-drift trend needs full z-coverage) but only
                 * a strided subset of (y,x) tiles when the grid is large enough to keep stats
                 * robust -- the norm histogram and per-z papyrus mean converge on a fraction of
                 * the xy tiles. Cuts pass-1b I/O ~4x on big volumes (a full redundant read else). */
                if (pny >= 4 && (iy & 1)) continue;
                if (pnx >= 4 && (ix & 1)) continue;
                long z0 = iz * ptile, y0 = iy * ptile, x0 = ix * ptile;
                long dz = lmin(ptile, Z - z0), dy = lmin(ptile, Y - y0), dx = lmin(ptile, X - x0);
                size_t n = (size_t)dz * dy * dx;
                if (fy_zarr_read(&zin, z0, y0, x0, dz, dy, dx, pb) != 0) continue;
                if (!any_nonzero(pb, n)) continue;
                if (cfg->do_normalize) fy_hist_accumulate_u8(&lh, pb, n);
                if (cfg->do_zdrift) {
                    /* z-drift only needs a coarse PER-Z trend (mean papyrus brightness per slice)
                     * to decide coherence/slope -- accumulate on a 4x4 spatial SUBSAMPLE in x,y
                     * (16x less work, statistically identical per-z mean over a 256-tile slab). */
                    const int ss = 4;
                    for (long z = 0; z < dz; z++) {
                        const unsigned char *sl = pb + (size_t)z * dy * dx;
                        double s = 0; long c = 0;
                        for (long yy = 0; yy < dy; yy += ss)
                            for (long xx = 0; xx < dx; xx += ss) {
                                float v = sl[(size_t)yy * dx + xx] * (1.0f / 255.0f);
                                if (v > 0.10f) { s += v; c++; }
                            }
                        ls[z0 + z] += s; lc[z0 + z] += c;
                    }
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

    /* ===================== OUTPUT zarr (same shape as input; chunk = tile) ===================== */
    long oshape[3] = { Z, Y, X };
    long ochunk[3] = { tile, tile, tile };
    fy_zarr zout;
    if (fy_zarr_create(&zout, out_root, oshape, ochunk) != 0) {
        fprintf(stderr, "pipeline: cannot create %s\n", out_root);
        free(zsums); free(zcnts); free(zfactor); return 1;
    }

    /* ===================== PASS 2: parallel per-tile ===================== */
    long ntz = ceil_div(Z, tile), nty = ceil_div(Y, tile), ntx = ceil_div(X, tile);
    long ntiles = ntz * nty * ntx;
    long out_done = 0;
    if (verbose) fprintf(stderr, "[pass2] %ld tiles, halo=%d\n", ntiles, halo);

    #pragma omp parallel
    {
        long bz_max = tile + 2 * halo;
        size_t cap = (size_t)bz_max * bz_max * bz_max;
        unsigned char *u8 = (unsigned char *)malloc(cap);
        float *f = (float *)malloc(sizeof(float) * cap);
        float *orig = (float *)malloc(sizeof(float) * cap);
        float *b1 = (float *)malloc(sizeof(float) * cap);
        float *b2 = (float *)malloc(sizeof(float) * cap);
        unsigned char *ob = (unsigned char *)malloc(cap);
        float *ws = (float *)malloc(sizeof(float) * 6 * cap);   /* guided-filter workspace, reused */

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
                         have_dec_range, dec_lo, dec_hi, b1, b2, ws);

            long iz0 = z0 - rz0, iy0 = y0 - ry0, ix0 = x0 - rx0;

            /* tile == output chunk: convert the inner tile to u8 and write the chunk directly. */
            long oi = 0;
            for (long zz = iz0; zz < iz0 + tz; zz++)
            for (long yy = iy0; yy < iy0 + ty; yy++)
            for (long xx = ix0; xx < ix0 + tx; xx++) {
                float v = f[((size_t)zz * hy + yy) * hx + xx] * 255.0f + 0.5f;
                ob[oi++] = v < 0 ? 0 : (v > 255 ? 255 : (unsigned char)v);
            }
            fy_zarr_write_chunk(&zout, iz, iy, ix, ob, tz, ty, tx);
            #pragma omp atomic
            out_done++;
            if (verbose && (out_done % 64 == 0)) {
                #pragma omp critical (prog)
                fprintf(stderr, "\r[pass2] %ld/%ld tiles", out_done, ntiles);
            }
        }
        free(u8); free(f); free(orig); free(b1); free(b2); free(ob); free(ws);
    }

    if (verbose) fprintf(stderr, "\n[done] %ld tiles, out shape %ldx%ldx%ld\n", out_done, Z, Y, X);
    free(zsums); free(zcnts); free(zfactor);
    return 0;
}
