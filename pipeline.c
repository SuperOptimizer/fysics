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
/* deterministic per-voxel dither in [0,1): integer hash of the GLOBAL voxel coordinate,
 * so the quantization is tiling-invariant (seam-free) and bit-reproducible. Used as
 * out = floor(v*255 + d) -- an unbiased dithered quantizer (E[out] = v*255), which kills
 * the contour banding that round-to-nearest leaves in flat papyrus regions. */
static inline float dither01(long z, long y, long x) {
    unsigned int h = (unsigned int)(z * 73856093L) ^ (unsigned int)(y * 19349663L)
                   ^ (unsigned int)(x * 83492791L);
    h ^= h >> 16; h *= 0x7feb352dU; h ^= h >> 15; h *= 0x846ca68bU; h ^= h >> 16;
    return (float)(h >> 8) * (1.0f / 16777216.0f);
}
/* quantize one processed voxel (f in [0,1] nominal) to u8 at GLOBAL coords (gz,gy,gx) */
static inline unsigned char quant_u8(float v, long gz, long gy, long gx, int no_dither) {
    float t = v * 255.0f + (no_dither ? 0.5f : dither01(gz, gy, gx));
    if (t <= 0.0f) return 0;
    int q = (int)t;                 /* floor for non-negative t */
    return q > 255 ? 255 : (unsigned char)q;
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
    /* measured PSF anisotropy (sigma_z/sigma_xy from noise autocorrelation): used
     * by the matched deconv to invert the broader z blur of helical scans. */
    p.psf_sigma_z_vox = (c->psf_z_ratio > 1.0 && c->psf_sigma_vox > 0)
        ? c->psf_z_ratio * c->psf_sigma_vox : 0.0;
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
/* guided_eps_tile <= 0 -> use cfg->guided_eps (the global calibration); a positive
 * value overrides it (the radially-varying eps from the fn(r) fit). */
static void process_tile(float *f, const float *orig, const unsigned char *u8orig,
                         int nz, int ny, int nx, const fy_pipeline_cfg *cfg,
                         int have_dec_range, double dec_lo, double dec_hi,
                         float *b1, float *b2, float *ws, double guided_eps_tile) {
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

    /* (c) GUIDED DENOISE (fast subsampled coefficients by default; eps is the global
     * calibration or, when the radial fn(r) fit is active, the per-tile value). */
    if (cfg->guided_eps > 0 || guided_eps_tile > 0) {
        double eps = guided_eps_tile > 0 ? guided_eps_tile : cfg->guided_eps;
        int gs = cfg->guided_subsample == 0 ? 2 : cfg->guided_subsample;
        if (fy_guided_denoise_fast_ws(f, b1, nz, ny, nx, 2, eps, gs, ws) == 0)
            memcpy(f, b1, sizeof(float) * N);
    }

    /* (d) AIR-ZERO: scratch smooth (box) -> local valley clamped to global+-band -> zero.
     * Box-smooth ORIG directly into b2 (no pre-copy, no copy-back -- box_mean isn't in==out
     * safe but orig!=b2). The smoothed scratch lives in b2, read only for the histogram. */
    if (cfg->do_air_zero) {
        int sr = cfg->scratch_passes > 0 ? cfg->scratch_passes : 5;
        long hist[256]; memset(hist, 0, sizeof(hist));
        /* The scratch is a heavy box smooth whose ONLY jobs are (a) tightening the
         * histogram modes for the valley decision and (b) a fuzzy threshold field.
         * Both are low-pass products, so compute it on a 2x DECIMATED copy (~8x less
         * box work, the former #1 hotspot) and compare through a trilinear upsample. */
        int lz = (nz + 1) / 2, ly = (ny + 1) / 2, lx = (nx + 1) / 2;
        size_t nl = (size_t)lz * ly * lx;
        int lowres = (lz >= 2 && ly >= 2 && lx >= 2);
        if (lowres) {
            for (int z = 0; z < lz; z++)
                for (int y = 0; y < ly; y++) {
                    const float *irow = orig + ((size_t)(z * 2) * ny + (size_t)(y * 2)) * nx;
                    float *lrow = b2 + ((size_t)z * ly + y) * lx;
                    for (int x = 0; x < lx; x++) lrow[x] = irow[(size_t)x * 2];
                }
            int rl = (sr + 1) / 2; if (rl < 1) rl = 1;
            fy_box_smooth(b2, b1, b1 + nl, lz, ly, lx, rl);   /* smooth -> b1[0..nl) */
            for (size_t i = 0; i < nl; i++) {
                int b = (int)(b1[i] * 255.0f + 0.5f); if (b < 0) b = 0; if (b > 255) b = 255; hist[b]++;
            }
        } else {
            fy_box_smooth(orig, b2, b1, nz, ny, nx, sr);   /* b1 = tmp */
            for (size_t i = 0; i < N; i++) {
                int b = (int)(b2[i] * 255.0f + 0.5f); if (b < 0) b = 0; if (b > 255) b = 255; hist[b]++;
            }
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
        if (lowres) {
            /* compare against the trilinearly upsampled low-res smooth (4 low rows
             * blended once per output row; b2's decimated copy is dead -> row scratch) */
            const float *sl = b1;
            float *rowbuf = b2;
            for (int z = 0; z < nz; z++) {
                int z0 = z >> 1; if (z0 > lz - 2) z0 = lz - 2;
                float zf = 0.5f * z - z0; if (zf > 1.0f) zf = 1.0f;
                for (int y = 0; y < ny; y++) {
                    int y0 = y >> 1; if (y0 > ly - 2) y0 = ly - 2;
                    float yf = 0.5f * y - y0; if (yf > 1.0f) yf = 1.0f;
                    size_t r00 = ((size_t)z0 * ly + y0) * lx, r01 = r00 + lx;
                    size_t r10 = r00 + (size_t)ly * lx, r11 = r10 + lx;
                    float w00 = (1 - zf) * (1 - yf), w01 = (1 - zf) * yf;
                    float w10 = zf * (1 - yf), w11 = zf * yf;
                    for (int k = 0; k < lx; k++)
                        rowbuf[k] = w00 * sl[r00 + k] + w01 * sl[r01 + k] +
                                    w10 * sl[r10 + k] + w11 * sl[r11 + k];
                    float *frow = f + ((size_t)z * ny + y) * nx;
                    for (int x = 0; x < nx; x++) {
                        int x0 = x >> 1; if (x0 > lx - 2) x0 = lx - 2;
                        float fx = 0.5f * x - x0; if (fx > 1.0f) fx = 1.0f;
                        float v = rowbuf[x0] + (rowbuf[x0 + 1] - rowbuf[x0]) * fx;
                        if (v < cutf) frow[x] = 0.0f;
                    }
                }
            }
        } else {
            for (size_t i = 0; i < N; i++) if (b2[i] < cutf) f[i] = 0.0f;
        }
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

/* per-tile guided eps from the radial fn(r) fit (tile center vs rotation axis);
 * 0 -> caller uses the global cfg->guided_eps. */
static double eps_for_tile(const fy_pipeline_cfg *cfg, double ecy, double ecx,
                           long y0, long x0, long ty, long tx) {
    if (!cfg->have_eps_r || cfg->guided_eps <= 0 || cfg->eps_fn_med <= 0) return 0.0;
    double ry = y0 + ty / 2.0 - ecy, rx = x0 + tx / 2.0 - ecx;
    double fn = cfg->eps_fn_a + cfg->eps_fn_b * sqrt(ry * ry + rx * rx);
    double s = fn / cfg->eps_fn_med, s2 = s * s;
    if (s2 < 0.4) s2 = 0.4;
    if (s2 > 8.0) s2 = 8.0;
    return cfg->guided_eps * s2;
}

/* ============================================================================
 * fy_process_buffer -- the I/O-FREE half of fy_process_chunk: process ONE inner
 * tile from an ALREADY-READ halo'd u8 buffer. For export pipelines that own
 * their I/O (e.g. the streaming mca_export downloader pool). `u8buf` is the
 * hz*hy*hx region whose global origin is (rz0,ry0,rx0); the inner tile starts
 * at global (z0,y0,x0) and spans (tz,ty,tx) (caller clamps both). vol_y/vol_x
 * are the FULL volume Y/X (for the radial-eps / dering center). Thread-safe
 * (allocates its own scratch). Writes tz*ty*tx u8 to out; sets *all_air.
 * ========================================================================== */
int fy_process_buffer(const fy_pipeline_cfg *cfg, const unsigned char *u8buf,
                      long rz0, long ry0, long rx0, long hz, long hy, long hx,
                      long z0, long y0, long x0, long tz, long ty, long tx,
                      long vol_y, long vol_x,
                      unsigned char *out, int *all_air) {
    size_t hn = (size_t)hz * hy * hx;
    if (!any_nonzero(u8buf, hn)) {
        memset(out, 0, (size_t)tz * ty * tx);
        if (all_air) *all_air = 1;
        return 0;
    }
    float *f = malloc(sizeof(float)*hn), *orig = malloc(sizeof(float)*hn);
    float *b1 = malloc(sizeof(float)*hn), *b2 = malloc(sizeof(float)*hn);
    float *ws = malloc(sizeof(float)*4*hn);
    if (!f||!orig||!b1||!b2||!ws){ free(f);free(orig);free(b1);free(b2);free(ws); return 1; }
    if (cfg->have_norm) fy_norm_apply_u8(u8buf, f, hn, (unsigned char)cfg->norm_lo, (unsigned char)cfg->norm_hi);
    else u8_to_f01(u8buf, f, hn);
    if (cfg->have_dering && cfg->dering) fy_dering_apply(cfg->dering, f, rz0, ry0, rx0, hz, hy, hx,
        cfg->have_norm ? 1.0 / (cfg->norm_hi - cfg->norm_lo) : 1.0 / 255.0);
    if (cfg->have_zdrift && cfg->zdrift_factor) fy_zdrift_apply(f, (int)hz, (int)hy, (int)hx, (int)rz0, cfg->zdrift_factor);
    memcpy(orig, f, sizeof(float) * hn);   /* (fused conversions tried; the dering/zdrift
                                            * paths need f finalized first, so the copy stays) */
    { double ecy = cfg->dering_cy > 0 ? cfg->dering_cy : (vol_y - 1) / 2.0;
      double ecx = cfg->dering_cx > 0 ? cfg->dering_cx : (vol_x - 1) / 2.0;
      process_tile(f, orig, u8buf, (int)hz, (int)hy, (int)hx, cfg,
                   cfg->have_dec_range, cfg->dec_lo, cfg->dec_hi, b1, b2, ws,
                   eps_for_tile(cfg, ecy, ecx, y0, x0, ty, tx)); }
    { long iz0 = z0 - rz0, iy0 = y0 - ry0, ix0 = x0 - rx0, oi = 0;
      for (long zz = iz0; zz < iz0 + tz; zz++)
      for (long yy = iy0; yy < iy0 + ty; yy++)
      for (long xx = ix0; xx < ix0 + tx; xx++)
          out[oi++] = quant_u8(f[((size_t)zz * hy + yy) * hx + xx],
                               rz0 + zz, ry0 + yy, rx0 + xx, cfg->no_dither); }
    if (all_air) *all_air = 0;
    free(f); free(orig); free(b1); free(b2); free(ws);
    return 0;
}

/* ============================================================================
 * fy_process_chunk -- process ONE inner tile at (z0,y0,x0) with a halo'd read,
 * using the already-calibrated cfg. The pass-2 body, factored out so the FUSED
 * v2/v3 export can pull preprocessed chunks on demand. Allocates its own scratch
 * (self-contained / thread-safe). Writes the inner tile (tz*ty*tx u8) to `out`.
 * ========================================================================== */
int fy_process_chunk(const fy_zarr *zin, const fy_pipeline_cfg *cfg,
                     long z0, long y0, long x0, int tile,
                     unsigned char *out, int *out_tz, int *out_ty, int *out_tx, int *all_air) {
    long Z = zin->shape[0], Y = zin->shape[1], X = zin->shape[2];
    int halo = cfg->halo > 0 ? cfg->halo : 8;
    long tz = lmin(tile, Z - z0), ty = lmin(tile, Y - y0), tx = lmin(tile, X - x0);
    if (out_tz) *out_tz = (int)tz; if (out_ty) *out_ty = (int)ty; if (out_tx) *out_tx = (int)tx;
    long rz0 = lmax(0, z0 - halo), ry0 = lmax(0, y0 - halo), rx0 = lmax(0, x0 - halo);
    long rz1 = lmin(Z, z0 + tz + halo), ry1 = lmin(Y, y0 + ty + halo), rx1 = lmin(X, x0 + tx + halo);
    long hz = rz1 - rz0, hy = ry1 - ry0, hx = rx1 - rx0;
    unsigned char *u8 = malloc((size_t)hz * hy * hx);
    if (!u8) return 1;
    int rc = 0; int air = 1;
    if (fy_zarr_read(zin, rz0, ry0, rx0, hz, hy, hx, u8) != 0) { rc = 1; goto done; }
    rc = fy_process_buffer(cfg, u8, rz0, ry0, rx0, hz, hy, hx,
                           z0, y0, x0, tz, ty, tx, Y, X, out, &air);
done:
    if (all_air) *all_air = air;
    free(u8);
    return rc;
}

/* Calibrate the pipeline on `in_root` (PSF/eps/air-cut/gates + norm/zdrift global stats), filling
 * cfg (incl. zdrift_factor, which the caller must free). Implemented as fy_run_pipeline with a NULL
 * output (calibrate-only early return). For the fused export: call once, then fy_process_chunk. */
int fy_calibrate(const char *in_root, fy_pipeline_cfg *cfg, int tile, int verbose) {
    return fy_run_pipeline(in_root, NULL, cfg, tile, verbose);
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

    /* ===================== PASS 1: UNIFIED CALIBRATION SWEEP =====================
     * ONE parallel pass over budget-sampled 256^3 ptiles accumulates EVERYTHING:
     *   - streaming (cheap, every sampled ptile): norm histogram, per-z z-drift
     *     sums, dering ring profiles
     *   - heavy stats (capped at HEAVY_MAX sub-blocks volume-wide): flat-noise +
     *     radius pairs, PSF sigma + anisotropy, scratch-denoised air histogram --
     *     measured on the CENTRAL 128^3 sub-block of qualifying (>=60% occupancy)
     *     ptiles
     *   - up to KEEP_MAX qualifying sub-block COPIES retained for the post-sweep
     *     deconv-output-range measurement (the AUTO-DECONV gate is decided AFTER
     *     the sweep from the accumulated PSF sigmas -- old pre-pass thresholds --
     *     and must precede the halo/deconv setup).
     */
    enum { CMAX = 256, SUB = 128, KEEP_MAX = 32, HEAVY_MAX = 256 };
    long air_hist[256]; memset(air_hist, 0, sizeof(air_hist));
    double floors[CMAX]; int nfloor = 0, nsamp = 0;
    double rads[CMAX];                   /* sub-block-center radius paired with floors[] */
    double sigmas[CMAX]; int nsig = 0;
    double anisos[CMAX]; int nani = 0;   /* measured sigma_z/sigma_xy per sample sub-block */
    double ecy = cfg->dering_cy > 0 ? cfg->dering_cy : (Y - 1) / 2.0;   /* rotation axis */
    double ecx = cfg->dering_cx > 0 ? cfg->dering_cx : (X - 1) / 2.0;
    unsigned char *keep[KEEP_MAX]; int keep_d[KEEP_MAX][3]; int nkeep = 0;
    int nheavy = 0;                      /* atomic cap on heavy-stat sub-blocks */

    fy_hist_state nhist; fy_hist_init(&nhist);
    double *zsums = (double *)calloc(Z, sizeof(double));
    long *zcnts = (long *)calloc(Z, sizeof(long));
    float *zfactor = NULL;
    int have_norm = 0, have_zdrift = 0, have_dering = 0;
    fy_dering *der = NULL;

    long ptile = tile > 256 ? tile : 256;
    long pnz = ceil_div(Z, ptile), pny = ceil_div(Y, ptile), pnx = ceil_div(X, ptile);
    long pntiles = pnz * pny * pnx;
    if (cfg->do_dering) {
        der = (fy_dering *)malloc(sizeof(fy_dering));
        /* slab = 2 sweep z-tiles, so each tile lands in exactly ONE slab (lock-free merge) */
        if (!der || fy_dering_init(der, Z, Y, X, cfg->dering_cy > 0 ? cfg->dering_cy : -1,
                                   cfg->dering_cx > 0 ? cfg->dering_cx : -1,
                                   (int)(2 * ptile), 8) != 0) {
            free(der); der = NULL;
        }
    }
    /* S3 sweep: 2x ncpu threads (the allowed pool oversubscription cap) -- each
     * thread alternates batched-fetch / accumulate, so 2x covers the dead air
     * between batches; fy_zarr_read multiplexes the GETs within each call. */
    int sweep_threads = strncmp(in_root, "s3://", 5) == 0 ? 2 * omp_get_max_threads()
                                                          : omp_get_max_threads();
    #pragma omp parallel num_threads(sweep_threads)
    {
        fy_hist_state lh; fy_hist_init(&lh);
        double *ls = (double *)calloc(Z, sizeof(double));
        long *lc = (long *)calloc(Z, sizeof(long));
        unsigned char *pb = (unsigned char *)malloc((size_t)ptile * ptile * ptile);
        /* heavy-stat scratch (sub-block sized), allocated lazily on first use */
        size_t scap = (size_t)SUB * SUB * SUB;
        unsigned char *sub = NULL;
        float *sf = NULL, *so = NULL, *stmp = NULL, *sws = NULL;
        fy_dering dt; int have_dt = der && fy_dering_tile_init(&dt, der) == 0;
        #pragma omp for schedule(dynamic)
        for (long t = 0; t < pntiles; t++) {
            long iz = t / (pny * pnx), r = t % (pny * pnx), iy = r / pnx, ix = r % pnx;
            /* SAMPLE: read EVERY z-slab (the z-drift trend needs full z-coverage) but only
             * a hash-selected fraction of (y,x) tiles, sized by a CALIBRATION I/O BUDGET
             * (default 200 GB, cfg->calib_budget_gb) -- on a 27 TB volume the old strided
             * 25% sweep was ~7 TB of reads before processing could start. The stats
             * (norm histogram, per-z papyrus mean, ring profiles) converge on samples;
             * a floor keeps >=~32 tiles expected per z-slab for the z-drift profile. */
            if (pny >= 4 || pnx >= 4) {
                double total_gb = (double)Z * Y * X / 1e9;
                double budget = cfg->calib_budget_gb > 0 ? cfg->calib_budget_gb : 200.0;
                double frac = total_gb > budget ? budget / total_gb : 1.0;
                double floorf_ = 32.0 / ((double)pny * pnx);
                if (frac < floorf_) frac = floorf_;
                if (frac < 1.0) {
                    unsigned int hsh = (unsigned int)(iz * 73856093L) ^ (unsigned int)(iy * 19349663L)
                                     ^ (unsigned int)(ix * 83492791L);
                    hsh ^= hsh >> 16; hsh *= 0x7feb352dU; hsh ^= hsh >> 15; hsh *= 0x846ca68bU; hsh ^= hsh >> 16;
                    if (hsh > (unsigned int)(frac * 4294967295.0)) continue;
                }
            }
            long z0 = iz * ptile, y0 = iy * ptile, x0 = ix * ptile;
            long dz = lmin(ptile, Z - z0), dy = lmin(ptile, Y - y0), dx = lmin(ptile, X - x0);
            size_t n = (size_t)dz * dy * dx;
            if (fy_zarr_read(&zin, z0, y0, x0, dz, dy, dx, pb) != 0) continue;
            if (!any_nonzero(pb, n)) continue;
            if (cfg->do_normalize) fy_hist_accumulate_u8(&lh, pb, n);
            if (have_dt) {
                fy_dering_tile_reset(&dt);
                fy_dering_accumulate_u8(&dt, pb, y0, x0, dz, dy, dx, 2);
                #pragma omp critical (dering_merge)
                fy_dering_merge_tile(der, (int)(z0 / der->slab_z), &dt);
            }
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
            /* HEAVY calibration stats on the CENTRAL SUB^3 sub-block of this ptile,
             * capped at HEAVY_MAX qualifying sub-blocks volume-wide (atomic claim). */
            { int hv;
              #pragma omp atomic read
              hv = nheavy;
              if (hv >= HEAVY_MAX) continue; }
            if (!sub) {
                sub  = (unsigned char *)malloc(scap);
                sf   = (float *)malloc(sizeof(float) * scap);
                so   = (float *)malloc(sizeof(float) * scap);
                stmp = (float *)malloc(sizeof(float) * scap);
                sws  = (float *)malloc(sizeof(float) * 4 * scap);
                if (!sub || !sf || !so || !stmp || !sws) {
                    free(sub); free(sf); free(so); free(stmp); free(sws);
                    sub = NULL; sf = so = stmp = sws = NULL;
                    continue;
                }
            }
            long sdz = lmin(SUB, dz), sdy = lmin(SUB, dy), sdx = lmin(SUB, dx);
            long oz = (dz - sdz) / 2, oy = (dy - sdy) / 2, ox = (dx - sdx) / 2;
            size_t sn = (size_t)sdz * sdy * sdx;
            for (long zz = 0; zz < sdz; zz++)
                for (long yy = 0; yy < sdy; yy++)
                    memcpy(sub + ((size_t)zz * sdy + yy) * sdx,
                           pb + (((size_t)(oz + zz) * dy + (oy + yy)) * dx + ox), (size_t)sdx);
            { size_t occ = 0; for (size_t i = 0; i < sn; i++) if (sub[i] > 0) occ++;
              if ((double)occ / sn < 0.6) continue; }
            { int my;
              #pragma omp atomic capture
              { my = nheavy; nheavy++; }
              if (my >= HEAVY_MAX) continue; }
            u8_to_f01(sub, sf, sn);
            double fn = fy_flat_noise(sf, (int)sdz, (int)sdy, (int)sdx, 8);
            double aniso = -1, arz, arxy;
            if (fy_noise_aniso(sf, (int)sdz, (int)sdy, (int)sdx, &arz, &arxy) == 0 &&
                arz > 0.05 && arz < 0.95 && arxy > 0.05 && arxy < 0.95) {
                double ratio = sqrt(log(1.0 / arxy) / log(1.0 / arz));
                if (ratio > 0.5 && ratio < 3.0) aniso = ratio;
            }
            long lhist[256]; memset(lhist, 0, sizeof(lhist));
            if (cfg->do_air_zero) {
                memcpy(so, sf, sizeof(float) * sn);
                scratch_denoise(so, (int)sdz, (int)sdy, (int)sdx,
                                cfg->scratch_passes > 0 ? cfg->scratch_passes : 5, stmp, sws);
                for (size_t i = 0; i < sn; i++) {
                    int b = (int)(so[i] * 255.0f + 0.5f); if (b < 0) b = 0; if (b > 255) b = 255; lhist[b]++;
                }
            }
            double sg = measure_tile_psf_sigma(sub, (int)sdz, (int)sdy, (int)sdx);
            #pragma omp critical (calib_merge)
            {
                if (fn > 0 && nfloor < CMAX) {
                    double ry = y0 + oy + sdy / 2.0 - ecy, rx = x0 + ox + sdx / 2.0 - ecx;
                    rads[nfloor] = sqrt(ry * ry + rx * rx);
                    floors[nfloor++] = fn;
                }
                if (aniso > 0 && nani < CMAX) anisos[nani++] = aniso;
                for (int b = 0; b < 256; b++) air_hist[b] += lhist[b];
                if (sg > 0 && nsig < CMAX) sigmas[nsig++] = sg;
                if (nkeep < KEEP_MAX) {       /* retain a COPY for the deconv-range step */
                    unsigned char *cp = (unsigned char *)malloc(sn);
                    if (cp) {
                        memcpy(cp, sub, sn);
                        keep[nkeep] = cp;
                        keep_d[nkeep][0] = (int)sdz; keep_d[nkeep][1] = (int)sdy; keep_d[nkeep][2] = (int)sdx;
                        nkeep++;
                    }
                }
            }
            #pragma omp atomic
            nsamp++;
        }
        #pragma omp critical
        {
            if (cfg->do_normalize) fy_hist_merge(&nhist, &lh);
            for (long z = 0; z < Z; z++) { zsums[z] += ls[z]; zcnts[z] += lc[z]; }
        }
        if (have_dt) fy_dering_free(&dt);
        free(ls); free(lc); free(pb);
        free(sub); free(sf); free(so); free(stmp); free(sws);
    }

    /* commit PSF median + AUTO-DECONV gate FIRST, so it is decided BEFORE the
     * halo/sdec/rescale setup (deconv reach drives the halo; the gate must precede it). */
    if (nsig >= 3) {
        qsort(sigmas, nsig, sizeof(double), cmp_double);
        double med = percentile_sorted(sigmas, nsig, 50.0);
        cfg->psf_med = med; cfg->psf_p5 = percentile_sorted(sigmas, nsig, 5.0);
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

    /* halo: kernel halo from physics (deconv reach); air-zero/denoise are local r=2. */
    fy_physics base_ph = phys_from_cfg(cfg);
    int halo = cfg->do_deconv ? fy_kernel_halo(&base_ph) : 8;
    if (halo < 8) halo = 8;
    cfg->halo = halo;

    /* commit deconv global range (seam-safe rescale) over the RETAINED sub-blocks --
     * measured in the SAME windowed space as pass-2: de-window (u8/255 -> mu), deconv,
     * re-window, collect [0,1] values. Runs BEFORE the psf_z_ratio commit below, so these
     * sample deconvs are isotropic -- same as the old pass-1 sampling (only pass 2 uses
     * the ratio; a negligible inconsistency in the dec-range percentiles). */
    int have_dec_range = 0; double dec_lo = 0.0, dec_hi = 1.0;
    if (cfg->do_deconv && nkeep > 0) {
        size_t scap = (size_t)SUB * SUB * SUB;
        double *dec_vals = NULL; size_t dec_n = 0, dec_cap = 0;
        float *df = (float *)malloc(sizeof(float) * scap);
        float *ddec = (float *)malloc(sizeof(float) * scap);
        if (df && ddec) {
            fy_physics ph = scaled_phys_from_cfg(cfg);
            int windowed = (cfg->window_hi > cfg->window_lo);
            float wlo = (float)cfg->window_lo, wspan = (float)(cfg->window_hi - cfg->window_lo);
            for (int k = 0; k < nkeep; k++) {
                int kz = keep_d[k][0], ky = keep_d[k][1], kx = keep_d[k][2];
                size_t n = (size_t)kz * ky * kx;
                u8_to_f01(keep[k], df, n);
                if (windowed) for (size_t i = 0; i < n; i++) df[i] = wlo + df[i] * wspan;
                int rc = (cfg->use_matched_deconv && cfg->psf_sigma_vox > 0)
                    ? fy_deconvolve_matched(df, ddec, kz, ky, kx, &ph, cfg->deconv_tikhonov)
                    : fy_deconvolve(df, ddec, kz, ky, kx, &ph, -1.0);
                if (rc != 0) continue;
                if (dec_n + n > dec_cap) { dec_cap = (dec_n + n) * 2; dec_vals = realloc(dec_vals, sizeof(double) * dec_cap); }
                for (size_t i = 0; i < n; i++)
                    dec_vals[dec_n++] = windowed ? (ddec[i] - wlo) / wspan : ddec[i];
            }
        }
        free(df); free(ddec);
        if (dec_n > 0) {
            qsort(dec_vals, dec_n, sizeof(double), cmp_double);
            dec_lo = percentile_sorted(dec_vals, dec_n, 0.1);
            dec_hi = percentile_sorted(dec_vals, dec_n, 99.9);
            if (dec_hi - dec_lo < 1e-6) { dec_lo = 0.0; dec_hi = 1.0; }
            have_dec_range = 1;
        }
        free(dec_vals);
    }
    for (int k = 0; k < nkeep; k++) free(keep[k]);

    /* commit guided_eps = (k * median flat_nf)^2, k = cfg->denoise_k (profile: conservative 3.0
     * ~0.002 preserves detail, aggressive 4.2 ~0.004). Default 4.2 if unset. Unless pre-set. */
    /* RADIAL eps fit FIRST (median_sorted mutates floors[]): flat-noise rises toward the
     * rim on large scrolls (measured 3x center->edge at fixed intensity on PHercParis3 --
     * half-acquisition coverage + path length). Least-squares fn(r) = a + b*r over the
     * sample tiles, gated on enough samples, positive slope, decent correlation, and a
     * meaningful span; otherwise the single global eps stands. */
    cfg->have_eps_r = 0;
    if (nfloor >= 8) {
        double sr = 0, sf2 = 0, srr = 0, srf = 0, sff = 0;
        for (int i = 0; i < nfloor; i++) {
            sr += rads[i]; sf2 += floors[i];
            srr += rads[i] * rads[i]; srf += rads[i] * floors[i]; sff += floors[i] * floors[i];
        }
        double nn = nfloor;
        double den = nn * srr - sr * sr;
        if (den > 1e-9) {
            double b = (nn * srf - sr * sf2) / den;
            double a = (sf2 - b * sr) / nn;
            double cden = sqrt((nn * srr - sr * sr) * (nn * sff - sf2 * sf2));
            double corr = cden > 1e-12 ? (nn * srf - sr * sf2) / cden : 0;
            double rmax = 0; for (int i = 0; i < nfloor; i++) if (rads[i] > rmax) rmax = rads[i];
            double f0 = a, f1 = a + b * rmax;
            if (b > 0 && corr >= 0.5 && f0 > 0 && f1 / f0 >= 1.3) {
                cfg->eps_fn_a = a; cfg->eps_fn_b = b; cfg->have_eps_r = 1;
                if (verbose) fprintf(stderr, "[calib] radial eps ON: fn %.4f -> %.4f (corr %.2f)\n", f0, f1, corr);
            }
        }
    }
    double flat_nf_med = nfloor ? median_sorted(floors, nfloor) : 0.015;
    cfg->eps_fn_med = flat_nf_med;
    if (cfg->guided_eps <= 0) {
        double k = cfg->denoise_k > 0 ? cfg->denoise_k : 4.2;
        cfg->guided_eps = (k * flat_nf_med) * (k * flat_nf_med);
    }
    /* commit measured PSF anisotropy (median over sample sub-blocks, gated + clamped).
     * NOTE: the dec-range sample deconvs above ran isotropic; only pass 2 uses the ratio. */
    if (nani >= 5) {
        double ratio = median_sorted(anisos, nani);
        if (ratio >= 1.05) {
            cfg->psf_z_ratio = ratio < 1.6 ? ratio : 1.6;
            if (verbose) fprintf(stderr, "[calib] PSF anisotropy: sigma_z/sigma_xy = %.2f\n", cfg->psf_z_ratio);
        }
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
    /* (downsample removed -- the volume is not meaningfully oversampled; the auto-factor rested on
     * an unreliable PSF sigma. PSF is still measured above, but only for the auto-deconv gate.) */
    if (verbose) fprintf(stderr, "[calib] samples=%d flat_nf_n=%d guided_eps=%.5f air_cut=%d band=%d psf_med=%.2f deconv=%d halo=%d\n",
                         nsamp, nfloor, cfg->guided_eps, cfg->air_cut_u8, cfg->air_cut_band, cfg->psf_med, cfg->do_deconv, halo);

    /* dering gate: only keep rings that pass the sector vote (>=2 radii detected) */
    if (der) {
        long ndet = fy_dering_finalize(der, 15, 100, 0.5, 6.0);
        if (der->have_rings) have_dering = 1;
        else { fy_dering_free(der); free(der); der = NULL; }
        if (verbose) fprintf(stderr, "[pass1] dering: %ld ring radii detected -> %s\n",
                             ndet, have_dering ? "subtracting" : "skipped");
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

    /* publish resolved calibration STATE into cfg so fy_process_chunk (fused export) can use it. */
    cfg->have_norm = have_norm; cfg->have_zdrift = have_zdrift;
    cfg->have_dec_range = have_dec_range; cfg->dec_lo = dec_lo; cfg->dec_hi = dec_hi;
    cfg->zdrift_factor = zfactor; cfg->vol_z = Z; cfg->halo = halo;
    cfg->have_dering = have_dering; cfg->dering = der;

    /* CALIBRATE-ONLY mode (out_root==NULL): used by fy_calibrate for the fused export. Keep zfactor
     * alive in cfg (caller owns it); free the histogram accumulators. Do NOT free zfactor here. */
    if (out_root == NULL) { free(zsums); free(zcnts); return 0; }

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
    long out_done = 0, io_fail = 0;
    double ecy2 = cfg->dering_cy > 0 ? cfg->dering_cy : (Y - 1) / 2.0;   /* radial-eps center */
    double ecx2 = cfg->dering_cx > 0 ? cfg->dering_cx : (X - 1) / 2.0;
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
        float *ws = (float *)malloc(sizeof(float) * 4 * cap);   /* guided-filter workspace, reused */

        #pragma omp for schedule(dynamic)
        for (long t = 0; t < ntiles; t++) {
            long iz = t / (nty * ntx), r = t % (nty * ntx), iy = r / ntx, ix = r % ntx;
            long z0 = iz * tile, y0 = iy * tile, x0 = ix * tile;
            long tz = lmin(tile, Z - z0), ty = lmin(tile, Y - y0), tx = lmin(tile, X - x0);
            long rz0 = lmax(0, z0 - halo), ry0 = lmax(0, y0 - halo), rx0 = lmax(0, x0 - halo);
            long rz1 = lmin(Z, z0 + tz + halo), ry1 = lmin(Y, y0 + ty + halo), rx1 = lmin(X, x0 + tx + halo);
            long hz = rz1 - rz0, hy = ry1 - ry0, hx = rx1 - rx0;
            size_t hn = (size_t)hz * hy * hx;
            if (fy_zarr_read(&zin, rz0, ry0, rx0, hz, hy, hx, u8) != 0) {
                /* a failed read MUST NOT silently leave a fill-value hole in the output */
                #pragma omp atomic
                io_fail++;
                continue;
            }
            if (!any_nonzero(u8, hn)) continue;

            /* (a) INTENSITY CORRECTIONS FIRST: normalize, dering, then z-drift (gated). */
            if (have_norm) fy_norm_apply_u8(u8, f, hn, (unsigned char)cfg->norm_lo, (unsigned char)cfg->norm_hi);
            else u8_to_f01(u8, f, hn);
            if (have_dering) fy_dering_apply(der, f, rz0, ry0, rx0, hz, hy, hx,
                have_norm ? 1.0 / (cfg->norm_hi - cfg->norm_lo) : 1.0 / 255.0);
            if (have_zdrift) fy_zdrift_apply(f, (int)hz, (int)hy, (int)hx, (int)rz0, zfactor);
            memcpy(orig, f, sizeof(float) * hn);

            /* (b-e) per-tile chain. */
            process_tile(f, orig, u8, (int)hz, (int)hy, (int)hx, cfg,
                         have_dec_range, dec_lo, dec_hi, b1, b2, ws,
                         eps_for_tile(cfg, ecy2, ecx2, y0, x0, ty, tx));

            long iz0 = z0 - rz0, iy0 = y0 - ry0, ix0 = x0 - rx0;

            /* tile == output chunk: convert the inner tile to u8 (dithered, global-coord
             * keyed -> seam-free) and write the chunk directly. */
            long oi = 0;
            for (long zz = iz0; zz < iz0 + tz; zz++)
            for (long yy = iy0; yy < iy0 + ty; yy++)
            for (long xx = ix0; xx < ix0 + tx; xx++)
                ob[oi++] = quant_u8(f[((size_t)zz * hy + yy) * hx + xx],
                                    rz0 + zz, ry0 + yy, rx0 + xx, cfg->no_dither);
            if (fy_zarr_write_chunk(&zout, iz, iy, ix, ob, tz, ty, tx) != 0) {
                #pragma omp atomic
                io_fail++;
            }
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
    if (der) { fy_dering_free(der); free(der); cfg->dering = NULL; }
    if (io_fail > 0) {
        fprintf(stderr, "[pass2] ERROR: %ld tile read/write failures -- output is INCOMPLETE\n", io_fail);
        return 1;
    }
    return 0;
}
