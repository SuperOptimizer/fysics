/* stream.c -- streaming/chunked primitives for WHOLE-VOLUME processing at 20TB+.
 *
 * fysics does the per-chunk MATH; the caller (e.g. vc3d or a driver using fast zarr
 * I/O) owns chunk iteration and I/O. Operators split into two kinds:
 *
 *   LOCAL (deconv, denoise, mask): process one chunk (+halo) independently. Use the
 *     existing fy_deconvolve / fy_bilateral_denoise / fy_papyrus_mask per chunk.
 *
 *   GLOBAL (normalization, GLCAE global stage): need whole-volume statistics, so
 *     they're TWO-PASS:
 *       pass 1: stream every chunk -> fy_*_accumulate() into a small state struct
 *       finalize:                   -> compute the global mapping ONCE
 *       pass 2: stream every chunk -> fy_*_apply() using that mapping
 *     The state is tiny (a histogram), so 20TB never sits in RAM.
 *
 * Everything here is u8-friendly: helpers convert u8<->float, since the real
 * volumes are dense u8.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define L 256

/* ---------- u8 <-> float ---------- */
void fy_u8_to_float(const unsigned char *in, float *out, size_t n) {
    const float inv = 1.0f / 255.0f;
    for (size_t i = 0; i < n; i++) out[i] = in[i] * inv;
}
void fy_float_to_u8(const float *in, unsigned char *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = in[i] * 255.0f + 0.5f;
        out[i] = v < 0 ? 0 : (v > 255 ? 255 : (unsigned char)v);
    }
}

/* ---- recover physical attenuation from the per-volume u8 export window ----
 * Exact linear inverse of nabu's export windowing (see fysics.h). Per-voxel and
 * streaming-friendly: no global state, process any chunk independently. */
void fy_u8_to_phys(const unsigned char *in, float *out, size_t n,
                   double f32_min, double f32_max) {
    float lo = (float)f32_min;
    float scale = (float)((f32_max - f32_min) / 255.0);
    for (size_t i = 0; i < n; i++) out[i] = lo + in[i] * scale;
}

void fy_phys_to_u8(const float *in, unsigned char *out, size_t n,
                   double f32_min, double f32_max) {
    double span = f32_max - f32_min;
    float inv = (span != 0.0) ? (float)(255.0 / span) : 0.0f;
    float lo = (float)f32_min;
    for (size_t i = 0; i < n; i++) {
        float v = (in[i] - lo) * inv + 0.5f;
        out[i] = v < 0 ? 0 : (v > 255 ? 255 : (unsigned char)v);
    }
}

float fy_phys_to_u8_level(double phys, double f32_min, double f32_max) {
    double span = f32_max - f32_min;
    if (span == 0.0) return 0.0f;
    return (float)((phys - f32_min) / span * 255.0);
}

/* ================= GLOBAL: histogram-based ops (two-pass) ================= */
/* State accumulates a 256-bin histogram of u8 values over the whole volume.
 * Tiny + mergeable, so passes can even be parallelized (accumulate per thread,
 * then sum the histograms). */

void fy_hist_init(fy_hist_state *s) {
    memset(s->hist, 0, sizeof(s->hist));
    s->total = 0;
}

/* pass 1: call per chunk (u8) */
void fy_hist_accumulate_u8(fy_hist_state *s, const unsigned char *chunk, size_t n) {
    for (size_t i = 0; i < n; i++) s->hist[chunk[i]]++;
    s->total += n;
}

/* merge two states (for parallel accumulation) */
void fy_hist_merge(fy_hist_state *dst, const fy_hist_state *src) {
    for (int i = 0; i < L; i++) dst->hist[i] += src->hist[i];
    dst->total += src->total;
}

/* finalize: percentile value (0..255) from the global histogram */
int fy_hist_percentile_u8(const fy_hist_state *s, double pct) {
    long target = (long)(pct / 100.0 * (double)s->total);
    long acc = 0;
    for (int i = 0; i < L; i++) { acc += s->hist[i]; if (acc >= target) return i; }
    return L - 1;
}

/* ---- GLOBAL normalization (consistent across the whole volume) ---- */
/* finalize: compute lo/hi (u8) from global percentiles; pass 2 applies per chunk */
void fy_norm_finalize(const fy_hist_state *s, double lo_pct, double hi_pct,
                      unsigned char *lo_out, unsigned char *hi_out) {
    *lo_out = (unsigned char)fy_hist_percentile_u8(s, lo_pct);
    *hi_out = (unsigned char)fy_hist_percentile_u8(s, hi_pct);
}
/* pass 2: normalize a u8 chunk to [0,1] float using the GLOBAL lo/hi */
void fy_norm_apply_u8(const unsigned char *in, float *out, size_t n,
                      unsigned char lo, unsigned char hi) {
    float flo = lo, inv = (hi > lo) ? 1.0f / (float)(hi - lo) : 1.0f;
    for (size_t i = 0; i < n; i++) {
        float v = (in[i] - flo) * inv;
        out[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
}

/* ---- GLOBAL GLCAE mapping (the global stage, computed once) ----
 * The global GLCAE stage is exactly a histogram -> lambda-blend -> CDF mapping.
 * We compute it from the WHOLE-VOLUME histogram in finalize, then apply the same
 * 256-entry lookup to every chunk in pass 2. (Defined in glcae.c via the shared
 * global_mapping logic exposed below.) */
extern void fy_global_glcae_mapping_from_hist(const long *hist, long total, int *t_out);

void fy_glcae_global_finalize(const fy_hist_state *s, int *mapping_out) {
    fy_global_glcae_mapping_from_hist(s->hist, s->total, mapping_out);
}
/* pass 2: apply the global GLCAE lookup to a u8 chunk -> float [0,1] */
void fy_glcae_global_apply_u8(const unsigned char *in, float *out, size_t n,
                              const int *mapping) {
    float inv = 1.0f / (float)(L - 1);
    for (size_t i = 0; i < n; i++) out[i] = mapping[in[i]] * inv;
}

/* ---- auto air/papyrus threshold (Otsu) from the global histogram ----
 * Replaces a hardcoded threshold: finds the intensity that best separates the two
 * modes (air vs papyrus) by maximizing between-class variance (Otsu's method) over
 * the volume's actual histogram. Returns the threshold as a [0,1] fraction (u8/255),
 * matching the air_thresh convention. */
float fy_auto_air_thresh(const fy_hist_state *s) {
    long total = s->total;
    if (total <= 0) return 0.15f;
    /* total intensity */
    double sumAll = 0;
    for (int i = 0; i < 256; i++) sumAll += (double)i * s->hist[i];
    double wB = 0, sumB = 0, maxVar = -1;
    int best = 38; /* ~0.15*255 fallback */
    for (int t = 0; t < 256; t++) {
        wB += s->hist[t];
        if (wB == 0) continue;
        double wF = total - wB;
        if (wF == 0) break;
        sumB += (double)t * s->hist[t];
        double mB = sumB / wB;
        double mF = (sumAll - sumB) / wF;
        double var = wB * wF * (mB - mF) * (mB - mF);
        if (var > maxVar) { maxVar = var; best = t; }
    }
    return (float)best / 255.0f;
}
