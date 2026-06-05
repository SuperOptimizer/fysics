/* spectral.c -- multi-energy spectral decomposition (per-voxel).
 *
 * Given N CO-REGISTERED energy volumes of the SAME object, already converted to
 * physical attenuation (mu) on a COMMON scale (fy_u8_to_phys per energy), produce a
 * per-voxel "high-Z" / material-contrast channel that surfaces material a single
 * energy cannot. PURELY LOCAL (one voxel in, one voxel out -- no neighbourhood), so
 * it is the most vc3d/streaming-friendly kernel: apply per chunk, no halo.
 *
 * PHYSICS. Attenuation mu(E) = photoelectric + Compton. Photoelectric ~ Z^(3-4)/E^3
 * (steep falloff with energy, strong for high atomic number), Compton ~ density,
 * nearly flat with energy. So a LOW-Z material (papyrus, carbon Z~6) has a fairly
 * FLAT mu(E); a HIGHER-Z material has mu(E) that is much STEEPER at low energy. The
 * log-log slope d ln mu / d ln E (negative; more negative = more photoelectric = higher
 * Z) and the low/high energy ratio are Z-fingerprints that separate materials even
 * where any single energy cannot.
 *
 * Validated on PHerc0343P (43/62/77/89 keV): a coherent high-Z population, invisible
 * at single energy, with a real steep-photoelectric low-energy mu(E), reproducible
 * across ROIs and surviving tight registration (NOT a registration artifact). It is a
 * material-CONTRAST channel, NOT a calibrated material ID -- treat the output as an
 * exploratory contrast map, gated against the noise floor (low-mu voxels are noise-
 * dominated and must be suppressed, or the ratio explodes on near-zero denominators).
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Per-voxel spectral decomposition.
 *   mu      : N pointers, mu[e] is the physical-attenuation volume at energy e
 *             (all same nz,ny,nx, co-registered, dewindowed to a common scale).
 *   energies: the N energies in keV (must be increasing or at least distinct).
 *   n_energy: N (>=2; the photoelectric/Compton split needs >=2, robust with >=3).
 *   slope_out  : (optional, may be NULL) per-voxel log-log slope d ln mu/d ln E
 *                (Z proxy; more negative => higher Z). NaN-safe: set to 0 where gated.
 *   highz_out  : (optional, may be NULL) per-voxel high-Z CONTRAST score in [0,1]:
 *                combines a steep (very negative) slope with a real low-energy excess,
 *                gated so noise-dominated low-mu voxels score 0.
 *   mu_floor : voxels whose HIGHEST-energy mu is below this are noise-dominated and
 *              gated to 0 (validated gate: mu(high E) > ~0.02 on this data). Pass <=0
 *              to auto-set from the data (a low percentile of the high-energy volume).
 * Returns 0 on success. */
int fy_spectral_decompose(const float *const *mu, const double *energies, int n_energy,
                          int nz, int ny, int nx,
                          float *slope_out, float *highz_out, double mu_floor) {
    if (!mu || !energies || n_energy < 2) return 1;
    size_t n = (size_t)nz * ny * nx;

    /* precompute lnE and its centering for the least-squares slope */
    double *lnE = malloc(sizeof(double) * n_energy);
    if (!lnE) return 1;
    double lnE_mean = 0;
    for (int e = 0; e < n_energy; e++) {
        if (energies[e] <= 0) { free(lnE); return 1; }
        lnE[e] = log(energies[e]); lnE_mean += lnE[e];
    }
    lnE_mean /= n_energy;
    double Sxx = 0;                       /* sum (lnE-mean)^2, constant over voxels */
    for (int e = 0; e < n_energy; e++) { double d = lnE[e] - lnE_mean; Sxx += d * d; }
    if (Sxx < 1e-12) { free(lnE); return 1; }

    /* identify the highest-energy index (for the noise gate) */
    int hi = 0; for (int e = 1; e < n_energy; e++) if (energies[e] > energies[hi]) hi = e;
    int lo = 0; for (int e = 1; e < n_energy; e++) if (energies[e] < energies[lo]) lo = e;

    /* auto noise floor: a low percentile of the high-energy attenuation if not given.
     * Sample (stride) to keep it cheap, then use ~the 30th percentile of positive mu
     * as "papyrus-or-air level"; gate a bit above it. */
    if (mu_floor <= 0) {
        size_t cap = n < 200000 ? n : 200000;
        size_t stride = n / cap; if (stride < 1) stride = 1;
        float *samp = malloc(sizeof(float) * (n / stride + 1));
        size_t m = 0;
        for (size_t i = 0; i < n; i += stride) { float v = mu[hi][i]; if (v > 0) samp[m++] = v; }
        if (m > 8) {
            /* simple selection of the 30th percentile via partial sort (qsort ok here) */
            for (size_t a = 0; a < m; a++) for (size_t b = a + 1; b < m; b++)
                if (samp[b] < samp[a]) { float t = samp[a]; samp[a] = samp[b]; samp[b] = t; }
            mu_floor = samp[(size_t)(0.30 * m)];
        } else mu_floor = 0.02;
        free(samp);
    }

    for (size_t i = 0; i < n; i++) {
        /* gate: need real signal at the HIGH energy (noise-dominated low-mu -> skip) */
        double mu_hi = mu[hi][i];
        if (mu_hi <= mu_floor) {
            if (slope_out) slope_out[i] = 0.0f;
            if (highz_out) highz_out[i] = 0.0f;
            continue;
        }
        /* least-squares slope of ln mu vs ln E (only over positive-mu energies) */
        double Sxy = 0, lnmu_sum = 0; int cnt = 0;
        double lnmu[16]; int idx[16];
        for (int e = 0; e < n_energy && e < 16; e++) {
            double v = mu[e][i];
            if (v > 1e-6) { lnmu[cnt] = log(v); idx[cnt] = e; lnmu_sum += lnmu[cnt]; cnt++; }
        }
        if (cnt < 2) { if (slope_out) slope_out[i] = 0.0f; if (highz_out) highz_out[i] = 0.0f; continue; }
        double lnmu_mean = lnmu_sum / cnt;
        /* recompute centered Sxx over the available energies (cnt may be < n_energy) */
        double sxx = 0, sxy = 0, lx_mean = 0;
        for (int k = 0; k < cnt; k++) lx_mean += lnE[idx[k]];
        lx_mean /= cnt;
        for (int k = 0; k < cnt; k++) {
            double dx = lnE[idx[k]] - lx_mean;
            sxx += dx * dx; sxy += dx * (lnmu[k] - lnmu_mean);
        }
        double slope = (sxx > 1e-12) ? sxy / sxx : 0.0;   /* d ln mu / d ln E */
        (void)Sxy;
        if (slope_out) slope_out[i] = (float)slope;

        if (highz_out) {
            /* high-Z contrast: photoelectric makes slope strongly NEGATIVE. Papyrus
             * (low-Z) slope ~ -0.5..-1 (mild); high-Z material much steeper (< -1.5).
             * Map a "steepness beyond papyrus" to [0,1], multiplied by a low-energy
             * EXCESS confirmation (mu_lo notably above mu_hi). Both must hold. */
            double mu_lo = mu[lo][i];
            double steep = -slope;                       /* positive for falling mu(E) */
            /* soft ramp: 0 at slope=-1 (papyrus-ish), ->1 by slope=-3 */
            double s = (steep - 1.0) / 2.0;
            if (s < 0) s = 0; if (s > 1) s = 1;
            /* low-energy excess ratio, normalized; >1 means low energy attenuates more */
            double ratio = (mu_hi > 1e-6) ? mu_lo / mu_hi : 1.0;
            double r = (ratio - 1.0) / 2.0;              /* 0 at ratio=1, ->1 by ratio=3 */
            if (r < 0) r = 0; if (r > 1) r = 1;
            highz_out[i] = (float)(s * r);               /* both conditions -> high score */
        }
    }
    free(lnE);
    return 0;
}
