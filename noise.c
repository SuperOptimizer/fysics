/* noise.c -- per-volume noise-model estimation.
 *
 * Measured empirically (145 cubes, 18 scrolls): the noise in nabu-reconstructed
 * scroll volumes is SIGNAL-DEPENDENT (var ~= g*I + b) and its level varies 1.5-3.3x
 * scroll-to-scroll at the SAME resolution -- so denoise strength must be estimated
 * per-volume from the data, not hardcoded or predicted from voxel size.
 *
 * Method: local mean & variance in (win)^3 windows; bin voxels by local mean; in
 * each intensity bin the LOW percentile of local variance is the noise floor (high
 * local variance = edges/structure, rejected). Robust IRLS line fit var = g*I + b.
 * The slope can be ill-conditioned, so the PRIMARY output is the noise std at a
 * reference intensity (robust); g,b are secondary.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* separable 3D box SUM over a (2r+1)^3 window via 1D moving sums (O(N)). */
static void box_sum(const float *in, double *out, double *tmp,
                    int nz, int ny, int nx, int r) {
    size_t n = (size_t)nz * ny * nx;
    /* X */
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) {
        size_t row = ((size_t)z * ny + y) * nx;
        double s = 0;
        for (int x = 0; x <= r && x < nx; x++) s += in[row + x];
        for (int x = 0; x < nx; x++) {
            out[row + x] = s;
            int add = x + r + 1, sub = x - r;
            if (add < nx) s += in[row + add];
            if (sub >= 0) s -= in[row + sub];
        }
    }
    /* Y (in place via tmp) */
    memcpy(tmp, out, sizeof(double) * n);
    for (int z = 0; z < nz; z++) for (int x = 0; x < nx; x++) {
        double s = 0;
        for (int y = 0; y <= r && y < ny; y++) s += tmp[((size_t)z*ny+y)*nx+x];
        for (int y = 0; y < ny; y++) {
            out[((size_t)z*ny+y)*nx+x] = s;
            int add = y + r + 1, sub = y - r;
            if (add < ny) s += tmp[((size_t)z*ny+add)*nx+x];
            if (sub >= 0) s -= tmp[((size_t)z*ny+sub)*nx+x];
        }
    }
    /* Z */
    memcpy(tmp, out, sizeof(double) * n);
    for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        double s = 0;
        for (int z = 0; z <= r && z < nz; z++) s += tmp[((size_t)z*ny+y)*nx+x];
        for (int z = 0; z < nz; z++) {
            out[((size_t)z*ny+y)*nx+x] = s;
            int add = z + r + 1, sub = z - r;
            if (add < nz) s += tmp[((size_t)add*ny+y)*nx+x];
            if (sub >= 0) s -= tmp[((size_t)sub*ny+y)*nx+x];
        }
    }
}

/* in-place ascending sort (small arrays) for percentile */
static int cmp_d(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
static double percentile(double *vals, int n, double pct) {
    if (n <= 0) return 0.0;
    qsort(vals, n, sizeof(double), cmp_d);
    double idx = pct / 100.0 * (n - 1);
    int lo = (int)idx; double f = idx - lo;
    if (lo + 1 < n) return vals[lo] * (1 - f) + vals[lo + 1] * f;
    return vals[lo];
}

int fy_estimate_noise(const float *in, int nz, int ny, int nx,
                      int win, double flat_pct, double ref_intensity,
                      fy_noise_model *out) {
    if (!out) return 1;
    if (win < 3) win = 5;
    if (flat_pct <= 0) flat_pct = 10.0;
    int r = win / 2;
    size_t n = (size_t)nz * ny * nx;
    double *sum = malloc(sizeof(double) * n);
    double *sqsum = malloc(sizeof(double) * n);
    double *tmp = malloc(sizeof(double) * n);
    float *isq = malloc(sizeof(float) * n);
    if (!sum || !sqsum || !tmp || !isq) { free(sum); free(sqsum); free(tmp); free(isq); return 1; }
    for (size_t i = 0; i < n; i++) isq[i] = in[i] * in[i];

    box_sum(in, sum, tmp, nz, ny, nx, r);
    box_sum(isq, sqsum, tmp, nz, ny, nx, r);
    double wcount = (double)win * win * win;   /* approximate count (edges differ) */

    /* per-voxel local mean & variance (variance = E[x^2]-E[x]^2, clamped >=0) */
    /* reuse `tmp` for local mean, `sqsum` becomes local var */
    for (size_t i = 0; i < n; i++) {
        double m = sum[i] / wcount;
        double v = sqsum[i] / wcount - m * m;
        tmp[i] = m;                 /* local mean */
        sqsum[i] = v > 0 ? v : 0;   /* local variance */
    }

    /* intensity range from 1..99 percentile of local mean (avoid outliers) */
    double *means_copy = malloc(sizeof(double) * n);
    for (size_t i = 0; i < n; i++) means_copy[i] = tmp[i];
    double lo = percentile(means_copy, (int)n, 1.0);
    double hi = percentile(means_copy, (int)n, 99.0);
    free(means_copy);
    if (hi <= lo) { hi = lo + 1e-6; }

    const int NB = 24;
    /* collect per-bin local-variance floors */
    double bin_I[NB], bin_floor[NB]; long bin_cnt[NB];
    int used = 0;
    /* gather variances per bin into temp arrays (bounded memory: one pass count then fill) */
    /* simple approach: for each bin, scan and collect; cheap enough for a chunk */
    for (int bk = 0; bk < NB; bk++) {
        double b0 = lo + (hi - lo) * bk / NB;
        double b1 = lo + (hi - lo) * (bk + 1) / NB;
        /* count first */
        long c = 0;
        for (size_t i = 0; i < n; i++) if (tmp[i] >= b0 && tmp[i] < b1) c++;
        if (c < 200) continue;
        double *vbuf = malloc(sizeof(double) * c);
        if (!vbuf) continue;
        long k = 0;
        for (size_t i = 0; i < n; i++) if (tmp[i] >= b0 && tmp[i] < b1) vbuf[k++] = sqsum[i];
        double floor_v = percentile(vbuf, (int)c, flat_pct);
        free(vbuf);
        bin_I[used] = 0.5 * (b0 + b1);
        bin_floor[used] = floor_v;
        bin_cnt[used] = c;
        used++;
    }
    out->n_bins_used = used;

    /* robust IRLS line fit var = g*I + b over (bin_I, bin_floor), weights ~ sqrt(count) */
    double g = 0, b = 0;
    if (used >= 3) {
        double w[NB];
        for (int i = 0; i < used; i++) w[i] = sqrt((double)bin_cnt[i]);
        for (int it = 0; it < 4; it++) {
            double Sw=0, Swx=0, Swy=0, Swxx=0, Swxy=0;
            for (int i = 0; i < used; i++) {
                double wi = w[i], xi = bin_I[i], yi = bin_floor[i];
                Sw += wi; Swx += wi*xi; Swy += wi*yi; Swxx += wi*xi*xi; Swxy += wi*xi*yi;
            }
            double det = Sw*Swxx - Swx*Swx;
            if (fabs(det) < 1e-20) break;
            g = (Sw*Swxy - Swx*Swy) / det;
            b = (Swxx*Swy - Swx*Swxy) / det;
            /* reweight by residual (robust) */
            double resid[NB], med;
            for (int i = 0; i < used; i++) resid[i] = fabs(bin_floor[i] - (g*bin_I[i] + b));
            double rc[NB]; memcpy(rc, resid, sizeof(double)*used);
            med = percentile(rc, used, 50.0) + 1e-9;
            for (int i = 0; i < used; i++) {
                double rr = resid[i] / (2.0 * med);
                w[i] = sqrt((double)bin_cnt[i]) / (1.0 + rr*rr);
            }
        }
    } else if (used > 0) {
        /* not enough bins for a slope: flat noise model */
        double s = 0; for (int i = 0; i < used; i++) s += bin_floor[i];
        b = s / used; g = 0;
    } else {
        /* no flat bins at all: fall back to global low-percentile variance */
        double *vc = malloc(sizeof(double) * n);
        for (size_t i = 0; i < n; i++) vc[i] = sqsum[i];
        b = percentile(vc, (int)n, flat_pct);
        free(vc); g = 0;
    }
    if (!isfinite(g)) g = 0;
    if (!isfinite(b)) b = 0;
    out->g = g; out->b = b;

    /* PRIMARY robust output: noise STD at the reference intensity.
     * var_ref = g*ref + b, clamped >= 0; std = sqrt. Local-window variance slightly
     * UNDER-estimates point variance (the window pre-averages), so scale up by a
     * small correction factor for an unbiased-ish estimate. */
    double var_ref = g * ref_intensity + b;
    if (var_ref < 0) var_ref = b > 0 ? b : 0;
    /* window-averaging bias: local var of a (win^3) box mean underestimates by
     * roughly the box's own variance reduction; empirically ~1.0-1.3x. Use a mild
     * fixed correction (the denoisers are tuned to relative level, not absolute). */
    double corr = 1.15;
    out->noise_ref = sqrt(var_ref) * corr;
    out->ref_intensity = ref_intensity;

    free(sum); free(sqsum); free(tmp); free(isq);
    return 0;
}

/* Detail-safe guided-filter eps from an estimated noise level. The measured knee is
 * eps ~= 0.014*var(data); var ~ (k*noise_ref)^2 with k~3 reproduces it. Floored to a
 * small positive value so a near-zero noise estimate still yields a valid (gentle) eps. */
double fy_guided_eps_for_noise(double noise_ref) {
    double k = 3.0;
    double eps = (k * noise_ref) * (k * noise_ref);
    if (!(eps > 1e-8)) eps = 1e-8;
    return eps;
}

/* ---- noise-correlation anisotropy (PSF z/xy ratio) ----
 * Lag-1 autocorrelation of the residual (voxel - block mean) per axis, measured in
 * FLAT 8^3 blocks (block variance below the 20th percentile, so structure is mostly
 * excluded and the residual is dominated by PSF-correlated noise). For a Gaussian
 * PSF of width sigma, rho(1) = exp(-1/(4 sigma^2)) -- so the per-axis rho gives the
 * relative PSF widths without needing edges. */
int fy_noise_aniso(const float *v, int nz, int ny, int nx,
                   double *rho_z, double *rho_xy) {
    const int B = 8;
    int bz = nz / B, by = ny / B, bx = nx / B;
    long nb = (long)bz * by * bx;
    if (nb < 64) return 1;
    /* pass 1: block variances to find the flat-block threshold */
    float *bvar = (float *)malloc(sizeof(float) * nb);
    if (!bvar) return 1;
    long nv = 0;
    for (int z = 0; z < bz; z++) for (int y = 0; y < by; y++) for (int x = 0; x < bx; x++) {
        double s = 0, s2 = 0; int nonzero = 1;
        for (int dz = 0; dz < B; dz++) for (int dy = 0; dy < B; dy++) {
            const float *row = v + ((size_t)(z*B+dz) * ny + (y*B+dy)) * nx + (size_t)x*B;
            for (int dx = 0; dx < B; dx++) { float t = row[dx]; if (t <= 0) nonzero = 0; s += t; s2 += (double)t*t; }
        }
        if (!nonzero) continue;
        double m = s / (B*B*B);
        bvar[nv++] = (float)(s2 / (B*B*B) - m*m);
    }
    if (nv < 64) { free(bvar); return 1; }
    /* 20th percentile threshold (nth_element-lite: sort the small array) */
    for (long i = 1; i < nv; i++) { float k = bvar[i]; long j = i - 1;
        while (j >= 0 && bvar[j] > k) { bvar[j+1] = bvar[j]; j--; } bvar[j+1] = k; }
    float thresh = bvar[nv / 5];
    /* pass 2: per-axis lag-1 products over residuals in flat blocks */
    double num_z = 0, num_y = 0, num_x = 0, den = 0;
    long used = 0;
    float blk[8*8*8];
    for (int z = 0; z < bz; z++) for (int y = 0; y < by; y++) for (int x = 0; x < bx; x++) {
        double s = 0, s2 = 0; int nonzero = 1;
        for (int dz = 0; dz < B; dz++) for (int dy = 0; dy < B; dy++) {
            const float *row = v + ((size_t)(z*B+dz) * ny + (y*B+dy)) * nx + (size_t)x*B;
            for (int dx = 0; dx < B; dx++) {
                float t = row[dx]; if (t <= 0) nonzero = 0;
                blk[(dz*B+dy)*B+dx] = t; s += t; s2 += (double)t*t;
            }
        }
        if (!nonzero) continue;
        double m = s / (B*B*B);
        if (s2 / (B*B*B) - m*m > thresh) continue;
        for (int i = 0; i < B*B*B; i++) blk[i] -= (float)m;
        for (int dz = 0; dz < B; dz++) for (int dy = 0; dy < B; dy++) for (int dx = 0; dx < B; dx++) {
            float c = blk[(dz*B+dy)*B+dx];
            den += (double)c * c;
            if (dz+1 < B) num_z += (double)c * blk[((dz+1)*B+dy)*B+dx];
            if (dy+1 < B) num_y += (double)c * blk[(dz*B+dy+1)*B+dx];
            if (dx+1 < B) num_x += (double)c * blk[(dz*B+dy)*B+dx+1];
        }
        used++;
    }
    free(bvar);
    if (used < 32 || den <= 0) return 1;
    /* lag-1 sums cover (B-1)/B of the pairs the denominator covers; the bias is the
     * same on every axis so it cancels in the z/xy RATIO -- correct it anyway. */
    double corr = (double)B / (B - 1);
    *rho_z  = corr * num_z / den;
    *rho_xy = corr * 0.5 * (num_y + num_x) / den;
    return 0;
}
