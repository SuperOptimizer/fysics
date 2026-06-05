/* fuse.c -- LANDMARK affine fit + MULTI-RESOLUTION FUSION for scroll CT.
 *
 * Two scans of the SAME scroll at different voxel sizes (e.g. PHerc0139 1.129um
 * and 2.399um) carry INDEPENDENT measurements of the same object. Fusing them can
 * beat the single-scan "no new resolution" wall: where two independent noise
 * realizations AGREE, that is real signal -- so averaging the shared band denoises
 * it, while the fine scan still supplies the high-frequency detail.
 *
 * This file provides the two reusable kernels the fusion pipeline needs that are
 * not already in register.c:
 *
 *   1. fy_affine_from_points -- fit a 3x4 affine (or rigid+isotropic-scale
 *      similarity) from N matched 3D point pairs, by least squares, with optional
 *      RANSAC for robustness to mismatched landmarks. This is the TRUSTWORTHY way
 *      to register these scans: intensity NCC/MI is non-discriminative on the
 *      self-similar laminar medium at a common coarse scale, so registration must
 *      be driven by matched landmarks (hand-placed, or from a feature detector).
 *
 *   2. fy_fuse_multiscale -- given the fine and coarse scans ALREADY resampled
 *      onto a common grid, frequency-split and fuse: high band from the fine scan,
 *      low/mid band a per-band SNR-weighted combination of both. The coarse scan's
 *      low band is an independent confirmation of the fine scan's low band, so the
 *      fused low band has lower noise than either input.
 *
 * Geometry / conventions match register.c:
 *   row-major, x fastest: idx = (z*ny+y)*nx + x.
 *   A 3x4 affine M (12 doubles, row-major) is a PULL map output->input:
 *       in_coord = M[0..2] . out_coord + M[3]   (z row), etc.  (see register.c).
 *   Point arrays are N*3 doubles in (z,y,x) order to match the index order.
 */
#include "fysics.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Small dense linear algebra (no BLAS dependency). All matrices row-major.
 * ========================================================================== */

/* Solve A x = b for an n-by-n system by Gaussian elimination w/ partial pivot.
 * A (n*n) and b (n) are overwritten; solution returned in x. Returns 0 ok, 1
 * singular. n is small (<=4 here). */
static int solve_dense(double *A, double *b, double *x, int n) {
    for (int c = 0; c < n; c++) {
        /* pivot */
        int piv = c; double best = fabs(A[c * n + c]);
        for (int r = c + 1; r < n; r++) {
            double v = fabs(A[r * n + c]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-12) return 1;
        if (piv != c) {
            for (int k = 0; k < n; k++) {
                double t = A[c * n + k]; A[c * n + k] = A[piv * n + k]; A[piv * n + k] = t;
            }
            double t = b[c]; b[c] = b[piv]; b[piv] = t;
        }
        double d = A[c * n + c];
        for (int r = 0; r < n; r++) {
            if (r == c) continue;
            double f = A[r * n + c] / d;
            if (f == 0.0) continue;
            for (int k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
            b[r] -= f * b[c];
        }
    }
    for (int i = 0; i < n; i++) x[i] = b[i] / A[i * n + i];
    return 0;
}

/* 3x3 symmetric eigen-decomposition via cyclic Jacobi. A (9, row-major, sym) is
 * destroyed; eigenvectors returned in V (9, columns are eigenvectors), eigenvalues
 * on the diagonal of A. Used for the Kabsch/Umeyama rotation. */
static void jacobi3(double *A, double *V) {
    for (int i = 0; i < 9; i++) V[i] = (i % 4 == 0) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 50; sweep++) {
        double off = fabs(A[1]) + fabs(A[2]) + fabs(A[5]);
        if (off < 1e-18) break;
        for (int p = 0; p < 3; p++) {
            for (int q = p + 1; q < 3; q++) {
                double apq = A[p * 3 + q];
                if (fabs(apq) < 1e-300) continue;
                double app = A[p * 3 + p], aqq = A[q * 3 + q];
                double phi = 0.5 * atan2(2.0 * apq, aqq - app);
                double c = cos(phi), s = sin(phi);
                for (int k = 0; k < 3; k++) {
                    double akp = A[k * 3 + p], akq = A[k * 3 + q];
                    A[k * 3 + p] = c * akp - s * akq;
                    A[k * 3 + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; k++) {
                    double apk = A[p * 3 + k], aqk = A[q * 3 + k];
                    A[p * 3 + k] = c * apk - s * aqk;
                    A[q * 3 + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; k++) {
                    double vkp = V[k * 3 + p], vkq = V[k * 3 + q];
                    V[k * 3 + p] = c * vkp - s * vkq;
                    V[k * 3 + q] = s * vkp + c * vkq;
                }
            }
        }
    }
}

/* ----------------------------------------------------------------------------
 * Fit FULL affine src->dst by least squares: dst ~ A_lin * src + t.
 * src,dst are n*3 (z,y,x). Result written to a 3x4 row-major map dst<-src:
 *   out[r*4 + 0..2] = A_lin row r,  out[r*4 + 3] = t[r].
 * Solves the 12 unknowns as 3 independent 4-param least-squares (one per out axis)
 * sharing the same 4x4 normal matrix [src 1]^T [src 1].
 * -------------------------------------------------------------------------- */
static int fit_affine_ls(const double *src, const double *dst, int n, double *out) {
    if (n < 4) return 1;
    double N[16]; for (int i = 0; i < 16; i++) N[i] = 0.0;
    double rhs[3][4]; for (int a = 0; a < 3; a++) for (int j = 0; j < 4; j++) rhs[a][j] = 0.0;
    for (int i = 0; i < n; i++) {
        double s[4] = { src[i * 3 + 0], src[i * 3 + 1], src[i * 3 + 2], 1.0 };
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) N[r * 4 + c] += s[r] * s[c];
        for (int a = 0; a < 3; a++)
            for (int j = 0; j < 4; j++) rhs[a][j] += dst[i * 3 + a] * s[j];
    }
    for (int a = 0; a < 3; a++) {
        double Ncopy[16], b[4], x[4];
        memcpy(Ncopy, N, sizeof(Ncopy));
        for (int j = 0; j < 4; j++) b[j] = rhs[a][j];
        if (solve_dense(Ncopy, b, x, 4)) return 1;
        out[a * 4 + 0] = x[0]; out[a * 4 + 1] = x[1];
        out[a * 4 + 2] = x[2]; out[a * 4 + 3] = x[3];
    }
    return 0;
}

/* ----------------------------------------------------------------------------
 * Fit a SIMILARITY src->dst (rotation R, single isotropic scale s, translation t):
 *   dst ~ s*R*src + t.   Closed-form (Umeyama 1991) -- the right model for two
 * scans of the same rigid object differing only by voxel-size ratio + remount pose.
 * out is the 3x4 row-major map dst<-src as above.
 * -------------------------------------------------------------------------- */
static int fit_similarity(const double *src, const double *dst, int n, double *out) {
    if (n < 3) return 1;
    double ms[3] = {0,0,0}, md[3] = {0,0,0};
    for (int i = 0; i < n; i++) for (int a = 0; a < 3; a++) {
        ms[a] += src[i * 3 + a]; md[a] += dst[i * 3 + a];
    }
    for (int a = 0; a < 3; a++) { ms[a] /= n; md[a] /= n; }
    /* covariance H = sum (dst-md)(src-ms)^T  (3x3), and src variance */
    double H[9]; for (int i = 0; i < 9; i++) H[i] = 0.0;
    double var_s = 0.0;
    for (int i = 0; i < n; i++) {
        double ds[3], dd[3];
        for (int a = 0; a < 3; a++) { ds[a] = src[i*3+a]-ms[a]; dd[a] = dst[i*3+a]-md[a]; }
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) H[r*3+c] += dd[r]*ds[c];
        var_s += ds[0]*ds[0]+ds[1]*ds[1]+ds[2]*ds[2];
    }
    /* SVD of H via eigendecomposition of H^T H and H H^T.
     * R = U V^T with a sign fix; s = trace(D R) / var_s. We get U,V from
     * symmetric eigensolves: H H^T = U S^2 U^T,  H^T H = V S^2 V^T. */
    double HHt[9], HtH[9];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
        double a = 0, b = 0;
        for (int k = 0; k < 3; k++) { a += H[r*3+k]*H[c*3+k]; b += H[k*3+r]*H[k*3+c]; }
        HHt[r*3+c] = a; HtH[r*3+c] = b;
    }
    double U[9], V[9], ev1[9], ev2[9];
    memcpy(ev1, HHt, sizeof ev1); jacobi3(ev1, U);
    memcpy(ev2, HtH, sizeof ev2); jacobi3(ev2, V);
    /* sort eigvecs by eigenvalue desc so U and V correspond */
    int oU[3] = {0,1,2}, oV[3] = {0,1,2};
    double dU[3] = {ev1[0], ev1[4], ev1[8]}, dV[3] = {ev2[0], ev2[4], ev2[8]};
    for (int i = 0; i < 3; i++) for (int j = i+1; j < 3; j++) {
        if (dU[oU[j]] > dU[oU[i]]) { int t=oU[i]; oU[i]=oU[j]; oU[j]=t; }
        if (dV[oV[j]] > dV[oV[i]]) { int t=oV[i]; oV[i]=oV[j]; oV[j]=t; }
    }
    /* R = U_sorted * V_sorted^T, columns reordered */
    double Us[9], Vs[9];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
        Us[r*3+c] = U[r*3 + oU[c]];
        Vs[r*3+c] = V[r*3 + oV[c]];
    }
    double R[9];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
        double a = 0; for (int k = 0; k < 3; k++) a += Us[r*3+k]*Vs[c*3+k];
        R[r*3+c] = a;
    }
    /* fix reflection: if det(R) < 0, flip the last column of Us */
    double det = R[0]*(R[4]*R[8]-R[5]*R[7]) - R[1]*(R[3]*R[8]-R[5]*R[6]) + R[2]*(R[3]*R[7]-R[4]*R[6]);
    if (det < 0) {
        for (int r = 0; r < 3; r++) Us[r*3+2] = -Us[r*3+2];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
            double a = 0; for (int k = 0; k < 3; k++) a += Us[r*3+k]*Vs[c*3+k];
            R[r*3+c] = a;
        }
    }
    /* scale s = trace(D * R_alignment)/var_s ; trace(D R) = sum H .* R */
    double tr = 0.0; for (int i = 0; i < 9; i++) tr += H[i] * R[i];
    double s = (var_s > 1e-30) ? tr / var_s : 1.0;
    /* t = md - s R ms */
    for (int r = 0; r < 3; r++) {
        double Rs = 0; for (int c = 0; c < 3; c++) Rs += R[r*3+c]*ms[c];
        out[r*4+0] = s*R[r*3+0]; out[r*4+1] = s*R[r*3+1]; out[r*4+2] = s*R[r*3+2];
        out[r*4+3] = md[r] - s*Rs;
    }
    return 0;
}

static void apply34(const double *M, const double *p, double *q) {
    for (int r = 0; r < 3; r++)
        q[r] = M[r*4+0]*p[0] + M[r*4+1]*p[1] + M[r*4+2]*p[2] + M[r*4+3];
}

/* ----------------------------------------------------------------------------
 * fy_affine_from_points: fit dst<-src from N matched 3D point pairs.
 *   src, dst : n*3 doubles (z,y,x).
 *   model    : 0 = full affine (12 dof), 1 = similarity (rigid + iso scale, 7 dof).
 *   ransac_iters : if >0, run RANSAC (random minimal subsets, keep the model with
 *                  most inliers within `inlier_thresh` of the dst point), then
 *                  refit on all inliers. If 0, plain least-squares on all points.
 *   inlier_thresh : RANSAC inlier distance (in dst voxel units). Ignored if no RANSAC.
 *   M_out    : 3x4 row-major result, the dst<-src map (so to pull the moving=src
 *              volume onto the fixed=dst grid you INVERT it; or fit with src=fixed,
 *              dst=moving to get the pull map directly -- see FUSION.md).
 *   inlier_mask : optional (n ints, may be NULL), 1 if point i is an inlier.
 *   resid_rms_out : optional, RMS residual (dst units) over inliers (or all).
 * Returns 0 on success, 1 on failure (too few points / degenerate).
 * -------------------------------------------------------------------------- */
int fy_affine_from_points(const double *src, const double *dst, int n,
                          int model, int ransac_iters, double inlier_thresh,
                          double *M_out, int *inlier_mask, double *resid_rms_out) {
    if (!src || !dst || !M_out || n < 3) return 1;
    int minpts = (model == 1) ? 3 : 4;
    if (n < minpts) return 1;

    int (*fit)(const double*, const double*, int, double*) =
        (model == 1) ? fit_similarity : fit_affine_ls;

    int *best_in = (int*)malloc((size_t)n * sizeof(int));
    int *cur_in  = (int*)malloc((size_t)n * sizeof(int));
    if (!best_in || !cur_in) { free(best_in); free(cur_in); return 1; }

    double bestM[12];
    int best_count = -1;

    if (ransac_iters <= 0) {
        if (fit(src, dst, n, bestM)) { free(best_in); free(cur_in); return 1; }
        for (int i = 0; i < n; i++) best_in[i] = 1;
        best_count = n;
    } else {
        unsigned int seed = 12345u;
        double *ss = (double*)malloc((size_t)minpts * 3 * sizeof(double));
        double *dd = (double*)malloc((size_t)minpts * 3 * sizeof(double));
        if (!ss || !dd) { free(ss); free(dd); free(best_in); free(cur_in); return 1; }
        for (int it = 0; it < ransac_iters; it++) {
            /* sample `minpts` distinct indices (rejection) */
            int idx[4];
            for (int k = 0; k < minpts; k++) {
                int r, dup;
                do {
                    seed = seed * 1103515245u + 12345u;
                    r = (int)((seed >> 8) % (unsigned)n);
                    dup = 0; for (int j = 0; j < k; j++) if (idx[j] == r) dup = 1;
                } while (dup);
                idx[k] = r;
            }
            for (int k = 0; k < minpts; k++)
                for (int a = 0; a < 3; a++) {
                    ss[k*3+a] = src[idx[k]*3+a];
                    dd[k*3+a] = dst[idx[k]*3+a];
                }
            double M[12];
            if (fit(ss, dd, minpts, M)) continue;
            int count = 0;
            for (int i = 0; i < n; i++) {
                double q[3]; apply34(M, &src[i*3], q);
                double dz = q[0]-dst[i*3+0], dy = q[1]-dst[i*3+1], dx = q[2]-dst[i*3+2];
                double e = sqrt(dz*dz + dy*dy + dx*dx);
                cur_in[i] = (e <= inlier_thresh) ? 1 : 0;
                count += cur_in[i];
            }
            if (count > best_count) {
                best_count = count;
                memcpy(bestM, M, sizeof bestM);
                memcpy(best_in, cur_in, (size_t)n * sizeof(int));
            }
        }
        free(ss); free(dd);
        if (best_count < minpts) { free(best_in); free(cur_in); return 1; }
        /* refit on all inliers */
        double *si = (double*)malloc((size_t)best_count*3*sizeof(double));
        double *di = (double*)malloc((size_t)best_count*3*sizeof(double));
        if (!si || !di) { free(si); free(di); free(best_in); free(cur_in); return 1; }
        int m = 0;
        for (int i = 0; i < n; i++) if (best_in[i]) {
            for (int a = 0; a < 3; a++) { si[m*3+a]=src[i*3+a]; di[m*3+a]=dst[i*3+a]; }
            m++;
        }
        double Mref[12];
        if (!fit(si, di, m, Mref)) memcpy(bestM, Mref, sizeof bestM);
        free(si); free(di);
    }

    memcpy(M_out, bestM, sizeof bestM);
    if (inlier_mask) memcpy(inlier_mask, best_in, (size_t)n * sizeof(int));
    if (resid_rms_out) {
        double se = 0.0; int m = 0;
        for (int i = 0; i < n; i++) if (best_in[i]) {
            double q[3]; apply34(bestM, &src[i*3], q);
            double dz = q[0]-dst[i*3+0], dy = q[1]-dst[i*3+1], dx = q[2]-dst[i*3+2];
            se += dz*dz+dy*dy+dx*dx; m++;
        }
        *resid_rms_out = (m > 0) ? sqrt(se / m) : 0.0;
    }
    free(best_in); free(cur_in);
    return 0;
}

/* ============================================================================
 * MULTI-RESOLUTION FUSION
 * ========================================================================== */

/* separable 3D gaussian blur, arbitrary sigma, reflect-border. in!=out. */
static void gauss3d(const float *in, float *out, int nz, int ny, int nx, double sigma) {
    if (sigma <= 0.0) { memcpy(out, in, (size_t)nz*ny*nx*sizeof(float)); return; }
    int rad = (int)(3.0 * sigma + 0.5); if (rad < 1) rad = 1;
    int klen = 2*rad + 1;
    double *k = (double*)malloc((size_t)klen*sizeof(double));
    double s = 0.0;
    for (int i = -rad; i <= rad; i++) { double w = exp(-0.5*i*i/(sigma*sigma)); k[i+rad]=w; s+=w; }
    for (int i = 0; i < klen; i++) k[i] /= s;
    size_t N = (size_t)nz*ny*nx;
    float *tmp = (float*)malloc(N*sizeof(float));
    float *a = (float*)malloc(N*sizeof(float));
    memcpy(a, in, N*sizeof(float));
    /* x */
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) {
        const float *row = a + ((size_t)z*ny+y)*nx;
        float *orow = tmp + ((size_t)z*ny+y)*nx;
        for (int x = 0; x < nx; x++) {
            double acc = 0.0;
            for (int t = -rad; t <= rad; t++) {
                int xi = x+t; if (xi<0) xi=-xi; if (xi>=nx) xi=2*nx-2-xi;
                acc += k[t+rad]*row[xi];
            }
            orow[x] = (float)acc;
        }
    }
    /* y */
    for (int z = 0; z < nz; z++) for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            double acc = 0.0;
            for (int t = -rad; t <= rad; t++) {
                int yi = y+t; if (yi<0) yi=-yi; if (yi>=ny) yi=2*ny-2-yi;
                acc += k[t+rad]*tmp[((size_t)z*ny+yi)*nx+x];
            }
            a[((size_t)z*ny+y)*nx+x] = (float)acc;
        }
    }
    /* z */
    for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        for (int z = 0; z < nz; z++) {
            double acc = 0.0;
            for (int t = -rad; t <= rad; t++) {
                int zi = z+t; if (zi<0) zi=-zi; if (zi>=nz) zi=2*nz-2-zi;
                acc += k[t+rad]*a[((size_t)zi*ny+y)*nx+x];
            }
            out[((size_t)z*ny+y)*nx+x] = (float)acc;
        }
    }
    free(k); free(tmp); free(a);
}

/* ----------------------------------------------------------------------------
 * fy_fuse_multiscale: fuse a FINE and a COARSE scan already on the SAME grid.
 *
 * Model: split each volume into LOW (gaussian, sigma=split_sigma) and HIGH
 * (residual) bands.
 *   - HIGH band: taken from the FINE scan only (the coarse scan has no real detail
 *     there; its "high band" is pure noise). Optionally scaled by `high_gain`.
 *   - LOW band: the two scans are INDEPENDENT measurements of the same low-freq
 *     structure, so we average them with weights ~ inverse noise variance:
 *         low_fused = w_fine*low_fine + w_coarse*low_coarse,  w_fine+w_coarse=1.
 *     With two independent estimates of equal mean, the averaged variance is
 *         var = w_f^2 var_f + w_c^2 var_c, minimized at w_f = var_c/(var_f+var_c).
 *     Pass the measured per-scan low-band noise variances (var_fine, var_coarse);
 *     if both <=0, defaults to a plain 0.5/0.5 average.
 *   out = low_fused + high_gain*high_fine.
 *
 * The coarse low band must be intensity-matched to the fine scan first (different
 * energies => different contrast); we do a per-call affine intensity match
 * (least-squares a*coarse_low+b ~ fine_low over the overlap) so NCC-style contrast
 * differences don't bias the average. mask (u8, !=0 = valid overlap) may be NULL.
 *
 * Returns 0 on success.
 * -------------------------------------------------------------------------- */
int fy_fuse_multiscale(const float *fine, const float *coarse,
                       const unsigned char *mask,
                       int nz, int ny, int nx,
                       double split_sigma, double var_fine, double var_coarse,
                       double high_gain, float *out) {
    if (!fine || !coarse || !out || nz<=0 || ny<=0 || nx<=0) return 1;
    size_t N = (size_t)nz*ny*nx;
    float *lf = (float*)malloc(N*sizeof(float));
    float *lc = (float*)malloc(N*sizeof(float));
    if (!lf || !lc) { free(lf); free(lc); return 1; }
    gauss3d(fine, lf, nz, ny, nx, split_sigma);
    gauss3d(coarse, lc, nz, ny, nx, split_sigma);

    /* intensity-match coarse_low to fine_low: a*lc+b ~ lf, least squares on overlap */
    double sx=0, sy=0, sxx=0, sxy=0; long cnt=0;
    for (size_t i = 0; i < N; i++) {
        if (mask && !mask[i]) continue;
        double X = lc[i], Y = lf[i];
        sx += X; sy += Y; sxx += X*X; sxy += X*Y; cnt++;
    }
    double a = 1.0, b = 0.0;
    if (cnt > 1) {
        double denom = cnt*sxx - sx*sx;
        if (fabs(denom) > 1e-12) { a = (cnt*sxy - sx*sy)/denom; b = (sy - a*sx)/cnt; }
    }

    double wf = 0.5, wc = 0.5;
    if (var_fine > 0.0 && var_coarse > 0.0) {
        wf = var_coarse / (var_fine + var_coarse);
        wc = 1.0 - wf;
    }

    for (size_t i = 0; i < N; i++) {
        double lcm = a*lc[i] + b;              /* contrast-matched coarse low */
        double low = wf*lf[i] + wc*lcm;
        double high = (double)fine[i] - (double)lf[i];
        if (mask && !mask[i]) { out[i] = fine[i]; continue; }  /* no fusion outside overlap */
        out[i] = (float)(low + high_gain*high);
    }
    free(lf); free(lc);
    return 0;
}
