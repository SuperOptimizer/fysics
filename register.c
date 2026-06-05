/* register.c -- LAYER 1 of 3D volume registration: affine resampler +
 * intensity-based rigid/affine registration.
 *
 * Goal: register multiple CT scans of the SAME scroll taken at different
 * resolution/energy/time (e.g. PHerc0139 at 1.129um vs 2.399um) so they can be
 * fused. This layer handles the GLOBAL affine part (scale from differing voxel
 * size, rotation, translation, shear from optics/mounting). Physical warping
 * (temperature/movement) is LAYER 2 (deformable/Demons) -- not here, but the
 * trilinear sampler is factored out so a displacement-field warp can reuse it.
 *
 * Geometry convention (matches the rest of fysics):
 *   row-major, x fastest: idx = (z*ny+y)*nx + x.
 *   The 3x4 affine M (12 doubles, row-major) maps OUTPUT voxel coords -> INPUT
 *   voxel coords (the standard "pull"/backward-warp used by all resamplers):
 *       [zi]   [ M0 M1 M2 ] [zo]   [ M3 ]
 *       [yi] = [ M4 M5 M6 ] [yo] + [ M7 ]
 *       [xi]   [ M8 M9 M10] [xo]   [ M11]
 *   so M's translation column is element [3],[7],[11]. Order is (z,y,x) to match
 *   the loop/index order; callers building M must use the same ordering.
 *
 * Metric: Normalized Cross-Correlation (NCC) over the overlap region. NCC is
 * invariant to affine intensity change (a*I+b), which makes it robust to the
 * contrast/brightness difference between two energies -- plain SSD is WRONG for
 * multimodal data. NCC is NOT invariant to arbitrary nonlinear contrast; for
 * truly different modalities Mutual Information (MI) is more robust. MI is left
 * as a documented hook (see fy_register_metric / TODO below).
 */
#include "fysics.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- trilinear sampler (the reusable workhorse) ----------------------------
 * Sample volume `in` at continuous coords (z,y,x). Out-of-bounds -> 0.
 * Factored out so the deformable (Layer 2) warp can reuse the exact same
 * interpolation. Coords are in INPUT voxel units. */
float fy_sample_trilinear(const float *in, int nz, int ny, int nx,
                          float z, float y, float x) {
    /* standard convention: anything outside [0, n-1] is out-of-bounds -> 0 */
    if (z < 0.0f || y < 0.0f || x < 0.0f ||
        z > (float)(nz - 1) || y > (float)(ny - 1) || x > (float)(nx - 1))
        return 0.0f;

    int z0 = (int)z, y0 = (int)y, x0 = (int)x;
    int z1 = z0 + 1, y1 = y0 + 1, x1 = x0 + 1;
    if (z1 > nz - 1) z1 = nz - 1;
    if (y1 > ny - 1) y1 = ny - 1;
    if (x1 > nx - 1) x1 = nx - 1;
    float fz = z - z0, fy = y - y0, fx = x - x0;

    size_t s_y = (size_t)nx, s_z = (size_t)ny * nx;
    size_t b = (size_t)z0 * s_z + (size_t)y0 * s_y + x0;
    float c000 = in[b];
    float c001 = in[b + (x1 - x0)];
    float c010 = in[b + (y1 - y0) * s_y];
    float c011 = in[b + (y1 - y0) * s_y + (x1 - x0)];
    float c100 = in[b + (z1 - z0) * s_z];
    float c101 = in[b + (z1 - z0) * s_z + (x1 - x0)];
    float c110 = in[b + (z1 - z0) * s_z + (y1 - y0) * s_y];
    float c111 = in[b + (z1 - z0) * s_z + (y1 - y0) * s_y + (x1 - x0)];

    float c00 = c000 * (1 - fx) + c001 * fx;
    float c01 = c010 * (1 - fx) + c011 * fx;
    float c10 = c100 * (1 - fx) + c101 * fx;
    float c11 = c110 * (1 - fx) + c111 * fx;
    float c0 = c00 * (1 - fy) + c01 * fy;
    float c1 = c10 * (1 - fy) + c11 * fy;
    return c0 * (1 - fz) + c1 * fz;
}

/* ---- affine resampler -------------------------------------------------------
 * Resample `in` (nz,ny,nx) into `out` (onz,ony,onx) under 3x4 affine M mapping
 * output voxel coords -> input voxel coords. Trilinear; out-of-bounds -> 0. */
int fy_warp_affine(const float *in, float *out, int nz, int ny, int nx,
                   const double *M, int onz, int ony, int onx) {
    if (!in || !out || !M) return 1;
    for (int zo = 0; zo < onz; zo++) {
        for (int yo = 0; yo < ony; yo++) {
            /* precompute the part of the map independent of xo */
            double bz = M[0] * zo + M[1] * yo + M[3];
            double by = M[4] * zo + M[5] * yo + M[7];
            double bx = M[8] * zo + M[9] * yo + M[11];
            float *orow = out + ((size_t)zo * ony + yo) * onx;
            for (int xo = 0; xo < onx; xo++) {
                float zi = (float)(bz + M[2] * xo);
                float yi = (float)(by + M[6] * xo);
                float xi = (float)(bx + M[10] * xo);
                orow[xo] = fy_sample_trilinear(in, nz, ny, nx, zi, yi, xi);
            }
        }
    }
    return 0;
}

/* ---- gaussian blur (separable, sigma=0.8) + 2x decimation -------------------
 * Build a pyramid level: anti-alias (gaussian) then drop every other voxel.
 * sigma ~0.8 with a 2x decimation is the standard cheap anti-alias kernel. */
static void gaussian_blur_sep(const float *in, float *out, float *tmp,
                              int nz, int ny, int nx) {
    /* 5-tap gaussian, sigma~0.8: weights normalized */
    const float w[5] = {0.0276f, 0.2452f, 0.4544f, 0.2452f, 0.0276f};
    size_t s_y = (size_t)nx, s_z = (size_t)ny * nx;
    /* X */
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) {
        const float *r = in + (size_t)z * s_z + (size_t)y * s_y;
        float *o = out + (size_t)z * s_z + (size_t)y * s_y;
        for (int x = 0; x < nx; x++) {
            float acc = 0;
            for (int k = -2; k <= 2; k++) {
                int xx = x + k; if (xx < 0) xx = 0; if (xx > nx - 1) xx = nx - 1;
                acc += w[k + 2] * r[xx];
            }
            o[x] = acc;
        }
    }
    /* Y (out->tmp) */
    for (int z = 0; z < nz; z++) for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            float acc = 0;
            for (int k = -2; k <= 2; k++) {
                int yy = y + k; if (yy < 0) yy = 0; if (yy > ny - 1) yy = ny - 1;
                acc += w[k + 2] * out[(size_t)z * s_z + (size_t)yy * s_y + x];
            }
            tmp[(size_t)z * s_z + (size_t)y * s_y + x] = acc;
        }
    }
    /* Z (tmp->out) */
    for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        for (int z = 0; z < nz; z++) {
            float acc = 0;
            for (int k = -2; k <= 2; k++) {
                int zz = z + k; if (zz < 0) zz = 0; if (zz > nz - 1) zz = nz - 1;
                acc += w[k + 2] * tmp[(size_t)zz * s_z + (size_t)y * s_y + x];
            }
            out[(size_t)z * s_z + (size_t)y * s_y + x] = acc;
        }
    }
}

/* 2x anti-aliased downsample. Output dims are nz/2 etc (floor). out must be
 * pre-allocated by caller of size (nz/2)*(ny/2)*(nx/2). Returns 0 on success;
 * fills *onz,*ony,*onx with the output dims. */
int fy_downsample2x(const float *in, float *out, int nz, int ny, int nx,
                    int *onz, int *ony, int *onx) {
    if (!in || !out) return 1;
    int dz = nz / 2, dy = ny / 2, dx = nx / 2;
    if (dz < 1) dz = 1;
    if (dy < 1) dy = 1;
    if (dx < 1) dx = 1;
    if (onz) *onz = dz;
    if (ony) *ony = dy;
    if (onx) *onx = dx;

    size_t n = (size_t)nz * ny * nx;
    float *blur = malloc(sizeof(float) * n);
    float *tmp  = malloc(sizeof(float) * n);
    if (!blur || !tmp) { free(blur); free(tmp); return 1; }
    gaussian_blur_sep(in, blur, tmp, nz, ny, nx);

    size_t s_y = (size_t)nx, s_z = (size_t)ny * nx;
    for (int z = 0; z < dz; z++) for (int y = 0; y < dy; y++) for (int x = 0; x < dx; x++) {
        out[((size_t)z * dy + y) * dx + x] = blur[(size_t)(2 * z) * s_z + (size_t)(2 * y) * s_y + (2 * x)];
    }
    free(blur); free(tmp);
    return 0;
}

/* ---- similarity metric: NCC over the overlap region -------------------------
 * Compute NCC between `fixed` and the moving volume warped by M (sampled on the
 * fly). Only voxels whose moving-sample is in-bounds count toward the overlap.
 * To distinguish genuine in-bounds (possibly 0-valued) samples from
 * out-of-bounds, we test the mapped coordinate explicitly.
 *
 * NCC = sum((f-mf)*(m-mm)) / sqrt(sum((f-mf)^2)*sum((m-mm)^2)), range [-1,1],
 * 1 = perfect correlation. Invariant to affine intensity change of either image
 * -> robust to brightness/contrast differences between scans. */
double fy_ncc_warped(const float *fixed, const float *moving,
                     int nz, int ny, int nx, const double *M) {
    size_t s_y = (size_t)nx, s_z = (size_t)ny * nx;
    double sf = 0, sm = 0, sff = 0, smm = 0, sfm = 0;
    long cnt = 0;
    for (int zo = 0; zo < nz; zo++) {
        for (int yo = 0; yo < ny; yo++) {
            double bz = M[0] * zo + M[1] * yo + M[3];
            double by = M[4] * zo + M[5] * yo + M[7];
            double bx = M[8] * zo + M[9] * yo + M[11];
            const float *frow = fixed + (size_t)zo * s_z + (size_t)yo * s_y;
            for (int xo = 0; xo < nx; xo++) {
                float zi = (float)(bz + M[2] * xo);
                float yi = (float)(by + M[6] * xo);
                float xi = (float)(bx + M[10] * xo);
                /* in-bounds test (matches sampler) */
                if (zi < 0.0f || yi < 0.0f || xi < 0.0f ||
                    zi > (float)(nz - 1) || yi > (float)(ny - 1) || xi > (float)(nx - 1))
                    continue;
                float mv = fy_sample_trilinear(moving, nz, ny, nx, zi, yi, xi);
                float fv = frow[xo];
                sf += fv; sm += mv;
                sff += (double)fv * fv; smm += (double)mv * mv;
                sfm += (double)fv * mv;
                cnt++;
            }
        }
    }
    if (cnt < 32) return -2.0;   /* essentially no overlap: sentinel < -1 */
    double mf = sf / cnt, mm = sm / cnt;
    double cov = sfm / cnt - mf * mm;
    double vf = sff / cnt - mf * mf;
    double vm = smm / cnt - mm * mm;
    double denom = sqrt(vf * vm);
    if (denom < 1e-12) return 0.0;   /* one image is flat over overlap */
    return cov / denom;
}

/* ---- parameter <-> matrix conversions --------------------------------------
 * Params layout:
 *   rigid (rigid_only): 7 params [rz,ry,rx (rad), tz,ty,tx (vox), s (log-scale)]
 *     isotropic scale = exp(s). (log keeps the optimizer symmetric around 1.)
 *   affine: 12 params = the 3x4 matrix directly, as a DELTA from a base
 *     (identity-ish) so the optimizer searches near identity. We store affine
 *     as full 12 and optimize them directly.
 *
 * For the matrix we always compose: M = Tcenter * (linear) * Tcenter^-1, i.e.
 * rotation/scale about the volume CENTER (so a rotation doesn't also fling the
 * volume across the grid). Center is the fixed-volume center.
 */
typedef struct { double cz, cy, cx; } fy_center;

static void rigid_params_to_M(const double *p, fy_center c, double *M) {
    double rz = p[0], ry = p[1], rx = p[2];
    double tz = p[3], ty = p[4], tx = p[5];
    double s = exp(p[6]);
    /* rotation: Rz * Ry * Rx (about z, y, x axes), scaled by s */
    double cz = cos(rz), sz = sin(rz);
    double cy = cos(ry), sy = sin(ry);
    double cx = cos(rx), sx = sin(rx);
    /* R = Rz*Ry*Rx, acting on (z,y,x) vectors */
    double Rz[9] = { cz,-sz,0, sz,cz,0, 0,0,1 };
    double Ry[9] = { cy,0,sy, 0,1,0, -sy,0,cy };
    double Rx[9] = { 1,0,0, 0,cx,-sx, 0,sx,cx };
    double Rzy[9], R[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        double a = 0; for (int k = 0; k < 3; k++) a += Rz[i*3+k] * Ry[k*3+j];
        Rzy[i*3+j] = a;
    }
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        double a = 0; for (int k = 0; k < 3; k++) a += Rzy[i*3+k] * Rx[k*3+j];
        R[i*3+j] = a * s;
    }
    /* M linear = R ; translation so that center maps with offset (tz,ty,tx):
     * Mi = R*(o - c) + c + t  => Mi = R*o + (c - R*c + t) */
    double cc[3] = { c.cz, c.cy, c.cx };
    double t[3]  = { tz, ty, tx };
    for (int i = 0; i < 3; i++) {
        M[i*4+0] = R[i*3+0]; M[i*4+1] = R[i*3+1]; M[i*4+2] = R[i*3+2];
        double Rc = R[i*3+0]*cc[0] + R[i*3+1]*cc[1] + R[i*3+2]*cc[2];
        M[i*4+3] = cc[i] - Rc + t[i];
    }
}

/* affine params are the 9 linear entries (delta around base linear) + 3 trans,
 * also composed about the center. p[0..8] = linear row-major, p[9..11] = trans. */
static void affine_params_to_M(const double *p, fy_center c, double *M) {
    double R[9]; for (int i = 0; i < 9; i++) R[i] = p[i];
    double cc[3] = { c.cz, c.cy, c.cx };
    double t[3]  = { p[9], p[10], p[11] };
    for (int i = 0; i < 3; i++) {
        M[i*4+0] = R[i*3+0]; M[i*4+1] = R[i*3+1]; M[i*4+2] = R[i*3+2];
        double Rc = R[i*3+0]*cc[0] + R[i*3+1]*cc[1] + R[i*3+2]*cc[2];
        M[i*4+3] = cc[i] - Rc + t[i];
    }
}

/* ---- coordinate-descent optimizer (robust, derivative-free) -----------------
 * Maximize NCC by sweeping each parameter +/- a step, accepting improvements,
 * shrinking the step when no parameter improves. Simple, robust, no line-search
 * pathologies. Per-pyramid-level with a budget of sweeps. */
typedef void (*params_to_M_fn)(const double *p, fy_center c, double *M);

static double optimize_level(const float *fixed, const float *moving,
                             int nz, int ny, int nx, fy_center c,
                             double *p, int np, const double *step0,
                             params_to_M_fn to_M, int max_sweeps) {
    double M[12];
    to_M(p, c, M);
    double best = fy_ncc_warped(fixed, moving, nz, ny, nx, M);
    double *step = malloc(sizeof(double) * np);
    for (int i = 0; i < np; i++) step[i] = step0[i];

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        int improved = 0;
        for (int i = 0; i < np; i++) {
            for (int dir = -1; dir <= 1; dir += 2) {
                double save = p[i];
                p[i] = save + dir * step[i];
                to_M(p, c, M);
                double v = fy_ncc_warped(fixed, moving, nz, ny, nx, M);
                if (v > best + 1e-7) { best = v; improved = 1; }
                else p[i] = save;
            }
        }
        if (!improved) {
            /* shrink all steps; stop when steps are tiny */
            double maxstep = 0;
            for (int i = 0; i < np; i++) { step[i] *= 0.5; if (step[i] > maxstep) maxstep = step[i]; }
            if (maxstep < 1e-4) break;
        }
    }
    free(step);
    return best;
}

/* ---- public registration entry point ---------------------------------------
 * Coarse-to-fine intensity registration maximizing NCC(fixed, warp(moving,M)).
 *
 * M_out: 3x4 (12 doubles). On input it is the INITIAL guess (output->input map);
 *        pass identity for "same grid" or a scale-ratio diagonal for known voxel
 *        size ratio. On output it is the optimized map.
 * rigid_only: 1 = rotation + translation + isotropic scale (7 dof). 0 = full
 *        affine (12 dof).
 *
 * Both fixed and moving are sampled on the SAME grid (nz,ny,nx). If your two
 * scans have different voxel sizes, resample/crop them to a common grid first
 * (or encode the scale ratio in the initial M_out). */
int fy_register_affine(const float *fixed, const float *moving,
                       int nz, int ny, int nx, double *M_out, int rigid_only) {
    if (!fixed || !moving || !M_out) return 1;

    /* ---- build pyramids (up to 4 levels, stop when a dim < 16) ---- */
    const int MAXLVL = 4;
    const float *fpyr[MAXLVL], *mpyr[MAXLVL];
    int pz[MAXLVL], py[MAXLVL], px[MAXLVL];
    float *fbuf[MAXLVL], *mbuf[MAXLVL];   /* owned allocations (level>0) */
    for (int l = 0; l < MAXLVL; l++) { fbuf[l] = mbuf[l] = NULL; }

    fpyr[0] = fixed; mpyr[0] = moving;
    pz[0] = nz; py[0] = ny; px[0] = nx;
    int nlvl = 1;
    for (int l = 1; l < MAXLVL; l++) {
        int cz = pz[l-1], cy = py[l-1], cx = px[l-1];
        if (cz/2 < 16 || cy/2 < 16 || cx/2 < 16) break;
        int dz = cz/2, dy = cy/2, dx = cx/2;
        size_t dn = (size_t)dz * dy * dx;
        fbuf[l] = malloc(sizeof(float) * dn);
        mbuf[l] = malloc(sizeof(float) * dn);
        if (!fbuf[l] || !mbuf[l]) { free(fbuf[l]); free(mbuf[l]); fbuf[l]=mbuf[l]=NULL; break; }
        fy_downsample2x(fpyr[l-1], fbuf[l], cz, cy, cx, &pz[l], &py[l], &px[l]);
        fy_downsample2x(mpyr[l-1], mbuf[l], cz, cy, cx, &pz[l], &py[l], &px[l]);
        fpyr[l] = fbuf[l]; mpyr[l] = mbuf[l];
        nlvl++;
    }

    /* ---- decompose initial M_out into our parameter set --------------------
     * We optimize in parameter space (about each level's center). For the
     * initial guess we extract translation directly and, for rigid, the
     * isotropic scale from the matrix; rotation starts at 0. For affine we seed
     * the 12 params from M_out directly. The translation must be RESCALED per
     * pyramid level (coarse levels have coords / 2^l). */
    double p[12];
    int np;
    params_to_M_fn to_M;
    if (rigid_only) {
        np = 7; to_M = rigid_params_to_M;
        /* isotropic scale ~ cube-root of |det(linear)| */
        double *L = M_out;
        double det =
            L[0]*(L[5]*L[10]-L[6]*L[9]) - L[1]*(L[4]*L[10]-L[6]*L[8]) + L[2]*(L[4]*L[9]-L[5]*L[8]);
        double s = cbrt(fabs(det)); if (s < 1e-6) s = 1.0;
        p[0]=p[1]=p[2]=0.0;            /* rotation */
        p[3]=M_out[3]; p[4]=M_out[7]; p[5]=M_out[11];   /* translation (level 0) */
        p[6]=log(s);
    } else {
        np = 12; to_M = affine_params_to_M;
        p[0]=M_out[0]; p[1]=M_out[1]; p[2]=M_out[2];
        p[3]=M_out[4]; p[4]=M_out[5]; p[5]=M_out[6];
        p[6]=M_out[8]; p[7]=M_out[9]; p[8]=M_out[10];
        p[9]=M_out[3]; p[10]=M_out[7]; p[11]=M_out[11];
    }

    /* translation indices (in p) for per-level rescaling */
    int ti[3];
    if (rigid_only) { ti[0]=3; ti[1]=4; ti[2]=5; }
    else            { ti[0]=9; ti[1]=10; ti[2]=11; }

    /* ---- coarse-to-fine sweep ---- */
    for (int l = nlvl - 1; l >= 0; l--) {
        double scale = ldexp(1.0, -l);   /* 1/2^l : level-l coords are scaled */
        /* rescale translation params from level-0 units into this level */
        for (int k = 0; k < 3; k++) p[ti[k]] *= scale;

        fy_center c = { (pz[l]-1)*0.5, (py[l]-1)*0.5, (px[l]-1)*0.5 };

        double step[12];
        if (rigid_only) {
            double rstep = 0.06;                /* ~3.4 deg */
            double tstep = 2.0 * scale;         /* voxels, scaled to level */
            if (tstep < 0.5) tstep = 0.5;
            step[0]=step[1]=step[2]=rstep;
            step[3]=step[4]=step[5]=tstep;
            step[6]=0.04;                       /* log-scale */
        } else {
            for (int i = 0; i < 9; i++) step[i] = 0.04;   /* linear entries */
            double tstep = 2.0 * scale; if (tstep < 0.5) tstep = 0.5;
            step[9]=step[10]=step[11]=tstep;
        }

        int sweeps = 60;
        optimize_level(fpyr[l], mpyr[l], pz[l], py[l], px[l], c, p, np, step, to_M, sweeps);

        /* rescale translation back to level-0 units for the next finer level */
        for (int k = 0; k < 3; k++) p[ti[k]] /= scale;
    }

    /* ---- emit final M at full resolution ---- */
    fy_center c0 = { (nz-1)*0.5, (ny-1)*0.5, (nx-1)*0.5 };
    to_M(p, c0, M_out);

    for (int l = 1; l < nlvl; l++) { free(fbuf[l]); free(mbuf[l]); }
    return 0;
}
