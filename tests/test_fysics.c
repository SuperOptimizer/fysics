/* test_fysics.c -- correctness tests for the C kernels. */
#include "fysics.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
                              else { printf("ok:   %s\n", msg); } } while (0)

/* naive DFT for cross-checking the FFT on a small size */
static void naive_dft(const float *re, const float *im, float *ore, float *oim, int n, int sign) {
    for (int k = 0; k < n; k++) {
        double sr = 0, si = 0;
        for (int t = 0; t < n; t++) {
            double a = sign * 2.0 * M_PI * k * t / n;
            double c = cos(a), s = sin(a);
            sr += re[t] * c - im[t] * s;
            si += re[t] * s + im[t] * c;
        }
        ore[k] = (float)sr; oim[k] = (float)si;
    }
}

static void test_fft_vs_dft(void) {
    int n = 16;
    float re[16], im[16], dre[16], dim[16];
    for (int i = 0; i < n; i++) { re[i] = sinf(i * 0.7f) + 0.3f * i; im[i] = 0; }
    float re2[16], im2[16];
    for (int i = 0; i < n; i++) { re2[i] = re[i]; im2[i] = im[i]; }
    naive_dft(re, im, dre, dim, n, -1);
    fy_fft1d(re2, im2, n, -1);
    double maxerr = 0;
    for (int i = 0; i < n; i++) {
        double e = fabs(re2[i] - dre[i]) + fabs(im2[i] - dim[i]);
        if (e > maxerr) maxerr = e;
    }
    CHECK(maxerr < 1e-3, "fft1d matches naive DFT");
}

static void test_fft_roundtrip(void) {
    int nz = 8, ny = 8, nx = 8, n = nz*ny*nx;
    float *re = malloc(sizeof(float)*n), *im = calloc(n, sizeof(float));
    float *orig = malloc(sizeof(float)*n);
    for (int i = 0; i < n; i++) { re[i] = (float)((i*2654435761u) % 1000) / 1000.0f; orig[i] = re[i]; }
    fy_fft3d(re, im, nz, ny, nx, -1);
    fy_fft3d(re, im, nz, ny, nx, +1);
    fy_fft3d_normalize(re, im, nz, ny, nx);
    double maxerr = 0;
    for (int i = 0; i < n; i++) { double e = fabs(re[i] - orig[i]); if (e > maxerr) maxerr = e; }
    CHECK(maxerr < 1e-4, "fft3d forward+inverse round-trips");
    free(re); free(im); free(orig);
}

static void test_fft_radix3(void) {
    /* 1-D radix-3 (n = 3*2^k) vs naive DFT */
    int n = 48;
    float re[48], im[48], dre[48], dim[48], re2[48], im2[48];
    for (int i = 0; i < n; i++) {
        re[i] = sinf(i * 0.7f) + 0.3f * cosf(i * 1.3f);
        im[i] = 0.2f * sinf(i * 0.31f);
        re2[i] = re[i]; im2[i] = im[i];
    }
    naive_dft(re, im, dre, dim, n, -1);
    fy_fft1d(re2, im2, n, -1);
    double maxerr = 0;
    for (int i = 0; i < n; i++) {
        double e = fabs(re2[i] - dre[i]) + fabs(im2[i] - dim[i]);
        if (e > maxerr) maxerr = e;
    }
    CHECK(maxerr < 1e-3, "fft1d radix-3 (n=48) matches naive DFT");

    /* 3-D round-trip on mixed sizes (12, 8, 24) incl. radix-3 axes */
    int nz = 12, ny = 8, nx = 24, nn = nz*ny*nx;
    float *rr = malloc(sizeof(float)*nn), *ii = calloc(nn, sizeof(float));
    float *orig = malloc(sizeof(float)*nn);
    for (int i = 0; i < nn; i++) { rr[i] = (float)((i*2654435761u) % 1000) / 1000.0f; orig[i] = rr[i]; }
    fy_fft3d(rr, ii, nz, ny, nx, -1);
    fy_fft3d(rr, ii, nz, ny, nx, +1);
    fy_fft3d_normalize(rr, ii, nz, ny, nx);
    maxerr = 0;
    for (int i = 0; i < nn; i++) { double e = fabs(rr[i] - orig[i]); if (e > maxerr) maxerr = e; }
    CHECK(maxerr < 1e-4, "fft3d radix-3 sizes round-trip");
    free(rr); free(ii); free(orig);

    CHECK(fy_next_fft_size(176) == 192 && fy_next_fft_size(128) == 128 &&
          fy_next_fft_size(160) == 192 && fy_next_fft_size(97) == 128,
          "fy_next_fft_size picks 3*2^k when smaller");
}

static void test_guided_fast(void) {
    /* fast (s=2) vs exact guided filter: same smoothing, small pointwise difference */
    int nz = 32, ny = 32, nx = 32, n = nz*ny*nx;
    float *in = malloc(sizeof(float)*n), *e = malloc(sizeof(float)*n), *fst = malloc(sizeof(float)*n);
    float *ws = malloc(sizeof(float)*fy_guided_ws_floats(nz,ny,nx));
    unsigned int rng = 12345;
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        rng = rng*1664525u + 1013904223u;
        float noise = ((rng >> 8) & 0xFFFF) / 65535.0f - 0.5f;
        float sig = (x < nx/2) ? 0.3f : 0.7f;        /* step edge + noise */
        in[((size_t)z*ny+y)*nx+x] = sig + 0.05f * noise;
    }
    double eps = 0.01*0.01 * 4;
    fy_guided_denoise_ws(in, e, nz, ny, nx, 2, eps, ws);
    fy_guided_denoise_fast_ws(in, fst, nz, ny, nx, 2, eps, 2, ws);
    double mad = 0, edge_exact = 0, edge_fast = 0;
    for (int i = 0; i < n; i++) mad += fabs(fst[i] - e[i]);
    mad /= n;
    /* edge must survive in both: mean difference across the step */
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) {
        size_t L = ((size_t)z*ny+y)*nx + nx/2 - 4, R = ((size_t)z*ny+y)*nx + nx/2 + 4;
        edge_exact += e[R] - e[L]; edge_fast += fst[R] - fst[L];
    }
    edge_exact /= nz*ny; edge_fast /= nz*ny;
    CHECK(mad < 0.01, "fast guided filter tracks exact (mean |diff| < 0.01)");
    CHECK(edge_fast > 0.8 * edge_exact, "fast guided filter preserves edges");
    free(in); free(e); free(fst); free(ws);
}

static void test_dering(void) {
    /* phantom: spiral winding (must SURVIVE) + true rings (must be detected+removed) + noise */
    int nz = 32, ny = 256, nx = 256;
    double cy = 127.5, cx = 127.5;
    size_t n = (size_t)nz * ny * nx;
    unsigned char *vol = malloc(n);
    unsigned int rng = 777;
    const double R1 = 60, A1 = +10, R2 = 110, A2 = -8;  /* injected rings (u8) */
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                double dy = y - cy, dx = x - cx;
                double r = sqrt(dy*dy + dx*dx), th = atan2(dy, dx);
                /* spiral wraps: radius drifts 192 px/turn (period 24) -> a full wrap
                 * period crosses each 45-degree sector, like the real winding */
                double v = 128 + 30 * cos(2*M_PI*(r - 192*th/(2*M_PI)) / 24);
                v += A1 * exp(-(r-R1)*(r-R1)/(2*0.8*0.8));   /* ring 1 */
                v += A2 * exp(-(r-R2)*(r-R2)/(2*0.8*0.8));   /* ring 2 */
                rng = rng*1664525u + 1013904223u;
                v += ((rng >> 8) & 0xFF) / 25.5 - 5.0;        /* +-5 noise */
                vol[((size_t)z*ny+y)*nx+x] = (unsigned char)(v < 1 ? 1 : (v > 255 ? 255 : v));
            }
    fy_dering d;
    CHECK(fy_dering_init(&d, nz, ny, nx, -1, -1, 64, 8) == 0, "dering init");
    fy_dering dt;
    CHECK(fy_dering_tile_init(&dt, &d) == 0, "dering tile init");
    fy_dering_accumulate_u8(&dt, vol, 0, 0, nz, ny, nx, 1);
    fy_dering_merge_tile(&d, 0, &dt);
    long ndet = fy_dering_finalize(&d, 15, 50, 2.0, 15.0);
    CHECK(d.have_rings, "dering detects rings");
    /* detected at the injected radii, correct signs */
    float r1 = 0, r2 = 0;
    for (int r = (int)R1-1; r <= (int)R1+1; r++) if (fabsf(d.ring[r]) > fabsf(r1)) r1 = d.ring[r];
    for (int r = (int)R2-1; r <= (int)R2+1; r++) if (fabsf(d.ring[r]) > fabsf(r2)) r2 = d.ring[r];
    CHECK(r1 > 4.0f, "ring 1 (+10 u8) detected");
    CHECK(r2 < -3.0f, "ring 2 (-8 u8) detected");
    /* spiral must NOT trigger detections away from the rings */
    int false_pos = 0;
    for (int r = 20; r < 120; r++)
        if (fabs(r - R1) > 4 && fabs(r - R2) > 4 && d.ring[r] != 0) false_pos++;
    CHECK(false_pos <= 3, "spiral winding not flagged as rings (sector vote)");
    /* apply and verify the ring is actually removed from the volume */
    float *f = malloc(sizeof(float)*n);
    for (size_t i = 0; i < n; i++) f[i] = vol[i] * (1.0f/255.0f);
    fy_dering_apply(&d, f, 0, 0, 0, nz, ny, nx, 1.0/255.0);
    /* residual ring at R1: angular mean minus local radial baseline, before vs after */
    double before = 0, after = 0, base_b = 0, base_a = 0; long c1 = 0, cb = 0;
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                double r = sqrt((y-cy)*(y-cy) + (x-cx)*(x-cx));
                size_t i = ((size_t)z*ny+y)*nx+x;
                if (fabs(r - R1) < 1.0) { before += vol[i]/255.0; after += f[i]; c1++; }
                else if (fabs(r - R1) > 5 && fabs(r - R1) < 8) { base_b += vol[i]/255.0; base_a += f[i]; cb++; }
            }
    double resid_b = before/c1 - base_b/cb, resid_a = after/c1 - base_a/cb;
    CHECK(fabs(resid_a) < 0.4 * fabs(resid_b), "apply removes >60% of ring amplitude");
    (void)ndet;
    fy_dering_free(&dt); fy_dering_free(&d); free(vol); free(f);
}

/* full-complex reference deconv (the pre-rfft implementation, via the public FFT):
 * reflect-pad to fy_next_fft_size, fft3d, multiply by capped Wiener weight, ifft, crop. */
static int ref_reflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) { if (i < 0) i = -i - 1; if (i >= n) i = 2 * n - i - 1; }
    return i;
}
static void ref_deconvolve(const float *in, float *out, int nz, int ny, int nx,
                           const fy_physics *p, double reg) {
    int pz = fy_next_fft_size(nz), py = fy_next_fft_size(ny), px = fy_next_fft_size(nx);
    size_t np = (size_t)pz * py * px;
    float *re = malloc(sizeof(float) * np), *im = calloc(np, sizeof(float));
    for (int z = 0; z < pz; z++) for (int y = 0; y < py; y++) {
        int sz = ref_reflect(z, nz), sy = ref_reflect(y, ny);
        for (int x = 0; x < px; x++)
            re[((size_t)z*py + y)*px + x] = in[((size_t)sz*ny + sy)*nx + ref_reflect(x, nx)];
    }
    fy_fft3d(re, im, pz, py, px, -1);
    for (int z = 0; z < pz; z++) for (int y = 0; y < py; y++) for (int x = 0; x < px; x++) {
        double fz = (z <= pz/2) ? (double)z/pz : (double)(z-pz)/pz;
        double fy = (y <= py/2) ? (double)y/py : (double)(y-py)/py;
        double fx = (x <= px/2) ? (double)x/px : (double)(x-px)/px;
        double fr = sqrt(fz*fz + fy*fy + fx*fx);
        double H = fy_recon_transfer(fr, p);
        double w = H / (H*H + reg);
        if (w > 8.0) w = 8.0;                       /* FY_MAX_DECONV_GAIN */
        re[((size_t)z*py + y)*px + x] *= (float)w;
        im[((size_t)z*py + y)*px + x] *= (float)w;
    }
    fy_fft3d(re, im, pz, py, px, +1);
    fy_fft3d_normalize(re, im, pz, py, px);
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++)
        out[((size_t)z*ny + y)*nx + x] = re[((size_t)z*py + y)*px + x];
    free(re); free(im);
}

static void test_deconv_rfft_equivalence(void) {
    /* the half-spectrum (R2C) deconv must match the full-complex reference,
     * incl. odd crop dims (exercises the unpaired-row paths) */
    int nz = 22, ny = 27, nx = 30;                  /* pads to 24 x 32 x 32 */
    size_t n = (size_t)nz * ny * nx;
    float *in = malloc(sizeof(float)*n), *a = malloc(sizeof(float)*n), *b = malloc(sizeof(float)*n);
    unsigned int rng = 99;
    for (size_t i = 0; i < n; i++) {
        rng = rng*1664525u + 1013904223u;
        in[i] = ((rng >> 8) & 0xFFFF) / 65535.0f;
    }
    fy_physics p = {1000.0, 78.0, 220.0, 2.4, 1.2, 4.0};
    CHECK(fy_deconvolve(in, a, nz, ny, nx, &p, 0.015) == 0, "rfft deconv runs");
    ref_deconvolve(in, b, nz, ny, nx, &p, 0.015);
    double maxd = 0, scale = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]); if (d > maxd) maxd = d;
        double s = fabs(b[i]); if (s > scale) scale = s;
    }
    CHECK(maxd < 1e-3 * (scale > 1 ? scale : 1), "rfft deconv == full-complex reference (LUT-tolerance)");
    free(in); free(a); free(b);
}

static void test_downscale_cbox(void) {
    /* a thin bright sheet (1 voxel of 200 in a 40-background cell) must survive
     * cbox visibly better than box; zero cells stay zero; box = exact mean */
    unsigned char in[2*2*2*8]; int ox,oy,oz;
    unsigned char out_box[8], out_cbox[8];
    /* volume 4x2x2: first 2x2x2 cell = sheet, second = all zero */
    unsigned char v[16]={40,40,40,40, 40,40,40,200,  0,0,0,0, 0,0,0,0};
    (void)in;
    fy_downscale2x(v,4,2,2,out_box,&ox,&oy,&oz,FY_DS_BOX,0.5f);
    fy_downscale2x(v,4,2,2,out_cbox,&ox,&oy,&oz,FY_DS_CBOX,0.5f);
    CHECK(ox==2&&oy==1&&oz==1,"downscale dims");
    CHECK(out_box[0]==60,"box = exact mean (60)");
    CHECK(out_cbox[0]==130,"cbox pushes toward the sheet (mean+0.5*(200-60)=130)");
    CHECK(out_box[1]==0&&out_cbox[1]==0,"zero cells stay zero (occupancy-safe)");
}

static void test_noise_aniso(void) {
    /* white noise blurred 3-tap along z -> rho_z >> rho_xy; ratio detects it */
    int nz = 64, ny = 64, nx = 64; size_t n = (size_t)nz*ny*nx;
    float *v = malloc(sizeof(float)*n), *b = malloc(sizeof(float)*n);
    unsigned int rng = 42;
    for (size_t i = 0; i < n; i++) {
        rng = rng*1664525u + 1013904223u;
        v[i] = 0.5f + (((rng >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 0.1f;
    }
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        size_t i = ((size_t)z*ny+y)*nx+x;
        size_t zm = z > 0 ? i - (size_t)ny*nx : i, zp = z < nz-1 ? i + (size_t)ny*nx : i;
        b[i] = 0.25f*v[zm] + 0.5f*v[i] + 0.25f*v[zp];
    }
    double rz, rxy;
    CHECK(fy_noise_aniso(b, nz, ny, nx, &rz, &rxy) == 0, "noise aniso runs");
    CHECK(rz > rxy + 0.1, "z-blurred noise: rho_z > rho_xy");
    double rz2, rxy2;
    CHECK(fy_noise_aniso(v, nz, ny, nx, &rz2, &rxy2) == 0 && fabs(rz2 - rxy2) < 0.08,
          "isotropic noise: rho_z ~ rho_xy");
    free(v); free(b);
}

static void blur3_axis(const float *in, float *out, int nz, int ny, int nx, int axis) {
    size_t s = axis == 0 ? (size_t)ny*nx : (axis == 1 ? (size_t)nx : 1);
    int len[3] = {nz, ny, nx};
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        size_t i = ((size_t)z*ny+y)*nx+x;
        int pos = axis == 0 ? z : (axis == 1 ? y : x);
        size_t m = pos > 0 ? i - s : i, p = pos < len[axis]-1 ? i + s : i;
        out[i] = 0.25f*in[m] + 0.5f*in[i] + 0.25f*in[p];
    }
}

static void test_matched_aniso(void) {
    /* volume blurred MORE in z; the anisotropic matched inverse recovers z detail better */
    int nz = 48, ny = 48, nx = 48; size_t n = (size_t)nz*ny*nx;
    float *v = malloc(sizeof(float)*n), *bl = malloc(sizeof(float)*n), *tmp = malloc(sizeof(float)*n);
    float *iso = malloc(sizeof(float)*n), *ani = malloc(sizeof(float)*n);
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++)
        v[((size_t)z*ny+y)*nx+x] = 0.5f + 0.2f*sinf(z*1.1f) + 0.2f*sinf(x*1.1f);
    blur3_axis(v, tmp, nz, ny, nx, 0);          /* z, twice (broader) */
    blur3_axis(tmp, bl, nz, ny, nx, 0);
    blur3_axis(bl, tmp, nz, ny, nx, 2);         /* x, once */
    memcpy(bl, tmp, sizeof(float)*n);
    fy_physics p = {1000, 78, 220, 2.4, 0, 0, 1.0, 0};   /* PSF only, no unsharp */
    fy_deconvolve_matched(bl, iso, nz, ny, nx, &p, 0.05);
    p.psf_sigma_z_vox = 1.4;
    fy_deconvolve_matched(bl, ani, nz, ny, nx, &p, 0.05);
    double dz_t = 0, dz_i = 0, dz_a = 0;
    int fin = 1;
    for (int z = 1; z < nz-1; z++) for (int y = 8; y < ny-8; y++) for (int x = 8; x < nx-8; x++) {
        size_t i = ((size_t)z*ny+y)*nx+x, s = (size_t)ny*nx;
        dz_t += fabs(v[i+s]-v[i-s]); dz_i += fabs(iso[i+s]-iso[i-s]); dz_a += fabs(ani[i+s]-ani[i-s]);
        if (!isfinite(ani[i])) fin = 0;
    }
    CHECK(fin, "aniso matched deconv output finite");
    CHECK(dz_a > dz_i * 1.02, "aniso inverse recovers more z detail than isotropic");
    CHECK(dz_a < dz_t * 1.5, "aniso inverse does not overshoot wildly");
    free(v); free(bl); free(tmp); free(iso); free(ani);
}

static void test_paganin_transfer(void) {
    fy_physics p = {1000.0, 78.0, 220.0, 2.4, 1.2, 4.0};
    /* nabu reference values (cycles/voxel grid, px=2.4): computed in python */
    double T0  = fy_paganin_transfer(0.0, &p);
    double Tmid = fy_paganin_transfer(0.1, &p);
    double Thi = fy_paganin_transfer(0.4, &p);
    CHECK(fabs(T0 - 1.0) < 1e-9, "paganin T(0) == 1");
    CHECK(Tmid < T0 && Thi < Tmid, "paganin T decreasing (low-pass)");
    /* exact nabu value at f=0.1 cyc/voxel: 1/(1+1000*L*D*pi*(0.1/2.4)^2) */
    double L = 1.23984199e-3/78.0, D = 220000.0, f = 0.1/2.4;
    double ref = 1.0/(1.0 + 1000.0*L*D*M_PI*f*f);
    CHECK(fabs(Tmid - ref) < 1e-9, "paganin matches nabu formula exactly");
}

static void test_deconvolve_sharpens(void) {
    fy_physics p = {1000.0, 78.0, 220.0, 2.4, 1.2, 4.0};
    int nz = 32, ny = 32, nx = 32, n = nz*ny*nx;
    float *in = malloc(sizeof(float)*n), *out = malloc(sizeof(float)*n);
    /* a smooth ramp + bump (low-freq dominated) */
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        double r = (x-16.0)*(x-16.0)+(y-16.0)*(y-16.0)+(z-16.0)*(z-16.0);
        in[(z*ny+y)*nx+x] = (float)exp(-r/50.0);
    }
    int rc = fy_deconvolve(in, out, nz, ny, nx, &p, 0.05);
    CHECK(rc == 0, "deconvolve returns success");
    /* output finite and shape preserved (values change) */
    int finite = 1, changed = 0;
    for (int i = 0; i < n; i++) { if (!isfinite(out[i])) finite = 0; if (fabsf(out[i]-in[i])>1e-4f) changed = 1; }
    CHECK(finite, "deconvolve output is finite");
    CHECK(changed, "deconvolve modifies the volume");
    free(in); free(out);
}

static void test_halo_reasonable(void) {
    fy_physics p = {1000.0, 78.0, 220.0, 2.4, 1.2, 4.0};
    int h = fy_kernel_halo(&p);
    CHECK(h >= 8 && h <= 96, "kernel halo in [8,96] (local neighborhood)");
    printf("      (halo = %d voxels)\n", h);
}























static void test_estimate_noise(void){
    /* synthetic volume: smooth signal + KNOWN white noise of std sigma.
     * The estimator should recover noise_ref close to sigma. */
    int nz=48,ny=48,nx=48; size_t n=(size_t)nz*ny*nx;
    float *v=malloc(4*n);
    unsigned int rng=12345u;
    double sigma=0.05;
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        /* slowly-varying signal in [0.2,0.8] so flat regions exist */
        double s=0.5+0.3*sin(x*0.10)*cos(y*0.09);
        /* box-muller-ish cheap gaussian from two uniforms */
        rng=rng*1664525u+1013904223u; double u1=((rng>>8)&0xffffff)/(double)0x1000000;
        rng=rng*1664525u+1013904223u; double u2=((rng>>8)&0xffffff)/(double)0x1000000;
        if(u1<1e-9)u1=1e-9;
        double g0=sqrt(-2*log(u1))*cos(6.283185307*u2);
        v[((size_t)z*ny+y)*nx+x]=(float)(s+sigma*g0);
    }
    fy_noise_model nm;
    int rc=fy_estimate_noise(v,nz,ny,nx,5,10.0,0.5,&nm);
    CHECK(rc==0, "fy_estimate_noise runs");
    /* recovered noise std within ~40% of truth (local-window est is approximate) */
    CHECK(nm.noise_ref > 0.5*sigma && nm.noise_ref < 1.7*sigma,
          "estimated noise_ref ~ true sigma");
    /* higher-noise volume must estimate a higher level (monotone, the key property) */
    float *v2=malloc(4*n); double sigma2=0.10;
    rng=999u;
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        double s=0.5+0.3*sin(x*0.10)*cos(y*0.09);
        rng=rng*1664525u+1013904223u; double u1=((rng>>8)&0xffffff)/(double)0x1000000;
        rng=rng*1664525u+1013904223u; double u2=((rng>>8)&0xffffff)/(double)0x1000000;
        if(u1<1e-9)u1=1e-9;
        double g0=sqrt(-2*log(u1))*cos(6.283185307*u2);
        v2[((size_t)z*ny+y)*nx+x]=(float)(s+sigma2*g0);
    }
    fy_noise_model nm2;
    fy_estimate_noise(v2,nz,ny,nx,5,10.0,0.5,&nm2);
    CHECK(nm2.noise_ref > nm.noise_ref, "noisier volume -> higher estimated noise (monotone)");
    /* eps mapping: higher noise -> larger eps, monotone & positive */
    double e1=fy_guided_eps_for_noise(nm.noise_ref), e2=fy_guided_eps_for_noise(nm2.noise_ref);
    CHECK(e1>0 && e2>e1, "guided eps grows with noise level");
    free(v); free(v2);
}





static void test_deltabeta_scale(void){
    /* fine/strong-filter volume (small H_nyq) -> partial inversion (<1);
     * coarse/mild-filter volume (large H_nyq) -> full inversion (~1). */
    fy_physics fine = {500,59,200,1.129,2.5,4.0,0.0};   /* 1.129um, strong filter */
    fy_physics coarse = {1000,113,1200,9.362,1.2,4.0,0.0}; /* 9.362um */
    double sf=fy_auto_deltabeta_scale(&fine), sc=fy_auto_deltabeta_scale(&coarse);
    CHECK(sf>=0.25 && sf<0.6, "fine volume -> partial delta_beta inversion (~0.35)");
    CHECK(sc>0.9, "coarse volume -> full delta_beta inversion (~1.0)");
    CHECK(sc>sf, "coarse inverts more fully than fine (regime ordering)");
}

static void test_dewindow(void){
    /* exact linear round-trip u8 -> physical -> u8 (per-volume window) */
    double f0=-0.03, f1=0.21;
    unsigned char u8[256]; for(int i=0;i<256;i++) u8[i]=(unsigned char)i;
    float phys[256]; unsigned char back[256];
    fy_u8_to_phys(u8, phys, 256, f0, f1);
    fy_phys_to_u8(phys, back, 256, f0, f1);
    int maxerr=0; for(int i=0;i<256;i++){int e=abs((int)back[i]-(int)u8[i]); if(e>maxerr)maxerr=e;}
    CHECK(maxerr==0, "u8<->phys window round-trips exactly");
    /* physical endpoints land where expected */
    CHECK(fabs(phys[0]-f0)<1e-6 && fabs(phys[255]-f1)<1e-6, "phys window maps 0..255 -> f0..f1");
    /* a fixed physical value maps to different u8 levels under different windows */
    float a=fy_phys_to_u8_level(0.0,-0.03,0.145), b=fy_phys_to_u8_level(0.0,-0.03,0.21);
    CHECK(fabs(a-b)>1.0, "fixed physical level -> different u8 per volume (the whole point)");
}



/* ============ registration (register.c) ================================== */

/* structured test volume: a few gaussian blobs + low-freq sinusoid so NCC has
 * gradient to follow. Asymmetric so rotations/translations are distinguishable. */
static void make_struct_vol(float *v, int nz, int ny, int nx) {
    float blobs[6][4] = { /* z,y,x,sigma (fractions of dim) */
        {0.30f,0.30f,0.35f,0.10f}, {0.65f,0.40f,0.55f,0.08f},
        {0.45f,0.70f,0.30f,0.12f}, {0.55f,0.55f,0.72f,0.07f},
        {0.25f,0.60f,0.62f,0.09f}, {0.72f,0.28f,0.40f,0.06f},
    };
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        double fz = (double)z/nz, fy = (double)y/ny, fx = (double)x/nx;
        double val = 0.15 + 0.10*sin(6.0*fx)*cos(5.0*fy + 0.5)*sin(4.0*fz);
        for (int b = 0; b < 6; b++) {
            double dz=fz-blobs[b][0], dy=fy-blobs[b][1], dx=fx-blobs[b][2];
            double s=blobs[b][3];
            val += 0.8*exp(-(dz*dz+dy*dy+dx*dx)/(2*s*s));
        }
        if (val > 1.0) val = 1.0;
        v[(z*ny+y)*nx+x] = (float)val;
    }
}

























/* ============ LAYER 2: deformable / Demons =============================== */

/* a textured volume: blobs + several octaves of sinusoid so there is gradient
 * EVERYWHERE (Demons needs texture; flat regions suffer the aperture problem). */
static void make_textured_vol(float *v, int nz, int ny, int nx) {
    make_struct_vol(v, nz, ny, nx);
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        double fz=(double)z/nz, fy=(double)y/ny, fx=(double)x/nx;
        double t = 0.06*sin(18.0*fx)*sin(15.0*fy)
                 + 0.05*cos(13.0*fz+1.0)*sin(11.0*fx)
                 + 0.04*sin(21.0*fy+0.3)*cos(17.0*fz);
        double val = v[(z*ny+y)*nx+x] + t;
        if (val < 0) val = 0; if (val > 1) val = 1;
        v[(z*ny+y)*nx+x] = (float)val;
    }
}

/* build a known SMOOTH (low-frequency sinusoidal) displacement field, pull
 * convention, amplitude `amp` voxels. This is the field that GENERATES moving
 * from fixed: moving(x) = fixed(x + u(x)) ... but we want the registration to
 * recover the field that pulls moving back to fixed. We instead define moving =
 * warp_field(fixed, +u) and let demons recover ~ +u (so warp_field(moving,u_rec)
 * reproduces fixed). */
static void make_true_field(float *ux, float *uy, float *uz, int nz,int ny,int nx, double amp) {
    for (int z=0; z<nz; z++) for (int y=0; y<ny; y++) for (int x=0; x<nx; x++) {
        double fz=(double)z/nz, fy=(double)y/ny, fx=(double)x/nx;
        size_t i=(z*ny+y)*nx+x;
        ux[i]=(float)(amp*sin(2.0*M_PI*fy)*cos(2.0*M_PI*fz));
        uy[i]=(float)(amp*sin(2.0*M_PI*fx)*cos(1.0*M_PI*fz));
        uz[i]=(float)(amp*0.7*sin(2.0*M_PI*fx)*sin(1.5*M_PI*fy));
    }
}

static double pearson(const float *a, const float *b, size_t n) {
    double sa=0,sb=0,saa=0,sbb=0,sab=0;
    for(size_t i=0;i<n;i++){sa+=a[i];sb+=b[i];saa+=(double)a[i]*a[i];sbb+=(double)b[i]*b[i];sab+=(double)a[i]*b[i];}
    double ma=sa/n,mb=sb/n,cov=sab/n-ma*mb,va=saa/n-ma*ma,vb=sbb/n-mb*mb;
    double d=sqrt(va*vb); return d<1e-12?0.0:cov/d;
}

/* mean |gradient| magnitude of a field component (smoothness proxy) */
static double field_grad_mag(const float *u, int nz,int ny,int nx) {
    double s=0; long c=0;
    for(int z=1;z<nz-1;z++)for(int y=1;y<ny-1;y++)for(int x=1;x<nx-1;x++){
        size_t i=(z*ny+y)*nx+x;
        double gx=0.5*(u[i+1]-u[i-1]);
        double gy=0.5*(u[i+nx]-u[i-nx]);
        double gz=0.5*(u[i+nx*ny]-u[i-nx*ny]);
        s+=sqrt(gx*gx+gy*gy+gz*gz); c++;
    }
    return c? s/c : 0;
}









/* ============ LANDMARK affine fit + multi-resolution fusion ============== */

static double det33(const double *M) { /* linear part of a 3x4, leading 3 cols */
    return M[0]*(M[5]*M[10]-M[6]*M[9]) - M[1]*(M[4]*M[10]-M[6]*M[8])
         + M[2]*(M[4]*M[9]-M[5]*M[8]);
}









/* ---- phase correlation: recover a KNOWN sub-voxel shift -------------------- */
/* Build `moving` as `fixed` circularly shifted by a SUB-VOXEL (dz,dy,dx) using the
 * Fourier shift theorem (multiply spectrum by exp(-i 2pi (k.s/N))) -- this is an
 * EXACT band-limited sub-voxel translation, the proper ground truth for a sub-
 * pixel phase-correlation test. `fixed` is broadband (random) texture, like real
 * CT papyrus; phase correlation is a broadband estimator (a low-frequency-only
 * field gives a broad, bias-prone peak -- not representative). */
static void fourier_shift(const float *in, float *out, int nz,int ny,int nx,
                          double sz,double sy,double sx) {
    size_t N=(size_t)nz*ny*nx;
    float *re=malloc(4*N),*im=calloc(N,4);
    for (size_t i=0;i<N;i++) re[i]=in[i];
    fy_fft3d(re,im,nz,ny,nx,-1);
    for (int z=0;z<nz;z++) for (int y=0;y<ny;y++) for (int x=0;x<nx;x++) {
        int kz=(z<=nz/2)?z:z-nz, ky=(y<=ny/2)?y:y-ny, kx=(x<=nx/2)?x:x-nx;
        double ph = -2.0*M_PI*((double)kz*sz/nz + (double)ky*sy/ny + (double)kx*sx/nx);
        double c=cos(ph), s=sin(ph);
        size_t i=((size_t)z*ny+y)*nx+x;
        float r=re[i], m=im[i];
        re[i]=(float)(r*c - m*s); im[i]=(float)(r*s + m*c);
    }
    fy_fft3d(re,im,nz,ny,nx,+1);
    fy_fft3d_normalize(re,im,nz,ny,nx);
    for (size_t i=0;i<N;i++) out[i]=re[i];
    free(re);free(im);
}


/* ---- mutual information: peaks at correct alignment ----------------------- */


/* Coherence-enhancing diffusion: two parallel bright sheets separated by a thin
 * dark gap, plus noise. CED must (a) drop the noise on the sheets AND (b) keep the
 * dark gap dark (NOT merge the layers). A plain isotropic blur would fill the gap. */





/* ---- IN-PLANE TEXTURE ENHANCER ----
 * Volume: a thick bright SHEET (slab perpendicular to z, normal = +z) that carries a
 * fine IN-PLANE ripple (high-freq sinusoid along x, within the sheet plane) = the
 * "crackle" texture. PLUS a cross-sheet intensity STEP (background jumps across z) =
 * the layering we must NOT amplify. PLUS noise everywhere. We assert:
 *   (a) in-plane texture-band energy INCREASES after the filter;
 *   (b) the cross-sheet step is NOT amplified (~unchanged/suppressed -- a plain unsharp
 *       mask would boost it, the steered one must not);
 *   (c) noise-only regions are NOT blown up: texture/noise ratio IMPROVES (gating). */




/* build a 3x4 affine (output->input map) for rotation about center + iso scale
 * + translation. angles in radians (about z,y,x), s = iso scale, t = (z,y,x). */
static void build_M(double *M, double rz,double ry,double rx,double s,
                    double tz,double ty,double tx, double cz,double cy,double cx) {
    double Cz=cos(rz),Sz=sin(rz),Cy=cos(ry),Sy=sin(ry),Cx=cos(rx),Sx=sin(rx);
    double Rz[9]={Cz,-Sz,0, Sz,Cz,0, 0,0,1};
    double Ry[9]={Cy,0,Sy, 0,1,0, -Sy,0,Cy};
    double Rx[9]={1,0,0, 0,Cx,-Sx, 0,Sx,Cx};
    double Rzy[9],R[9];
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){double a=0;for(int k=0;k<3;k++)a+=Rz[i*3+k]*Ry[k*3+j];Rzy[i*3+j]=a;}
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){double a=0;for(int k=0;k<3;k++)a+=Rzy[i*3+k]*Rx[k*3+j];R[i*3+j]=a*s;}
    double cc[3]={cz,cy,cx},t[3]={tz,ty,tx};
    for(int i=0;i<3;i++){
        M[i*4+0]=R[i*3+0]; M[i*4+1]=R[i*3+1]; M[i*4+2]=R[i*3+2];
        double Rc=R[i*3+0]*cc[0]+R[i*3+1]*cc[1]+R[i*3+2]*cc[2];
        M[i*4+3]=cc[i]-Rc+t[i];
    }
}

/* invert a 3x4 affine (treats it as 4x4 with [0 0 0 1]). out maps the other way. */
static int invert_M(const double *M, double *inv) {
    double a=M[0],b=M[1],c=M[2], d=M[4],e=M[5],f=M[6], g=M[8],h=M[9],i=M[10];
    double det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
    if (fabs(det)<1e-12) return 1;
    double id=1.0/det;
    double L[9];
    L[0]=(e*i-f*h)*id; L[1]=(c*h-b*i)*id; L[2]=(b*f-c*e)*id;
    L[3]=(f*g-d*i)*id; L[4]=(a*i-c*g)*id; L[5]=(c*d-a*f)*id;
    L[6]=(d*h-e*g)*id; L[7]=(b*g-a*h)*id; L[8]=(a*e-b*d)*id;
    double t[3]={M[3],M[7],M[11]};
    for(int r=0;r<3;r++){
        inv[r*4+0]=L[r*3+0]; inv[r*4+1]=L[r*3+1]; inv[r*4+2]=L[r*3+2];
        inv[r*4+3]=-(L[r*3+0]*t[0]+L[r*3+1]*t[1]+L[r*3+2]*t[2]);
    }
    return 0;
}

/* mean abs diff over the interior (avoid the zero-padded border) of two vols */
static double interior_mad(const float *a, const float *b, int nz,int ny,int nx, int m) {
    double s=0; long c=0;
    for(int z=m;z<nz-m;z++)for(int y=m;y<ny-m;y++)for(int x=m;x<nx-m;x++){
        s+=fabs((double)a[(z*ny+y)*nx+x]-b[(z*ny+y)*nx+x]); c++;
    }
    return c? s/c : 0;
}

static void test_warp_identity(void) {
    int nz=24,ny=24,nx=24,n=nz*ny*nx;
    float *in=malloc(4*n),*out=malloc(4*n);
    make_struct_vol(in,nz,ny,nx);
    double I[12]={1,0,0,0, 0,1,0,0, 0,0,1,0};
    fy_warp_affine(in,out,nz,ny,nx,I,nz,ny,nx);
    double mad=interior_mad(in,out,nz,ny,nx,0);
    CHECK(mad<1e-6, "warp_affine identity reproduces input");
    free(in);free(out);
}

static void test_warp_translation(void) {
    int nz=24,ny=24,nx=24,n=nz*ny*nx;
    float *in=malloc(4*n),*out=malloc(4*n);
    make_struct_vol(in,nz,ny,nx);
    /* output->input shift of +3 in x: out[x] = in[x+3], i.e. content moves -3 */
    double M[12]={1,0,0,0, 0,1,0,0, 0,0,1,3};
    fy_warp_affine(in,out,nz,ny,nx,M,nz,ny,nx);
    /* check out at (z,y,x) equals in at (z,y,x+3) for interior */
    double maxe=0;
    for(int z=5;z<nz-5;z++)for(int y=5;y<ny-5;y++)for(int x=5;x<nx-8;x++){
        double e=fabs((double)out[(z*ny+y)*nx+x]-in[(z*ny+y)*nx+(x+3)]);
        if(e>maxe)maxe=e;
    }
    CHECK(maxe<1e-5, "warp_affine integer translation shifts exactly");
    free(in);free(out);
}

static void test_warp_roundtrip(void) {
    int nz=28,ny=28,nx=28,n=nz*ny*nx;
    float *in=malloc(4*n),*mid=malloc(4*n),*back=malloc(4*n);
    make_struct_vol(in,nz,ny,nx);
    double cz=(nz-1)*0.5,cy=(ny-1)*0.5,cx=(nx-1)*0.5;
    double M[12],Minv[12];
    build_M(M, 0.10,0.05,0.07, 1.05, 1.5,-2.0,1.0, cz,cy,cx);
    invert_M(M,Minv);
    fy_warp_affine(in,mid,nz,ny,nx,M,nz,ny,nx);
    fy_warp_affine(mid,back,nz,ny,nx,Minv,nz,ny,nx);
    double mad=interior_mad(in,back,nz,ny,nx,5);
    CHECK(mad<0.03, "warp_affine round-trip (M then M^-1) recovers input");
    free(in);free(mid);free(back);
}

static void test_warp_field_identity(void) {
    int nz=24,ny=24,nx=24; size_t n=(size_t)nz*ny*nx;
    float *in=malloc(4*n),*out=malloc(4*n);
    float *zx=calloc(n,4),*zy=calloc(n,4),*zz=calloc(n,4);
    make_textured_vol(in,nz,ny,nx);
    fy_warp_field(in,out,nz,ny,nx,zx,zy,zz);
    double mad=interior_mad(in,out,nz,ny,nx,0);
    CHECK(mad<1e-6, "warp_field with zero field == identity");
    free(in);free(out);free(zx);free(zy);free(zz);
}

static void test_ncc_self(void) {
    int nz=24,ny=24,nx=24,n=nz*ny*nx;
    float *a=malloc(4*n),*b=malloc(4*n);
    make_struct_vol(a,nz,ny,nx);
    double I[12]={1,0,0,0, 0,1,0,0, 0,0,1,0};
    double self=fy_ncc_warped(a,a,nz,ny,nx,I);
    CHECK(self>0.999, "NCC of a volume with itself == 1");
    /* affine intensity change b = 0.4*a + 0.2 -> NCC must STILL be ~1 */
    for(int i=0;i<n;i++) b[i]=0.4f*a[i]+0.2f;
    double inv=fy_ncc_warped(a,b,nz,ny,nx,I);
    CHECK(inv>0.999, "NCC invariant to affine intensity change (a*I+b)");
    free(a);free(b);
}

/* core helper: recover a known transform. Returns recovery errors via pointers.
 * We apply Mknown (the transform that GENERATED moving from fixed) and check the
 * recovered M_out matches Mknown's INVERSE warp -- equivalently, that warping
 * moving by M_out reproduces fixed. We measure residual NCC + image error. */
static void run_recovery(int rigid, double angle, double scale, double tx,
                         double gamma_contrast,
                         double *ncc_out, double *mad_out) {
    int nz=64,ny=64,nx=64,n=nz*ny*nx;
    float *fixed=malloc(4*n),*moving=malloc(4*n),*warped=malloc(4*n);
    make_struct_vol(fixed,nz,ny,nx);
    double cz=(nz-1)*0.5,cy=(ny-1)*0.5,cx=(nx-1)*0.5;
    /* Mgen maps moving-grid -> fixed-grid; we generate moving by sampling fixed
     * at Mgen^-1... simplest: define moving(x) = fixed(Mgen . x) via warp with
     * Mgen as the output->input map. Then fixed = warp(moving, Mgen^-1). The
     * registration should recover M_out ~ Mgen^-1. */
    double Mgen[12], Mtrue[12];
    build_M(Mgen, angle, angle*0.6, angle*0.8, scale, tx, -tx*0.7, tx*0.5, cz,cy,cx);
    fy_warp_affine(fixed,moving,nz,ny,nx,Mgen,nz,ny,nx);
    invert_M(Mgen,Mtrue);   /* the transform registration should find */

    /* optional contrast change on moving to test multimodal robustness */
    if (gamma_contrast > 0) {
        for(int i=0;i<n;i++){ float v=moving[i]; if(v<0)v=0; moving[i]=powf(v,(float)gamma_contrast)*0.7f+0.05f; }
    }

    double M_out[12]={1,0,0,0, 0,1,0,0, 0,0,1,0};   /* identity init, same grid */
    fy_register_affine(fixed,moving,nz,ny,nx,M_out,rigid);

    fy_warp_affine(moving,warped,nz,ny,nx,M_out,nz,ny,nx);
    *ncc_out = fy_ncc_warped(fixed,moving,nz,ny,nx,M_out);
    *mad_out = interior_mad(fixed,warped,nz,ny,nx,6);
    (void)Mtrue;
    free(fixed);free(moving);free(warped);
}

static void test_register_rigid(void) {
    double ncc,mad;
    run_recovery(1, 0.08, 1.0, 2.5, 0.0, &ncc,&mad);
    printf("     [rigid] recovered NCC=%.4f  interior MAD=%.4f\n", ncc, mad);
    CHECK(ncc>0.98, "register rigid recovers transform (NCC>0.98)");
    CHECK(mad<0.05, "register rigid: warped moving matches fixed (MAD<0.05)");
}

static void test_register_affine(void) {
    double ncc,mad;
    run_recovery(0, 0.08, 1.06, 2.5, 0.0, &ncc,&mad);
    printf("     [affine] recovered NCC=%.4f  interior MAD=%.4f\n", ncc, mad);
    CHECK(ncc>0.98, "register affine recovers transform (NCC>0.98)");
    CHECK(mad<0.05, "register affine: warped moving matches fixed (MAD<0.05)");
}

static void test_register_contrast(void) {
    double ncc,mad;
    /* moving has gamma=1.8 + rescale: very different contrast from fixed */
    run_recovery(1, 0.07, 1.0, 2.0, 1.8, &ncc,&mad);
    printf("     [contrast/multimodal] recovered NCC=%.4f  interior MAD(vs orig contrast)=%.4f\n", ncc, mad);
    /* MAD here compares warped (gamma'd) moving to original fixed -> intensities
     * differ by design; geometry is what we test, via NCC. */
    CHECK(ncc>0.97, "register robust to contrast change (NCC>0.97, multimodal-ready)");
}

static void test_demons_known_warp(void) {
    int nz=64,ny=64,nx=64; size_t n=(size_t)nz*ny*nx;
    float *fixed=malloc(4*n),*moving=malloc(4*n),*warped=malloc(4*n);
    float *tx=malloc(4*n),*ty=malloc(4*n),*tz=malloc(4*n);     /* true field */
    float *ux=calloc(n,4),*uy=calloc(n,4),*uz=calloc(n,4);     /* recovered  */
    make_textured_vol(fixed,nz,ny,nx);
    make_true_field(tx,ty,tz,nz,ny,nx, 4.0);                   /* ~4 vox max */
    fy_warp_field(fixed,moving,nz,ny,nx,tx,ty,tz);             /* moving=fixed(x+u) */

    double mad_before=interior_mad(fixed,moving,nz,ny,nx,6);
    double I[12]={1,0,0,0,0,1,0,0,0,0,1,0};
    double ncc_before=fy_ncc_warped(fixed,moving,nz,ny,nx,I);

    fy_register_demons(fixed,moving,nz,ny,nx,ux,uy,uz, 80, 1.5, 1.5);
    fy_warp_field(moving,warped,nz,ny,nx,ux,uy,uz);

    double mad_after=interior_mad(fixed,warped,nz,ny,nx,6);
    double ncc_after=pearson(fixed,warped,n);
    /* field recovery: the recovered PULL field that maps moving->fixed is the
     * INVERSE of the field that generated moving (moving=fixed(x+u_true)), so
     * u_rec ~ -u_true. We test the magnitude of the correlation. */
    double fc = pearson(ux,tx,n);   /* x-component; expect strongly NEGATIVE */
    printf("     [demons known-warp] NCC %.4f->%.4f  MAD %.4f->%.4f  field-corr(x)=%.3f (expect <0, |.| high)\n",
           ncc_before,ncc_after,mad_before,mad_after,fc);
    CHECK(ncc_after>ncc_before+0.01, "demons raises NCC vs unregistered");
    CHECK(ncc_after>0.97, "demons reaches high NCC on textured known warp");
    CHECK(mad_after<0.6*mad_before, "demons cuts intensity MAD substantially");
    CHECK(fabs(fc)>0.5, "recovered field correlates with true field (x-comp, inverse sign)");
    free(fixed);free(moving);free(warped);free(tx);free(ty);free(tz);
    free(ux);free(uy);free(uz);
}

static void test_demons_regularization(void) {
    int nz=48,ny=48,nx=48; size_t n=(size_t)nz*ny*nx;
    float *fixed=malloc(4*n),*moving=malloc(4*n);
    float *tx=malloc(4*n),*ty=malloc(4*n),*tz=malloc(4*n);
    make_textured_vol(fixed,nz,ny,nx);
    make_true_field(tx,ty,tz,nz,ny,nx, 3.0);
    fy_warp_field(fixed,moving,nz,ny,nx,tx,ty,tz);

    float *ax=calloc(n,4),*ay=calloc(n,4),*az=calloc(n,4);
    float *bx=calloc(n,4),*by=calloc(n,4),*bz=calloc(n,4);
    fy_register_demons(fixed,moving,nz,ny,nx,ax,ay,az, 60, 0.7, 1.5);  /* loose */
    fy_register_demons(fixed,moving,nz,ny,nx,bx,by,bz, 60, 2.5, 1.5);  /* stiff */
    double g_loose=field_grad_mag(ax,nz,ny,nx);
    double g_stiff=field_grad_mag(bx,nz,ny,nx);
    printf("     [demons regularization] grad(sigma=0.7)=%.4f  grad(sigma=2.5)=%.4f\n",g_loose,g_stiff);
    CHECK(g_stiff<g_loose, "larger field_sigma -> smoother (lower-gradient) field");
    CHECK(g_stiff<0.5, "stiff field stays smooth/bounded (no tearing)");
    free(fixed);free(moving);free(tx);free(ty);free(tz);
    free(ax);free(ay);free(az);free(bx);free(by);free(bz);
}

static void test_register_full_affine_plus_demons(void) {
    int nz=64,ny=64,nx=64; size_t n=(size_t)nz*ny*nx;
    float *fixed=malloc(4*n),*moving=malloc(4*n);
    float *aff_only=malloc(4*n),*full=malloc(4*n);
    float *tx=malloc(4*n),*ty=malloc(4*n),*tz=malloc(4*n);
    float *ux=calloc(n,4),*uy=calloc(n,4),*uz=calloc(n,4);
    make_textured_vol(fixed,nz,ny,nx);
    /* moving = affine(fixed) THEN smooth deformation: build a combined moving */
    double cz=(nz-1)*0.5,cy=(ny-1)*0.5,cx=(nx-1)*0.5;
    double Mgen[12];
    build_M(Mgen, 0.06,0.04,0.05, 1.03, 2.0,-1.5,1.0, cz,cy,cx);
    float *affd=malloc(4*n);
    fy_warp_affine(fixed,affd,nz,ny,nx,Mgen,nz,ny,nx);    /* affine part */
    make_true_field(tx,ty,tz,nz,ny,nx, 3.0);
    fy_warp_field(affd,moving,nz,ny,nx,tx,ty,tz);          /* + deformation */

    /* affine alone */
    double M_aff[12]={1,0,0,0,0,1,0,0,0,0,1,0};
    fy_register_affine(fixed,moving,nz,ny,nx,M_aff,0);
    fy_warp_affine(moving,aff_only,nz,ny,nx,M_aff,nz,ny,nx);
    double ncc_aff=pearson(fixed,aff_only,n);

    /* affine + demons via fy_register_full */
    double M_full[12]={1,0,0,0,0,1,0,0,0,0,1,0};
    fy_register_full(fixed,moving,nz,ny,nx,M_full,0, ux,uy,uz, 80,1.5,1.5);
    fy_warp_affine(moving,affd,nz,ny,nx,M_full,nz,ny,nx);
    fy_warp_field(affd,full,nz,ny,nx,ux,uy,uz);
    double ncc_full=pearson(fixed,full,n);

    printf("     [affine vs affine+demons] NCC affine=%.4f  affine+demons=%.4f\n",ncc_aff,ncc_full);
    CHECK(ncc_full>ncc_aff+0.005, "affine+demons beats affine alone on affine+deformable warp");
    free(fixed);free(moving);free(aff_only);free(full);free(affd);
    free(tx);free(ty);free(tz);free(ux);free(uy);free(uz);
}

static void test_phase_correlate_subvoxel(void) {
    int nz=32,ny=32,nx=32,n=nz*ny*nx;   /* pow2: exact circular shift */
    float *f=malloc(4*n),*m=malloc(4*n);
    /* broadband random texture, mildly smoothed (band-limited) */
    unsigned int seed=12345;
    for (int i=0;i<n;i++){ seed=seed*1103515245u+12345u; f[i]=(float)((seed>>9)&8191)/8191.0f; }
    float *tmp=malloc(4*n);
    for (int z=0;z<nz;z++) for (int y=0;y<ny;y++) for (int x=0;x<nx;x++){
        int xm=(x-1+nx)%nx, xp=(x+1)%nx;
        tmp[(z*ny+y)*nx+x]=(f[(z*ny+y)*nx+xm]+2*f[(z*ny+y)*nx+x]+f[(z*ny+y)*nx+xp])/4;
    }
    memcpy(f,tmp,4*n);
    struct { double z,y,x; } cases[3] = { {0.5,-0.25,0.0}, {0.33,0.66,-0.4}, {-0.7,0.2,0.85} };
    double worst=0;
    for (int i=0;i<3;i++) {
        fourier_shift(f,m,nz,ny,nx,cases[i].z,cases[i].y,cases[i].x);
        double s[3], pk;
        int rc = fy_phase_correlate(f,m,nz,ny,nx,s,&pk,0);  /* circular -> no window */
        double e = fabs(s[0]-cases[i].z)+fabs(s[1]-cases[i].y)+fabs(s[2]-cases[i].x);
        printf("     [pc] true(% .2f % .2f % .2f) got(% .3f % .3f % .3f) L1err=%.3f peak=%.3f rc=%d\n",
               cases[i].z,cases[i].y,cases[i].x, s[0],s[1],s[2], e, pk, rc);
        if (e>worst) worst=e;
    }
    CHECK(worst < 0.15, "phase_correlate recovers sub-voxel shift (<0.05 vox/axis avg)");
    /* contrast invariance: scale+offset moving, shift must still be recovered */
    fourier_shift(f,m,nz,ny,nx,0.4,-0.6,0.5);
    for (int i=0;i<n;i++) m[i] = 0.3f*m[i] + 0.4f;  /* affine intensity change */
    double s[3],pk; fy_phase_correlate(f,m,nz,ny,nx,s,&pk,0);
    double e = fabs(s[0]-0.4)+fabs(s[1]+0.6)+fabs(s[2]-0.5);
    printf("     [pc] contrast-changed moving: got(% .3f % .3f % .3f) L1err=%.3f\n", s[0],s[1],s[2], e);
    CHECK(e < 0.15, "phase_correlate is robust to brightness/contrast change");
    free(f);free(m);free(tmp);
}

/* ---- mutual information: peaks at correct alignment ----------------------- */
static void test_mutual_information_peaks(void) {
    int nz=32,ny=32,nx=32,n=nz*ny*nx;
    float *fixed=malloc(4*n), *moving=malloc(4*n);
    make_textured_vol(fixed,nz,ny,nx);
    /* moving = a NONLINEAR remap of fixed (sqrt) -> NCC's linear assumption is
     * violated but MI should not care; build it as a copy then remap below. */
    for (int i=0;i<n;i++) moving[i] = sqrtf(fixed[i] > 0 ? fixed[i] : 0);
    double I[12]={1,0,0,0, 0,1,0,0, 0,0,1,0};
    double mi0 = fy_mutual_information(fixed,moving,nz,ny,nx,I,32);
    /* shifted by 3 voxels in x -> misaligned -> MI should drop */
    double S[12]={1,0,0,0, 0,1,0,0, 0,0,1,3.0};
    double mis = fy_mutual_information(fixed,moving,nz,ny,nx,S,32);
    printf("     [mi] aligned MI=%.4f  shifted(3vox) MI=%.4f nats\n", mi0, mis);
    CHECK(mi0 > mis + 0.05, "MI is higher at correct alignment than when shifted");
    /* and MI of identical images > MI of independent noise */
    float *noise=malloc(4*n);
    unsigned int seed=99;
    for (int i=0;i<n;i++){ seed=seed*1103515245u+12345u; noise[i]=(float)((seed>>9)&1023)/1023.0f; }
    double mi_noise = fy_mutual_information(fixed,noise,nz,ny,nx,I,32);
    printf("     [mi] self(nonlin-remap) MI=%.4f  vs independent-noise MI=%.4f\n", mi0, mi_noise);
    CHECK(mi0 > mi_noise, "MI(dependent) > MI(independent noise)");
    free(fixed);free(moving);free(noise);
}

int main(void) {
    test_fft_vs_dft();
    test_fft_vs_dft();
    test_fft_roundtrip();
    test_fft_radix3();
    test_guided_fast();
    test_dering();
    test_downscale_cbox();
    test_noise_aniso();
    test_matched_aniso();
    test_deconv_rfft_equivalence();
    test_warp_identity();
    test_warp_translation();
    test_warp_roundtrip();
    test_warp_field_identity();
    test_ncc_self();
    test_register_rigid();
    test_register_affine();
    test_register_contrast();
    test_demons_known_warp();
    test_demons_regularization();
    test_register_full_affine_plus_demons();
    test_phase_correlate_subvoxel();
    test_mutual_information_peaks();
    test_paganin_transfer();
    test_deconvolve_sharpens();
    test_halo_reasonable();
    test_dewindow();
    test_estimate_noise();
    test_deltabeta_scale();
    printf("\n%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
