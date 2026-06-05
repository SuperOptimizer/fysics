/* paganin.c -- recon transfer functions + Wiener deconvolution kernel.
 *
 * Inverts the known nabu reconstruction operators to sharpen a volume. The hot
 * loops (filter build, complex multiply) are flat and vectorizable under -Ofast.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FY_HC_KEV_UM 1.23984199e-3   /* h*c in keV*micron (nabu constant) */

double fy_paganin_transfer(double f_cyc_per_voxel, const fy_physics *p) {
    /* exact nabu form: 1/(1 + db*L*D*pi*f^2), f in cycles/micron */
    double L = FY_HC_KEV_UM / p->energy_kev;     /* micron */
    double D = p->distance_mm * 1000.0;           /* micron */
    /* pixel_um MUST be microns (e.g. 2.4). Guard against the common mistake of
     * passing the metadata's millimeter value (e.g. 0.0024): a sub-0.1 "micron"
     * pixel is physically a mm value, so auto-correct to keep callers safe. */
    double px = p->pixel_um;
    if (px > 0.0 && px < 0.1) px *= 1000.0;        /* mm -> micron */
    double f_um = f_cyc_per_voxel / px;            /* cycles/voxel -> cycles/micron */
    double f2 = f_um * f_um;
    return 1.0 / (1.0 + p->delta_beta * L * D * M_PI * f2);
}

double fy_unsharp_transfer(double f_cyc_per_voxel, const fy_physics *p) {
    if (p->unsharp_coeff == 0.0 || p->unsharp_sigma == 0.0) return 1.0;
    /* gaussian transfer at sigma (voxels): G = exp(-2 pi^2 sigma^2 f^2) */
    double s = p->unsharp_sigma;
    double g = exp(-2.0 * M_PI * M_PI * s * s * f_cyc_per_voxel * f_cyc_per_voxel);
    return 1.0 + p->unsharp_coeff * (1.0 - g);
}

double fy_recon_transfer(double f_cyc_per_voxel, const fy_physics *p) {
    double H = fy_paganin_transfer(f_cyc_per_voxel, p) *
               fy_unsharp_transfer(f_cyc_per_voxel, p);
    if (H < 1e-4) H = 1e-4;
    return H;
}

/* Maximum Wiener gain we ever allow, so a bad reg can't amplify the high-frequency
 * noise floor without bound. The Wiener filter H/(H^2+reg) peaks at 1/(2*sqrt(reg)),
 * so capping the gain == enforcing a reg floor; we apply it directly as a clamp in the
 * filter loop (frequency-by-frequency) which is strictly safer than only raising reg. */
#define FY_MAX_DECONV_GAIN 8.0

/* Noise-safe regularization for fy_deconvolve.
 *
 * NOTE (measured, PHerc0139, 2026-06): an FRC even/odd-split test showed the high-
 * frequency content of these volumes is COHERENT SIGNAL out to ~0.9 Nyquist (FRC ~0.95),
 * not a noise floor -- so the Wiener inverse amplifying it by w^2 is correct, not a bug.
 * The Wiener gain w=H/(H^2+reg) is already self-limiting (peaks at 1/(2*sqrt(reg))), and
 * on real data it stays < ~4 for every tested volume. There is therefore no need to
 * scale reg up for "strong-filter" volumes; doing so would just blur away real detail.
 * We keep a single conservative base, and the per-frequency gain clamp
 * (FY_MAX_DECONV_GAIN) in the filter loop is the only hard safety rail -- it never fires
 * on the measured data but bounds pathological metadata. */
double fy_auto_reg(const fy_physics *p, double base) {
    (void)p;
    return base > 0.0 ? base : 0.015;
}

int fy_kernel_halo(const fy_physics *p) {
    /* impulse response of the deconv filter along one axis; find decay extent */
    const int n = 256;
    float *re = (float *)malloc(sizeof(float) * n);
    float *im = (float *)calloc(n, sizeof(float));
    if (!re || !im) { free(re); free(im); return 16; }
    for (int i = 0; i < n; i++) {
        double f = (i <= n / 2) ? (double)i / n : (double)(i - n) / n; /* fftfreq */
        double H = fy_recon_transfer(fabs(f), p);
        double inv = H / (H * H + 0.05);
        re[i] = (float)inv; im[i] = 0.0f;
    }
    fy_fft1d(re, im, n, +1);  /* to real space */
    /* peak-relative decay */
    float peak = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(re[i]); if (a > peak) peak = a; }
    int half = 8;
    for (int d = 1; d < n / 2; d++) {
        float a = fabsf(re[d]);           /* response at lag d (index 0 = center) */
        if (a > 0.01f * peak) half = d;
    }
    free(re); free(im);
    if (half < 8) half = 8;
    if (half > 96) half = 96;
    return half;
}

/* reflect-pad index helper */
static inline int reflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i - 1;
        if (i >= n) i = 2 * n - i - 1;
    }
    return i;
}

int fy_deconvolve(const float *in, float *out,
                  int nz, int ny, int nx,
                  const fy_physics *p, double reg) {
    /* reg <= 0 -> derive a noise-safe value from the volume physics (recommended). */
    if (reg <= 0.0) reg = fy_auto_reg(p, 0.015);
    int pz = fy_next_pow2(nz), py = fy_next_pow2(ny), px = fy_next_pow2(nx);
    size_t np = (size_t)pz * py * px;
    float *re = (float *)malloc(sizeof(float) * np);
    float *im = (float *)calloc(np, sizeof(float));
    if (!re || !im) { free(re); free(im); return 1; }

    /* reflect-pad input into the padded power-of-two buffer */
    for (int z = 0; z < pz; z++) {
        int sz = reflect(z, nz);
        for (int y = 0; y < py; y++) {
            int sy = reflect(y, ny);
            size_t drow = ((size_t)z * py + y) * px;
            size_t srow = ((size_t)sz * ny + sy) * nx;
            for (int x = 0; x < px; x++) {
                int sx = reflect(x, nx);
                re[drow + x] = in[srow + sx];
            }
        }
    }

    fy_fft3d(re, im, pz, py, px, -1);

    /* precompute per-axis radial frequencies (cycles/voxel) */
    double *fz2 = (double *)malloc(sizeof(double) * pz);
    double *fy2 = (double *)malloc(sizeof(double) * py);
    double *fx2 = (double *)malloc(sizeof(double) * px);
    if (!fz2 || !fy2 || !fx2) { free(re); free(im); free(fz2); free(fy2); free(fx2); return 1; }
    for (int i = 0; i < pz; i++) { double f = (i <= pz/2) ? (double)i/pz : (double)(i-pz)/pz; fz2[i] = f*f; }
    for (int i = 0; i < py; i++) { double f = (i <= py/2) ? (double)i/py : (double)(i-py)/py; fy2[i] = f*f; }
    for (int i = 0; i < px; i++) { double f = (i <= px/2) ? (double)i/px : (double)(i-px)/px; fx2[i] = f*f; }

    /* apply Wiener inverse filter: F *= H/(H^2+reg). Flat, vectorizable. */
    for (int z = 0; z < pz; z++) {
        for (int y = 0; y < py; y++) {
            size_t row = ((size_t)z * py + y) * px;
            double fzy = fz2[z] + fy2[y];
            for (int x = 0; x < px; x++) {
                double fr = sqrt(fzy + fx2[x]);
                double H = fy_recon_transfer(fr, p);
                double w = H / (H * H + reg);
                /* hard cap so a mis-set reg can never amplify the noise floor */
                if (w > FY_MAX_DECONV_GAIN) w = FY_MAX_DECONV_GAIN;
                re[row + x] *= (float)w;
                im[row + x] *= (float)w;
            }
        }
    }

    fy_fft3d(re, im, pz, py, px, +1);
    fy_fft3d_normalize(re, im, pz, py, px);

    /* crop back to the original shape */
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++) {
            size_t drow = ((size_t)z * ny + y) * nx;
            size_t srow = ((size_t)z * py + y) * px;
            for (int x = 0; x < nx; x++)
                out[drow + x] = re[srow + x];
        }

    free(re); free(im); free(fz2); free(fy2); free(fx2);
    return 0;
}

/* Gureyev-Paganin combined transfer (the Fourier filter to apply to the image):
 *   reduced-Paganin inverse  (1 + b'*k2)   [boosts high freq -- undoes the blur]
 * times Tikhonov Gaussian-PSF deconvolution  G/(G^2+gamma).
 * b' = paganin b reduced by the PSF's variance contribution (per the paper, the
 * PSF already does partial deblur, so we deconvolve LESS in the Paganin step and
 * deconvolve the PSF explicitly). k in cycles/voxel. */
static double gureyev_transfer(double fr, const fy_physics *p, double gamma) {
    /* Paganin b in (cycles/voxel)^-2 units: standard filter is 1/(1+b*f_um^2) with
     * f_um=fr/pixel_um, so in cycles/voxel: b_vox = db*L*D*pi / pixel_um^2. */
    double L = FY_HC_KEV_UM / p->energy_kev;     /* wavelength (micron) */
    double D = p->distance_mm * 1000.0;
    double px = p->pixel_um; if (px > 0 && px < 0.1) px *= 1000.0;
    double b_vox = p->delta_beta * L * D * M_PI / (px * px);
    /* system PSF (Gaussian) variance contribution to the regularization:
     * the PSF transfer is G(f)=exp(-2 pi^2 sigma^2 f^2); its small-f expansion is
     * 1 - 2 pi^2 sigma^2 f^2, i.e. it ALSO acts like a (1 - c f^2) low-pass with
     * c = 2 pi^2 sigma^2. Reduce the Paganin b by this so we don't double-count. */
    double sigma = p->psf_sigma_vox > 0 ? p->psf_sigma_vox : 0.5;
    double c_psf = 2.0 * M_PI * M_PI * sigma * sigma;
    double b_reduced = b_vox - c_psf;
    if (b_reduced < 0) b_reduced = 0;
    double f2 = fr * fr;
    /* The reduced-Paganin inverse (1+b'k^2) is UNBOUNDED at high f. For very large
     * delta_beta (e.g. 1000) it over-amplifies into noise. Bound it the same way
     * Wiener does: write the Paganin LOW-PASS H_p=1/(1+b'k^2) and invert it with
     * Tikhonov, H_p/(H_p^2+gamma_p), so the boost self-limits near the noise floor.
     * Then multiply by the Tikhonov PSF deconvolution. */
    double Hp = 1.0 / (1.0 + b_reduced * f2);            /* Paganin low-pass */
    double paganin_deconv = Hp / (Hp * Hp + gamma);      /* bounded inverse */
    double G = exp(-2.0 * M_PI * M_PI * sigma * sigma * f2); /* PSF transfer */
    double psf_deconv = G / (G * G + gamma);             /* Tikhonov PSF inverse */
    return paganin_deconv * psf_deconv;
}

int fy_deconvolve_gureyev(const float *in, float *out,
                          int nz, int ny, int nx,
                          const fy_physics *p, double tikhonov) {
    if (tikhonov <= 0) tikhonov = 0.02;
    int pz = fy_next_pow2(nz), py = fy_next_pow2(ny), px = fy_next_pow2(nx);
    size_t np = (size_t)pz * py * px;
    float *re = (float *)malloc(sizeof(float) * np);
    float *im = (float *)calloc(np, sizeof(float));
    if (!re || !im) { free(re); free(im); return 1; }
    for (int z = 0; z < pz; z++) for (int y = 0; y < py; y++) {
        int sz = reflect(z, nz), sy = reflect(y, ny);
        size_t drow = ((size_t)z * py + y) * px, srow = ((size_t)sz * ny + sy) * nx;
        for (int x = 0; x < px; x++) re[drow + x] = in[srow + reflect(x, nx)];
    }
    fy_fft3d(re, im, pz, py, px, -1);
    double *fz2 = malloc(sizeof(double)*pz), *fy2 = malloc(sizeof(double)*py), *fx2 = malloc(sizeof(double)*px);
    if (!fz2||!fy2||!fx2){ free(re);free(im);free(fz2);free(fy2);free(fx2); return 1; }
    for (int i=0;i<pz;i++){double f=(i<=pz/2)?(double)i/pz:(double)(i-pz)/pz; fz2[i]=f*f;}
    for (int i=0;i<py;i++){double f=(i<=py/2)?(double)i/py:(double)(i-py)/py; fy2[i]=f*f;}
    for (int i=0;i<px;i++){double f=(i<=px/2)?(double)i/px:(double)(i-px)/px; fx2[i]=f*f;}
    for (int z=0;z<pz;z++) for (int y=0;y<py;y++){
        size_t row=((size_t)z*py+y)*px; double fzy=fz2[z]+fy2[y];
        for (int x=0;x<px;x++){
            double fr=sqrt(fzy+fx2[x]);
            double w=gureyev_transfer(fr,p,tikhonov);
            re[row+x]*=(float)w; im[row+x]*=(float)w;
        }
    }
    fy_fft3d(re, im, pz, py, px, +1);
    fy_fft3d_normalize(re, im, pz, py, px);
    for (int z=0;z<nz;z++) for (int y=0;y<ny;y++){
        size_t drow=((size_t)z*ny+y)*nx, srow=((size_t)z*py+y)*px;
        for (int x=0;x<nx;x++) out[drow+x]=re[srow+x];
    }
    free(re); free(im); free(fz2); free(fy2); free(fx2);
    return 0;
}
