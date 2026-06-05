/* phasecorr.c -- sub-voxel translation (phase correlation) + Mutual Information.
 *
 * These are the two LOCAL similarity tools that bootstrap a coarse landmark seed
 * to SUB-VOXEL registration on the self-similar laminar scroll. Global intensity
 * NCC is non-discriminative on papyrus (it is self-similar across the whole
 * volume), but a single TEXTURED patch is not self-similar with itself under a
 * small shift, so a local metric on a textured patch is sharp.
 *
 * Geometry/index convention matches the rest of fysics: row-major, x fastest,
 * idx = (z*ny+y)*nx + x, axis order (z,y,x).
 */
#include "fysics.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- phase correlation -----------------------------------------------------
 * Fourier shift theorem: a pure translation t between f and m shows up as a
 * linear phase ramp in F* . M / |F* . M| (the normalized cross-power spectrum);
 * its inverse FFT is a delta at the shift. We:
 *   1) remove the mean and apply a separable Hann window (kills the FFT's
 *      periodic wrap-around for a non-periodic patch -> a clean single peak),
 *   2) zero-pad into the next-pow2 cube (fy_fft3d needs powers of two),
 *   3) form the normalized cross-power spectrum and inverse-FFT it,
 *   4) find the integer-voxel peak, unwrap it to a signed shift, and refine to
 *      sub-voxel by the Foroosh (2002) phase-correlation estimator along each
 *      axis (the model-correct inverse of the Dirichlet peak sampling; a parabola
 *      assumes a smooth peak and biases toward the integer).
 *
 * Returned shift (dz,dy,dx) is the displacement that aligns `moving` onto
 * `fixed`: moving(p) ~ fixed(p - shift). (If moving = roll(fixed, +s) then we
 * recover shift = +s.) Magnitude robust to brightness/contrast (the |.|
 * normalization discards magnitude, keeping only phase = the shift). */

/* 1-D Hann weights into w[0..n-1]. */
static void hann1d(float *w, int n) {
    if (n == 1) { w[0] = 1.0f; return; }
    for (int i = 0; i < n; i++)
        w[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (float)(n - 1));
}

/* Foroosh sub-voxel offset along one axis from the peak sample c0 and its two
 * neighbours (cm at -1, cp at +1). The phase-correlation Dirichlet main lobe
 * gives |delta| = cs/(cs +/- c0) toward the LARGER neighbour cs; pick the root in
 * [-1,1]. Returns the signed offset in (-1,1) of the true peak from c0. */
static double foroosh_off(double cm, double c0, double cp) {
    double cs; double sgn;
    if (cp >= cm) { cs = cp; sgn = +1.0; } else { cs = cm; sgn = -1.0; }
    if (cs <= 0.0) return 0.0;                 /* both neighbours non-positive */
    double d1 = cs / (cs + c0);
    double d2 = cs / (cs - c0);
    double d = (fabs(d1) <= fabs(d2)) ? d1 : d2;
    if (d > 1.0) d = 1.0; else if (d < -1.0) d = -1.0;
    return sgn * fabs(d);
}

int fy_phase_correlate(const float *fixed, const float *moving,
                       int nz, int ny, int nx,
                       double *shift, double *peak, int window) {
    if (!fixed || !moving || !shift || nz < 2 || ny < 2 || nx < 2) return 1;

    int pz = fy_next_pow2(nz), py = fy_next_pow2(ny), px = fy_next_pow2(nx);
    size_t pn = (size_t)pz * py * px;

    float *fr = calloc(pn, sizeof(float));
    float *fi = calloc(pn, sizeof(float));
    float *mr = calloc(pn, sizeof(float));
    float *mi = calloc(pn, sizeof(float));
    if (!fr || !fi || !mr || !mi) { free(fr); free(fi); free(mr); free(mi); return 1; }

    /* per-axis Hann windows (or all-ones) */
    float *wz = malloc(sizeof(float) * nz);
    float *wy = malloc(sizeof(float) * ny);
    float *wx = malloc(sizeof(float) * nx);
    if (!wz || !wy || !wx) { free(fr);free(fi);free(mr);free(mi);free(wz);free(wy);free(wx); return 1; }
    if (window) { hann1d(wz, nz); hann1d(wy, ny); hann1d(wx, nx); }
    else { for (int i=0;i<nz;i++) wz[i]=1; for (int i=0;i<ny;i++) wy[i]=1; for (int i=0;i<nx;i++) wx[i]=1; }

    /* means over the (unwindowed) input */
    double sf = 0, sm = 0;
    size_t n = (size_t)nz * ny * nx;
    for (int z=0; z<nz; z++) for (int y=0; y<ny; y++) {
        const float *fp = fixed + ((size_t)z*ny+y)*nx;
        const float *mp = moving + ((size_t)z*ny+y)*nx;
        for (int x=0; x<nx; x++) { sf += fp[x]; sm += mp[x]; }
    }
    float mf = (float)(sf / n), mm = (float)(sm / n);

    /* fill padded real parts: (val-mean)*window, into the [0..n) corner */
    for (int z=0; z<nz; z++) for (int y=0; y<ny; y++) {
        const float *fp = fixed + ((size_t)z*ny+y)*nx;
        const float *mp = moving + ((size_t)z*ny+y)*nx;
        float wzy = wz[z]*wy[y];
        size_t pbase = ((size_t)z*py+y)*px;
        for (int x=0; x<nx; x++) {
            float w = wzy * wx[x];
            fr[pbase+x] = (fp[x]-mf)*w;
            mr[pbase+x] = (mp[x]-mm)*w;
        }
    }
    free(wz); free(wy); free(wx);

    fy_fft3d(fr, fi, pz, py, px, -1);
    fy_fft3d(mr, mi, pz, py, px, -1);

    /* Normalized cross-power spectrum R = conj(F)*M / |conj(F)*M|, stored in
     * (fr,fi). conj(F)*M = (fr-i fi)(mr+i mi): real = fr*mr+fi*mi, imag = fr*mi-fi*mr.
     * Normalizing to unit magnitude keeps ONLY the phase ramp (=the shift) and
     * discards spectral magnitude -> robust to brightness/contrast (cross-energy). */
    double maxmag = 0.0;
    for (size_t i = 0; i < pn; i++) {
        float cr = fr[i]*mr[i] + fi[i]*mi[i];
        float ci = fr[i]*mi[i] - fi[i]*mr[i];
        float mag = sqrtf(cr*cr + ci*ci);
        if (mag > maxmag) maxmag = mag;
        if (mag > 1e-12f) { fr[i] = cr/mag; fi[i] = ci/mag; }
        else { fr[i] = 0.0f; fi[i] = 0.0f; }
    }
    free(mr); free(mi);
    if (maxmag < 1e-20) {  /* one image flat -> no phase info */
        free(fr); free(fi);
        shift[0]=shift[1]=shift[2]=0.0; if (peak) *peak = 0.0; return 1;
    }

    /* inverse FFT -> phase-correlation surface in fr (real; ci negligible). */
    fy_fft3d(fr, fi, pz, py, px, +1);
    fy_fft3d_normalize(fr, fi, pz, py, px);

    /* integer-voxel peak */
    size_t pk = 0; float best = fr[0];
    for (size_t i = 1; i < pn; i++) if (fr[i] > best) { best = fr[i]; pk = i; }
    int kz = (int)(pk / ((size_t)py*px));
    int ky = (int)((pk / px) % py);
    int kx = (int)(pk % px);

    /* ---- sub-voxel refine: FOROOSH phase-correlation estimator --------------
     * The phase-correlation peak is a Dirichlet kernel; for a sub-voxel shift the
     * energy splits between the peak sample c0 and ONE neighbour cs. Foroosh
     * (2002) showed the sub-voxel offset is delta = cs/(cs +/- c0) (pick the sign
     * giving |delta|<=1), toward the larger neighbour. This is the model-correct
     * inverse of the Dirichlet sampling (a parabola, which assumes a smooth peak,
     * systematically biases toward the integer). Per-axis, using the in-line
     * neighbours of the peak (wrap on the padded grid). */
    #define PCV(zz,yy,xx) fr[((size_t)(zz)*py+(yy))*px+(xx)]
    int zm=(kz-1+pz)%pz, zp=(kz+1)%pz;
    int ym=(ky-1+py)%py, yp=(ky+1)%py;
    int xm=(kx-1+px)%px, xp=(kx+1)%px;
    double c0 = best;
    double oz = foroosh_off(PCV(zm,ky,kx), c0, PCV(zp,ky,kx));
    double oy = foroosh_off(PCV(kz,ym,kx), c0, PCV(kz,yp,kx));
    double ox = foroosh_off(PCV(kz,ky,xm), c0, PCV(kz,ky,xp));
    #undef PCV

    /* unwrap integer index to a signed shift on the padded grid, add sub-voxel */
    double dz = (double)kz + oz; if (dz > pz/2) dz -= pz;
    double dy = (double)ky + oy; if (dy > py/2) dy -= py;
    double dx = (double)kx + ox; if (dx > px/2) dx -= px;
    shift[0]=dz; shift[1]=dy; shift[2]=dx;
    if (peak) *peak = best;   /* normalized corr peak in [0,1] (1 = perfect match) */

    free(fr); free(fi);
    return 0;
}

/* ---- Mutual Information -----------------------------------------------------
 * MI from a joint histogram. Both images are scaled to [0,nbins) using their own
 * overlap min/max; this makes MI invariant to a monotone-affine intensity change
 * (the histogram bins rescale with it) while, unlike NCC, NOT assuming the
 * fixed<->moving relation is linear. */
double fy_mutual_information(const float *fixed, const float *moving,
                            int nz, int ny, int nx, const double *M, int nbins) {
    if (!fixed || !moving || !M || nbins < 2) return -1.0;
    size_t s_y = (size_t)nx, s_z = (size_t)ny * nx;

    /* pass 1: overlap min/max of each image (over in-bounds warped samples) */
    double fmin=1e300, fmax=-1e300, mmin=1e300, mmax=-1e300;
    long cnt = 0;
    for (int zo=0; zo<nz; zo++) for (int yo=0; yo<ny; yo++) {
        double bz = M[0]*zo + M[1]*yo + M[3];
        double by = M[4]*zo + M[5]*yo + M[7];
        double bx = M[8]*zo + M[9]*yo + M[11];
        const float *frow = fixed + (size_t)zo*s_z + (size_t)yo*s_y;
        for (int xo=0; xo<nx; xo++) {
            float zi=(float)(bz+M[2]*xo), yi=(float)(by+M[6]*xo), xi=(float)(bx+M[10]*xo);
            if (zi<0||yi<0||xi<0||zi>(float)(nz-1)||yi>(float)(ny-1)||xi>(float)(nx-1)) continue;
            float mv = fy_sample_trilinear(moving, nz, ny, nx, zi, yi, xi);
            float fv = frow[xo];
            if (fv<fmin)fmin=fv; if (fv>fmax)fmax=fv;
            if (mv<mmin)mmin=mv; if (mv>mmax)mmax=mv;
            cnt++;
        }
    }
    if (cnt < 32) return -1.0;
    double frange = fmax-fmin, mrange = mmax-mmin;
    if (frange < 1e-12 || mrange < 1e-12) return 0.0;  /* one flat -> 0 MI */

    long *jh = calloc((size_t)nbins*nbins, sizeof(long));
    if (!jh) return -1.0;
    double fsc = (nbins - 1e-6) / frange, msc = (nbins - 1e-6) / mrange;

    /* pass 2: accumulate the joint histogram */
    for (int zo=0; zo<nz; zo++) for (int yo=0; yo<ny; yo++) {
        double bz = M[0]*zo + M[1]*yo + M[3];
        double by = M[4]*zo + M[5]*yo + M[7];
        double bx = M[8]*zo + M[9]*yo + M[11];
        const float *frow = fixed + (size_t)zo*s_z + (size_t)yo*s_y;
        for (int xo=0; xo<nx; xo++) {
            float zi=(float)(bz+M[2]*xo), yi=(float)(by+M[6]*xo), xi=(float)(bx+M[10]*xo);
            if (zi<0||yi<0||xi<0||zi>(float)(nz-1)||yi>(float)(ny-1)||xi>(float)(nx-1)) continue;
            float mv = fy_sample_trilinear(moving, nz, ny, nx, zi, yi, xi);
            float fv = frow[xo];
            int fb = (int)((fv-fmin)*fsc); if (fb<0) fb=0; if (fb>=nbins) fb=nbins-1;
            int mb = (int)((mv-mmin)*msc); if (mb<0) mb=0; if (mb>=nbins) mb=nbins-1;
            jh[(size_t)fb*nbins+mb]++;
        }
    }

    /* marginals + MI = sum p(a,b) log( p(a,b)/(p(a)p(b)) ) in nats */
    double *pf = calloc(nbins, sizeof(double));
    double *pm = calloc(nbins, sizeof(double));
    if (!pf || !pm) { free(jh); free(pf); free(pm); return -1.0; }
    double inv = 1.0 / (double)cnt;
    for (int a=0; a<nbins; a++) for (int b=0; b<nbins; b++) {
        double p = jh[(size_t)a*nbins+b] * inv;
        pf[a] += p; pm[b] += p;
    }
    double mi = 0.0;
    for (int a=0; a<nbins; a++) for (int b=0; b<nbins; b++) {
        long c = jh[(size_t)a*nbins+b];
        if (!c) continue;
        double p = c * inv;
        double denom = pf[a]*pm[b];
        if (denom > 1e-300) mi += p * log(p/denom);
    }
    free(jh); free(pf); free(pm);
    if (mi < 0.0) mi = 0.0;   /* guard tiny negative from rounding */
    return mi;
}
