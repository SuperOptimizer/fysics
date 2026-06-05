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
 *      sub-voxel by 1-D parabolic interpolation along each axis.
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

    /* Phase-only normalized cross-power spectrum R = conj(F)*M / |conj(F)*M|.
     *   conj(F)*M = (fr - i fi)(mr + i mi):
     *     real = fr*mr + fi*mi ;  imag = fr*mi - fi*mr
     * With the test's convention M(k) = F(k)*exp(-2pi i k.s/N) (moving = fixed
     * shifted BY +s), conj(F)*M = |F|^2 exp(-2pi i k.s/N), so after |.|
     * normalization R(k) = exp(-2pi i k.s/N).  Its INVERSE FFT (sign +1,
     * sum_k R(k) exp(+2pi i k n/N)) peaks at n = s.  Discarding |.| makes the
     * estimate invariant to an affine intensity change of `moving`. */
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

    /* Correlation surface = inverse FFT of R (real part is the real-valued
     * phase-correlation function; its peak marks the integer shift). */
    fy_fft3d(fr, fi, pz, py, px, +1);
    float *surf = fr;   /* real part holds the correlation surface */

    size_t pk = 0; float best = surf[0];
    for (size_t i = 1; i < pn; i++) if (surf[i] > best) { best = surf[i]; pk = i; }
    int kz = (int)(pk / ((size_t)py*px));
    int ky = (int)((pk / px) % py);
    int kx = (int)(pk % px);

    /* Unwrap each peak index to a SIGNED shift via the fftfreq convention used by
     * the test's fourier_shift helper: index k -> k if k <= p/2 else k - p. */
    int iz = (kz <= pz/2) ? kz : kz - pz;
    int iy = (ky <= py/2) ? ky : ky - py;
    int ix = (kx <= px/2) ? kx : kx - px;

    /* Sub-voxel refine per axis by Foroosh's phase-correlation estimator. The
     * inverse FFT of a pure phase ramp exp(-2pi i k s/p) is a Dirichlet kernel,
     * which near the peak behaves like sinc -- so the two in-line side lobes c[-1]
     * and c[+1] (with circular wrap) obey c[+-1]/c[0] = d/(d -+ 1), giving the
     * unbiased estimate  d = c1/(c1 +- c0)  with the sign taken toward the LARGER
     * neighbour. Parabolic interpolation on this sinc-shaped surface is biased
     * toward 0; Foroosh is the matched estimator. */
    #define SURF(Z,Y,X) surf[(((size_t)(Z))*py + (Y))*px + (X)]
    double c0 = SURF(kz,ky,kx);
    double dz = 0.0, dy = 0.0, dx = 0.0;
    {
        int zm = (kz - 1 + pz) % pz, zp = (kz + 1) % pz;
        double cm = SURF(zm,ky,kx), cp = SURF(zp,ky,kx);
        double c1 = (cp >= cm) ? cp : cm;        /* larger side lobe */
        double s  = (cp >= cm) ? 1.0 : -1.0;     /* its direction */
        if (fabs(c1 - c0) > 1e-20 && fabs(c1 + c0) > 1e-20) {
            double d1 = c1/(c1 - c0), d2 = c1/(c1 + c0);
            dz = s * ((fabs(d1) <= fabs(d2)) ? d1 : d2);
        }
        if (dz >  0.5) dz =  0.5; if (dz < -0.5) dz = -0.5;
    }
    {
        int ym = (ky - 1 + py) % py, yp = (ky + 1) % py;
        double cm = SURF(kz,ym,kx), cp = SURF(kz,yp,kx);
        double c1 = (cp >= cm) ? cp : cm;
        double s  = (cp >= cm) ? 1.0 : -1.0;
        if (fabs(c1 - c0) > 1e-20 && fabs(c1 + c0) > 1e-20) {
            double d1 = c1/(c1 - c0), d2 = c1/(c1 + c0);
            dy = s * ((fabs(d1) <= fabs(d2)) ? d1 : d2);
        }
        if (dy >  0.5) dy =  0.5; if (dy < -0.5) dy = -0.5;
    }
    {
        int xm = (kx - 1 + px) % px, xp = (kx + 1) % px;
        double cm = SURF(kz,ky,xm), cp = SURF(kz,ky,xp);
        double c1 = (cp >= cm) ? cp : cm;
        double s  = (cp >= cm) ? 1.0 : -1.0;
        if (fabs(c1 - c0) > 1e-20 && fabs(c1 + c0) > 1e-20) {
            double d1 = c1/(c1 - c0), d2 = c1/(c1 + c0);
            dx = s * ((fabs(d1) <= fabs(d2)) ? d1 : d2);
        }
        if (dx >  0.5) dx =  0.5; if (dx < -0.5) dx = -0.5;
    }
    #undef SURF

    shift[0] = iz + dz;
    shift[1] = iy + dy;
    shift[2] = ix + dx;
    /* normalized peak height in [0,1]; ~1 for a clean single-shift match. */
    if (peak) *peak = (double)best / (double)pn;

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
