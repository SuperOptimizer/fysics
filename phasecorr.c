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

    /* Two cross-power spectra of C = conj(F)*M = (fr-i fi)(mr+i mi):
     *   real = fr*mr + fi*mi ;  imag = fr*mi - fi*mr
     * (a) PHASE-ONLY R = C/|C| -> a near-delta correlation surface, sharp integer
     *     peak + robust to contrast (magnitude discarded). Stored in (fr,fi).
     * (b) UN-NORMALIZED C (whitened only mildly) -> a SMOOTH band-limited peak whose
     *     true max sits at the sub-voxel shift WITHOUT the Dirichlet real/imag
     *     ambiguity of the phase-only surface. Stored in (cr2,ci2) for the
     *     upsampled-DFT sub-voxel refine. */
    float *cr2 = malloc(sizeof(float)*pn);
    float *ci2 = malloc(sizeof(float)*pn);
    if (!cr2 || !ci2) { free(fr);free(fi);free(mr);free(mi);free(cr2);free(ci2); return 1; }
    double maxmag = 0.0;
    for (size_t i = 0; i < pn; i++) {
        float cr = fr[i]*mr[i] + fi[i]*mi[i];
        float ci = fr[i]*mi[i] - fi[i]*mr[i];
        cr2[i] = cr; ci2[i] = ci;                 /* (b) un-normalized */
        float mag = sqrtf(cr*cr + ci*ci);
        if (mag > maxmag) maxmag = mag;
        if (mag > 1e-12f) { fr[i] = cr/mag; fi[i] = ci/mag; }  /* (a) phase-only */
        else { fr[i] = 0.0f; fi[i] = 0.0f; }
    }
    if (maxmag < 1e-20) {  /* one image flat -> no phase info */
        free(fr); free(fi); free(mr); free(mi); free(cr2); free(ci2);
        shift[0]=shift[1]=shift[2]=0.0; if (peak) *peak = 0.0; return 1;
    }
    free(mr); free(mi);

    /* integer-voxel peak from the PHASE-ONLY surface (sharp). */
    float *cr = malloc(sizeof(float)*pn);
    float *ci = malloc(sizeof(float)*pn);
    if (!cr || !ci) { free(fr);free(fi);free(cr2);free(ci2);free(cr);free(ci); return 1; }
    memcpy(cr, fr, sizeof(float)*pn);
    memcpy(ci, fi, sizeof(float)*pn);
    fy_fft3d(cr, ci, pz, py, px, +1);

    size_t pk = 0; float best = cr[0];
    for (size_t i = 1; i < pn; i++) if (cr[i] > best) { best = cr[i]; pk = i; }
    int kz = (int)(pk / ((size_t)py*px));
    int ky = (int)((pk / px) % py);
    int kx = (int)(pk % px);
    /* unwrap integer index to signed shift on the padded grid */
    int iz = (kz > pz/2) ? kz - pz : kz;
    int iy = (ky > py/2) ? ky - py : ky;
    int ix = (kx > px/2) ? kx - px : kx;
    free(cr); free(ci);

    /* ---- sub-voxel refine: UPSAMPLED-DFT (Guizar-Sicairos) ------------------
     * Evaluate the inverse DFT of the UN-NORMALIZED cross-power C on a fine grid
     * of step 1/usf spanning +/-1 voxel about the integer peak, via direct
     * (matrix) DFT -- separable, so O(W^3*(pz+py+px)) not O(N^2). We refine on C
     * (a smooth band-limited cross-correlation, |.| has its true max AT the
     * shift) rather than the phase-only surface (a Dirichlet kernel whose
     * real/imag split biases a sub-voxel max). The integer peak above (from the
     * sharp phase-only surface) sets the search center. IDFT sample at continuous
     * position d is sum_k C[k] exp(+i 2pi k d / p). */
    const int usf = 50;         /* 1/50-voxel grid */
    const double half = 1.0;    /* search +/-1.0 vox about the integer peak */
    int W = (int)(2*half*usf) + 1;
    /* precompute per-axis complex basis e[w][k] = exp(+i 2pi k d_w / p),
     * d_w = (peak + (w-center)/usf). Build as separable 1-D DFT evaluators. */
    /* To keep memory modest we evaluate the separable sum directly. Build, for
     * each axis, tables of the sampled position. */
    double bestv = -1e300, rdz=iz, rdy=iy, rdx=ix;
    /* Precompute axis phase tables: za[w][kz] etc would be large; instead do the
     * standard 3-stage separable contraction. Stage A: contract X. */
    /* positions along each axis */
    double *posz = malloc(sizeof(double)*W);
    double *posy = malloc(sizeof(double)*W);
    double *posx = malloc(sizeof(double)*W);
    for (int w=0; w<W; w++) {
        double off = (w - (W-1)/2)/(double)usf;
        posz[w] = iz + off; posy[w] = iy + off; posx[w] = ix + off;
    }
    /* cz3[w][kz], etc.: exp(+i 2pi kz posz[w]/pz). Store as 2 floats interleaved. */
    /* memory: W*p complex doubles per axis -> fine for W~301, p<=~512. */
    double *Ezr = malloc(sizeof(double)*W*pz), *Ezi = malloc(sizeof(double)*W*pz);
    double *Eyr = malloc(sizeof(double)*W*py), *Eyi = malloc(sizeof(double)*W*py);
    double *Exr = malloc(sizeof(double)*W*px), *Exi = malloc(sizeof(double)*W*px);
    if(!posz||!posy||!posx||!Ezr||!Ezi||!Eyr||!Eyi||!Exr||!Exi){
        free(fr);free(fi);free(cr2);free(ci2);free(posz);free(posy);free(posx);
        free(Ezr);free(Ezi);free(Eyr);free(Eyi);free(Exr);free(Exi);
        shift[0]=iz;shift[1]=iy;shift[2]=ix; if(peak)*peak=(double)best/(double)pn; return 0;
    }
    for (int w=0; w<W; w++) for (int k=0;k<pz;k++){ double a=2.0*M_PI*k*posz[w]/pz; Ezr[w*pz+k]=cos(a);Ezi[w*pz+k]=sin(a);}
    for (int w=0; w<W; w++) for (int k=0;k<py;k++){ double a=2.0*M_PI*k*posy[w]/py; Eyr[w*py+k]=cos(a);Eyi[w*py+k]=sin(a);}
    for (int w=0; w<W; w++) for (int k=0;k<px;k++){ double a=2.0*M_PI*k*posx[w]/px; Exr[w*px+k]=cos(a);Exi[w*px+k]=sin(a);}

    /* Stage A: Ax[kz][ky][wx] = sum_kx R[kz,ky,kx] * Ex[wx,kx]  (contract X) */
    double *Axr = malloc(sizeof(double)*(size_t)pz*py*W);
    double *Axi = malloc(sizeof(double)*(size_t)pz*py*W);
    /* Stage B: Bx[kz][wy][wx] = sum_ky Ax[kz,ky,wx]*Ey[wy,ky] */
    double *Bxr = malloc(sizeof(double)*(size_t)pz*W*W);
    double *Bxi = malloc(sizeof(double)*(size_t)pz*W*W);
    if(!Axr||!Axi||!Bxr||!Bxi){
        free(fr);free(fi);free(cr2);free(ci2);free(posz);free(posy);free(posx);
        free(Ezr);free(Ezi);free(Eyr);free(Eyi);free(Exr);free(Exi);free(Axr);free(Axi);free(Bxr);free(Bxi);
        shift[0]=iz;shift[1]=iy;shift[2]=ix; if(peak)*peak=(double)best/(double)pn; return 0;
    }
    for (int z=0; z<pz; z++) for (int y=0; y<py; y++) {
        const float *Rr = cr2 + ((size_t)z*py+y)*px;
        const float *Ri = ci2 + ((size_t)z*py+y)*px;
        for (int wx=0; wx<W; wx++) {
            const double *exr=Exr+wx*px, *exi=Exi+wx*px;
            double sr=0, si=0;
            for (int k=0;k<px;k++){ sr += Rr[k]*exr[k] - Ri[k]*exi[k]; si += Rr[k]*exi[k] + Ri[k]*exr[k]; }
            Axr[((size_t)z*py+y)*W+wx]=sr; Axi[((size_t)z*py+y)*W+wx]=si;
        }
    }
    for (int z=0; z<pz; z++) for (int wy=0; wy<W; wy++) {
        const double *eyr=Eyr+wy*py, *eyi=Eyi+wy*py;
        for (int wx=0; wx<W; wx++) {
            double sr=0,si=0;
            for (int y=0;y<py;y++){ double ar=Axr[((size_t)z*py+y)*W+wx], ai=Axi[((size_t)z*py+y)*W+wx];
                sr += ar*eyr[y]-ai*eyi[y]; si += ar*eyi[y]+ai*eyr[y]; }
            Bxr[((size_t)z*W+wy)*W+wx]=sr; Bxi[((size_t)z*W+wy)*W+wx]=si;
        }
    }
    /* Stage C: contract Z and find argmax magnitude over (wz,wy,wx) */
    for (int wz=0; wz<W; wz++) {
        const double *ezr=Ezr+wz*pz, *ezi=Ezi+wz*pz;
        for (int wy=0; wy<W; wy++) for (int wx=0; wx<W; wx++) {
            double sr=0,si=0;
            for (int z=0;z<pz;z++){ double br=Bxr[((size_t)z*W+wy)*W+wx], bi=Bxi[((size_t)z*W+wy)*W+wx];
                sr += br*ezr[z]-bi*ezi[z]; si += br*ezi[z]+bi*ezr[z]; }
            double mag = sr*sr + si*si;  /* |cross-correlation| at this sub-voxel pos */
            if (mag > bestv) { bestv=mag; rdz=posz[wz]; rdy=posy[wy]; rdx=posx[wx]; }
        }
    }
    shift[0]=rdz; shift[1]=rdy; shift[2]=rdx;
    /* normalized peak height in [0,1]: the un-normalized IFFT value / N.
     * For a clean single-shift match this approaches 1; lower = weaker match. */
    if (peak) *peak = (double)best / (double)pn;

    free(fr); free(fi); free(cr2); free(ci2);
    free(posz);free(posy);free(posx);
    free(Ezr);free(Ezi);free(Eyr);free(Eyi);free(Exr);free(Exi);
    free(Axr);free(Axi);free(Bxr);free(Bxi);
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
