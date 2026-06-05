/* texture.c -- IN-PLANE (surface-tangent) TEXTURE ENHANCER, 3D.
 *
 * Purpose: LIFT the fine in-plane "crackle" surface texture on papyrus sheets
 * (the faint surface texture that ink sits in, per Vesuvius Grand Prize findings)
 * while SUPPRESSING cross-sheet variation (the layered structure -- NOT ink) and
 * NOT amplifying noise. It is the COMPLEMENT of the coherence-diffusion filter
 * (diffusion.c smooths ALONG sheets); this one HIGH-PASSES ALONG sheets.
 *
 * HONEST SCOPE: there are NO ground-truth ink labels here, so this is NOT an "ink
 * detector" and is not validated as one. It is the right transform IF ink is a fine
 * in-plane texture signal: it enhances in-plane texture over noise and over layering,
 * auto-calibrated. The MEASURABLE claims it proves are: (1) in-plane texture-band
 * energy goes UP, (2) texture/noise ratio IMPROVES (gating), (3) cross-sheet/layering
 * variation is NOT amplified.
 *
 * Method (a STEERED, NOISE-GATED unsharp mask using the orientation field):
 *   1. ORIENTATION FIELD -- SAME structure tensor as diffusion.c (Weickert CED):
 *        J_rho = G_rho * ( grad u_sigma (x) grad u_sigma^T ),
 *      presmooth u at noise scale sigma, central-difference gradient, 6 outer-product
 *      components, smooth each with G_rho (3-pass box ~ Gaussian, integration scale).
 *      Eigen-decompose J per voxel (eig3_sym_vec, identical to diffusion.c) ->
 *      eigenvalues mu1>=mu2>=mu3 with eigenvectors. vec[0]=SHEET NORMAL (large mu,
 *      gradient across the sheet); vec[1],vec[2]=the two IN-PLANE tangent directions.
 *   2. IN-PLANE LOW-PASS (steered): smooth u ONLY within the tangent plane (along t1,t2),
 *      NEVER across the normal. Implemented as a steered separable box: sample u along
 *      +/- k*t1 and +/- k*t2 (k=1..R) by trilinear interpolation (reflecting bounds)
 *      and average. Because the average runs in the sheet plane, a cross-sheet
 *      intensity STEP (layering) is NOT removed by it (the step is along the normal,
 *      which the plane average does not cross), so the step does NOT appear in the
 *      detail and is NOT boosted. A fine in-plane ripple (crackle) DOES appear in the
 *      detail and IS boosted. This is the key difference from a PLAIN isotropic unsharp
 *      mask, which low-passes across the normal too and therefore boosts the step.
 *   3. DETAIL + TWO GATES: detail = u - inplane_lowpass.
 *      (a) NOISE GATE (Wiener-style): detail*det^2/(det^2 + (c*nf)^2). Below the noise
 *          scale nf this suppresses QUADRATICALLY (flat-region noise is cut ~5x at c=2)
 *          while supra-noise texture passes ~untouched -> texture/noise IMPROVES instead
 *          of uniformly scaling noise up. nf is the measured noise level (or, with the
 *          <0 sentinel, the detail's own robust scale 1.4826*MAD).
 *      (b) DIRECTIONAL GATE: hp^2/(hp^2 + hn^2), hp=|u-inplane_lp| (in-plane high-pass),
 *          hn=|u-normal_lp| (cross-sheet/normal high-pass). Genuine in-plane texture has
 *          hn~0 -> pass; a layering edge (or the residual leak from a slightly tilted
 *          estimated normal) has hn>>hp -> suppress. This is the second guarantee that
 *          the cross-sheet step is NOT amplified.
 *   4. OUTPUT: out = u + gain * noise_gate(detail) * directional_gate.
 *
 * AXIS ORDER (subtle): the structure tensor is built from (gx,gy,gz), so the
 * eigenvectors come out in (X,Y,Z) component order -- vec[k][0]=x, [1]=y, [2]=z. Steered
 * sampling maps them to (z,y,x) displacements accordingly.
 *
 * Entry points (mirror the CED auto pattern):
 *   fy_texture_enhance       -- explicit knobs.
 *   fy_texture_enhance_auto  -- self-calibrating (sigma from fy_estimate_noise; the
 *                               noise gate from the detail's own MAD; rho from the
 *                               measured layer spacing;
 *                               inplane_scale from the measured in-plane texture scale;
 *                               gain from strength {1,2,3}). NO-KNOBS vc3d entry point.
 *   fy_texture_enhance_halo  -- per-side halo for streaming/tiling.
 *
 * Pure C, libc+libm only, no SIMD intrinsics. Row-major, x-fastest:
 * idx=(z*ny+y)*nx+x. float data, returns int (0=ok).
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define IDX(z,y,x) (((size_t)(z)*ny+(y))*nx+(x))
#define CLAMP(v,lo,hi) ((v)<(lo)?(lo):((v)>(hi)?(hi):(v)))

static int cmp_float_asc(const void *a, const void *b){
    float x=*(const float*)a, y=*(const float*)b;
    return (x>y)-(x<y);
}

/* ---- separable Gaussian presmooth (noise scale). Reflecting boundary. ----
 * (identical approach to diffusion.c's gauss_blur) */
static void tex_gauss_blur(const float *in, float *out, float *tmp,
                           int nz, int ny, int nx, double sigma) {
    if (sigma <= 0) { memcpy(out, in, sizeof(float)*(size_t)nz*ny*nx); return; }
    int r = (int)ceil(3.0 * sigma); if (r < 1) r = 1;
    int w = 2*r+1;
    double *k = malloc(sizeof(double)*w); double sum=0;
    for (int i=-r;i<=r;i++){ k[i+r]=exp(-0.5*i*i/(sigma*sigma)); sum+=k[i+r]; }
    for (int i=0;i<w;i++) k[i]/=sum;
    float *a=out, *b=tmp;
    for (int z=0;z<nz;z++) for (int y=0;y<ny;y++){ size_t row=((size_t)z*ny+y)*nx;
        for (int x=0;x<nx;x++){ double s=0; for(int t=-r;t<=r;t++){int xx=CLAMP(x+t,0,nx-1);s+=k[t+r]*in[row+xx];} a[row+x]=(float)s; } }
    for (int z=0;z<nz;z++) for (int x=0;x<nx;x++)
        for (int y=0;y<ny;y++){ double s=0; for(int t=-r;t<=r;t++){int yy=CLAMP(y+t,0,ny-1);s+=k[t+r]*a[IDX(z,yy,x)];} b[IDX(z,y,x)]=(float)s; }
    for (int y=0;y<ny;y++) for (int x=0;x<nx;x++)
        for (int z=0;z<nz;z++){ double s=0; for(int t=-r;t<=r;t++){int zz=CLAMP(z+t,0,nz-1);s+=k[t+r]*b[IDX(zz,y,x)];} a[IDX(z,y,x)]=(float)s; }
    if (a != out) memcpy(out, a, sizeof(float)*(size_t)nz*ny*nx);
    free(k);
}

/* ---- separable box filter, reflecting boundary (one pass) ---- */
static void tex_box_pass(const float *in, float *out, float *tmp,
                         int nz, int ny, int nx, int r) {
    if (r < 1) { memcpy(out, in, sizeof(float)*(size_t)nz*ny*nx); return; }
    double inv = 1.0/(2*r+1);
    float *a=out, *b=tmp;
    for (int z=0;z<nz;z++) for (int y=0;y<ny;y++){ size_t row=((size_t)z*ny+y)*nx;
        for (int x=0;x<nx;x++){ double s=0; for(int t=-r;t<=r;t++){int xx=CLAMP(x+t,0,nx-1);s+=in[row+xx];} a[row+x]=(float)(s*inv); } }
    for (int z=0;z<nz;z++) for (int x=0;x<nx;x++)
        for (int y=0;y<ny;y++){ double s=0; for(int t=-r;t<=r;t++){int yy=CLAMP(y+t,0,ny-1);s+=a[IDX(z,yy,x)];} b[IDX(z,y,x)]=(float)(s*inv); }
    for (int y=0;y<ny;y++) for (int x=0;x<nx;x++)
        for (int z=0;z<nz;z++){ double s=0; for(int t=-r;t<=r;t++){int zz=CLAMP(z+t,0,nz-1);s+=b[IDX(zz,y,x)];} a[IDX(z,y,x)]=(float)(s*inv); }
    if (a != out) memcpy(out, a, sizeof(float)*(size_t)nz*ny*nx);
}

/* 3-pass box ~ Gaussian G_rho (smooth structure-tensor components). */
static void tex_box_gauss(float *buf, float *t1, float *t2, int nz,int ny,int nx, double rho){
    if (rho <= 0) return;
    int r = (int)floor(rho); if (r < 1) r = 1;
    tex_box_pass(buf, t1, t2, nz,ny,nx, r);
    tex_box_pass(t1, buf, t2, nz,ny,nx, r);
    tex_box_pass(buf, t1, t2, nz,ny,nx, r);
    memcpy(buf, t1, sizeof(float)*(size_t)nz*ny*nx);
}

/* ---- symmetric 3x3 eigen-decomposition (eigenvalues DESCENDING l1>=l2>=l3, with
 * orthonormal eigenvectors). IDENTICAL to diffusion.c's eig3_sym_vec (eigenvalue
 * part mirrors sheetness.c's eig3_sym; eigenvectors via null-space cross products
 * + Gram-Schmidt). Duplicated here (both are file-static in their TUs). ---- */
static void eig3_sym_vec(double a,double b,double c,double d,double e,double f,
                         double lam[3], double vec[3][3]) {
    double p1 = b*b + c*c + e*e;
    if (p1 == 0.0) {
        double v[3]={a,d,f}; int idx[3]={0,1,2};
        for(int i=0;i<2;i++)for(int j=0;j<2-i;j++) if(v[idx[j]]<v[idx[j+1]]){int t=idx[j];idx[j]=idx[j+1];idx[j+1]=t;}
        for(int i=0;i<3;i++){ lam[i]=v[idx[i]]; vec[i][0]=vec[i][1]=vec[i][2]=0; vec[i][idx[i]]=1.0; }
        return;
    }
    double q=(a+d+f)/3.0;
    double p2=(a-q)*(a-q)+(d-q)*(d-q)+(f-q)*(f-q)+2*p1;
    double p=sqrt(p2/6.0);
    double ba=(a-q)/p, bd=(d-q)/p, bf=(f-q)/p, bb=b/p, bc=c/p, be=e/p;
    double detB = ba*(bd*bf-be*be)-bb*(bb*bf-be*bc)+bc*(bb*be-bd*bc);
    double r=detB/2.0; if(r<-1)r=-1; if(r>1)r=1;
    double phi=acos(r)/3.0;
    double eig1=q+2*p*cos(phi);
    double eig3=q+2*p*cos(phi+2.0*M_PI/3.0);
    double eig2=3*q-eig1-eig3;
    double ev[3]={eig1,eig2,eig3};
    for(int i=0;i<2;i++)for(int j=0;j<2-i;j++) if(ev[j]<ev[j+1]){double t=ev[j];ev[j]=ev[j+1];ev[j+1]=t;}
    lam[0]=ev[0]; lam[1]=ev[1]; lam[2]=ev[2];
    for (int k=0;k<3;k++){
        double L=lam[k];
        double m00=a-L,m01=b,m02=c, m11=d-L,m12=e, m22=f-L;
        double rows[3][3]={{m00,m01,m02},{m01,m11,m12},{m02,m12,m22}};
        double best[3]={0,0,0}; double bestn=-1;
        for (int i=0;i<3;i++){ int j=(i+1)%3;
            double cx=rows[i][1]*rows[j][2]-rows[i][2]*rows[j][1];
            double cy=rows[i][2]*rows[j][0]-rows[i][0]*rows[j][2];
            double cz=rows[i][0]*rows[j][1]-rows[i][1]*rows[j][0];
            double nn=cx*cx+cy*cy+cz*cz;
            if (nn>bestn){bestn=nn;best[0]=cx;best[1]=cy;best[2]=cz;}
        }
        if (bestn <= 1e-30) { best[0]=(k==0);best[1]=(k==1);best[2]=(k==2); bestn=1; }
        double inv=1.0/sqrt(best[0]*best[0]+best[1]*best[1]+best[2]*best[2]);
        vec[k][0]=best[0]*inv; vec[k][1]=best[1]*inv; vec[k][2]=best[2]*inv;
    }
    for (int k=1;k<3;k++){
        for (int j=0;j<k;j++){
            double dot=vec[k][0]*vec[j][0]+vec[k][1]*vec[j][1]+vec[k][2]*vec[j][2];
            vec[k][0]-=dot*vec[j][0]; vec[k][1]-=dot*vec[j][1]; vec[k][2]-=dot*vec[j][2];
        }
        double nn=vec[k][0]*vec[k][0]+vec[k][1]*vec[k][1]+vec[k][2]*vec[k][2];
        if (nn<1e-20){
            double ax[3]={1,0,0};
            if (fabs(vec[0][0])>0.9){ax[0]=0;ax[1]=1;}
            for (int j=0;j<k;j++){ double dot=ax[0]*vec[j][0]+ax[1]*vec[j][1]+ax[2]*vec[j][2];
                ax[0]-=dot*vec[j][0];ax[1]-=dot*vec[j][1];ax[2]-=dot*vec[j][2]; }
            nn=ax[0]*ax[0]+ax[1]*ax[1]+ax[2]*ax[2];
            vec[k][0]=ax[0];vec[k][1]=ax[1];vec[k][2]=ax[2];
        }
        double inv=1.0/sqrt(nn); vec[k][0]*=inv;vec[k][1]*=inv;vec[k][2]*=inv;
    }
}

/* trilinear sample of `u` at continuous (z,y,x), reflecting (clamped) boundary. */
static inline double sample_trilin(const float *u, int nz,int ny,int nx,
                                   double zc,double yc,double xc){
    if (zc<0)zc=0; if (zc>nz-1)zc=nz-1;
    if (yc<0)yc=0; if (yc>ny-1)yc=ny-1;
    if (xc<0)xc=0; if (xc>nx-1)xc=nx-1;
    int z0=(int)floor(zc), y0=(int)floor(yc), x0=(int)floor(xc);
    int z1=z0<nz-1?z0+1:z0, y1=y0<ny-1?y0+1:y0, x1=x0<nx-1?x0+1:x0;
    double fz=zc-z0, fy=yc-y0, fx=xc-x0;
    double c000=u[IDX(z0,y0,x0)], c001=u[IDX(z0,y0,x1)];
    double c010=u[IDX(z0,y1,x0)], c011=u[IDX(z0,y1,x1)];
    double c100=u[IDX(z1,y0,x0)], c101=u[IDX(z1,y0,x1)];
    double c110=u[IDX(z1,y1,x0)], c111=u[IDX(z1,y1,x1)];
    double c00=c000*(1-fx)+c001*fx, c01=c010*(1-fx)+c011*fx;
    double c10=c100*(1-fx)+c101*fx, c11=c110*(1-fx)+c111*fx;
    double c0=c00*(1-fy)+c01*fy, c1=c10*(1-fy)+c11*fy;
    return c0*(1-fz)+c1*fz;
}

/* ---- IN-PLANE TEXTURE ENHANCER (explicit) ----
 *   in,out       : nz*ny*nx float (out may differ from in; in is not modified)
 *   sigma        : noise-scale presmoothing for the orientation gradient (~0.6-1 vox)
 *   rho          : integration scale = sheet/orientation coherence scale (~2-4 vox)
 *   gain         : unsharp gain on the in-plane detail (>0; ~0.5-3)
 *   inplane_scale: radius (vox) of the steered in-plane low-pass; the detail band is
 *                  texture finer than ~inplane_scale within the sheet plane (~1.5-4)
 *   noise_floor  : detail magnitudes below this (data units) are CORED (soft-threshold)
 *                  so noise is not amplified. ~the measured noise std. <=0 disables.
 * Returns 0 on success, 1 on allocation failure. */
int fy_texture_enhance(const float *in, float *out, int nz, int ny, int nx,
                       double sigma, double rho, double gain,
                       double inplane_scale, double noise_floor) {
    if (nz<1||ny<1||nx<1) return 1;
    if (sigma < 0) sigma = 0.7;
    if (rho   <= 0) rho   = 3.0;
    if (gain  <  0) gain  = 1.0;
    if (inplane_scale <= 0) inplane_scale = 2.0;
    /* noise_floor < 0 is a SENTINEL: auto-derive the floor from the detail's robust
     * scale (median absolute deviation) below. >=0 is used as an explicit threshold. */

    size_t n=(size_t)nz*ny*nx;
    float *us  = malloc(sizeof(float)*n);   /* presmoothed (for gradient)         */
    float *t1  = malloc(sizeof(float)*n);
    float *t2  = malloc(sizeof(float)*n);
    float *J11=malloc(sizeof(float)*n),*J12=malloc(sizeof(float)*n),*J13=malloc(sizeof(float)*n);
    float *J22=malloc(sizeof(float)*n),*J23=malloc(sizeof(float)*n),*J33=malloc(sizeof(float)*n);
    float *lp =malloc(sizeof(float)*n);     /* in-plane steered low-pass          */
    if(!us||!t1||!t2||!J11||!J12||!J13||!J22||!J23||!J33||!lp){
        free(us);free(t1);free(t2);free(J11);free(J12);free(J13);
        free(J22);free(J23);free(J33);free(lp); return 1;
    }

    /* (1) structure tensor -- SAME as diffusion.c */
    tex_gauss_blur(in, us, t1, nz,ny,nx, sigma);
    for (int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        int xm=CLAMP(x-1,0,nx-1),xp=CLAMP(x+1,0,nx-1);
        int ym=CLAMP(y-1,0,ny-1),yp=CLAMP(y+1,0,ny-1);
        int zm=CLAMP(z-1,0,nz-1),zp=CLAMP(z+1,0,nz-1);
        double gx=0.5*(us[IDX(z,y,xp)]-us[IDX(z,y,xm)]);
        double gy=0.5*(us[IDX(z,yp,x)]-us[IDX(z,ym,x)]);
        double gz=0.5*(us[IDX(zp,y,x)]-us[IDX(zm,y,x)]);
        size_t i=IDX(z,y,x);
        J11[i]=(float)(gx*gx); J12[i]=(float)(gx*gy); J13[i]=(float)(gx*gz);
        J22[i]=(float)(gy*gy); J23[i]=(float)(gy*gz); J33[i]=(float)(gz*gz);
    }
    tex_box_gauss(J11,t1,t2,nz,ny,nx,rho); tex_box_gauss(J12,t1,t2,nz,ny,nx,rho);
    tex_box_gauss(J13,t1,t2,nz,ny,nx,rho); tex_box_gauss(J22,t1,t2,nz,ny,nx,rho);
    tex_box_gauss(J23,t1,t2,nz,ny,nx,rho); tex_box_gauss(J33,t1,t2,nz,ny,nx,rho);

    /* (2,3) per-voxel: eigen-decompose -> sheet normal vec[0] and in-plane tangents
     * vec[1],vec[2]. Build TWO steered averages of `in` (trilinear sampling):
     *   lp     = IN-PLANE low-pass  (average along +/- k*t1, +/- k*t2)   -> stored in lp
     *   hn_mag = |in - NORMAL low-pass| = the cross-sheet (normal-direction) high-pass
     *            magnitude (average along +/- k*normal)                  -> stored in us
     * The in-plane detail = in - lp. But the in-plane sampling LEAKS a little of the
     * cross-sheet step when the estimated normal is slightly tilted (a tiny tangent
     * z-component crosses a sharp layering edge -> a large spurious detail). hn_mag
     * measures exactly that cross-sheet variation, and we use it below as a DIRECTIONAL
     * GATE: where variation along the normal dominates (a layering edge), the detail is
     * suppressed; where it is genuinely in-plane (texture, hn_mag~0) it is passed. */
    int R = (int)floor(inplane_scale + 0.5); if (R < 1) R = 1;
    for (int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        size_t i=IDX(z,y,x);
        double lam[3],vec[3][3];
        eig3_sym_vec(J11[i],J12[i],J13[i],J22[i],J23[i],J33[i],lam,vec);
        /* NOTE: the structure tensor is built from (gx,gy,gz), so eigenvector
         * components are in (X,Y,Z) order: vec[k][0]=x, vec[k][1]=y, vec[k][2]=z.
         * Displacements below must therefore use (dz=vec[][2], dy=vec[][1], dx=vec[][0]). */
        double n_dx=vec[0][0],n_dy=vec[0][1],n_dz=vec[0][2];   /* sheet normal       */
        double a_dx=vec[1][0],a_dy=vec[1][1],a_dz=vec[1][2];   /* in-plane tangent 1 */
        double b_dx=vec[2][0],b_dy=vec[2][1],b_dz=vec[2][2];   /* in-plane tangent 2 */
        double acc = in[i]; int cnt = 1;      /* in-plane low-pass accumulator */
        double accn = in[i]; int cntn = 1;    /* normal low-pass accumulator   */
        for (int k=1;k<=R;k++){
            acc += sample_trilin(in,nz,ny,nx, z+k*a_dz, y+k*a_dy, x+k*a_dx);
            acc += sample_trilin(in,nz,ny,nx, z-k*a_dz, y-k*a_dy, x-k*a_dx);
            acc += sample_trilin(in,nz,ny,nx, z+k*b_dz, y+k*b_dy, x+k*b_dx);
            acc += sample_trilin(in,nz,ny,nx, z-k*b_dz, y-k*b_dy, x-k*b_dx);
            cnt += 4;
            accn += sample_trilin(in,nz,ny,nx, z+k*n_dz, y+k*n_dy, x+k*n_dx);
            accn += sample_trilin(in,nz,ny,nx, z-k*n_dz, y-k*n_dy, x-k*n_dx);
            cntn += 2;
        }
        lp[i] = (float)(acc / cnt);
        us[i] = (float)fabs((double)in[i] - accn/cntn);   /* normal high-pass mag */
    }

    /* in-plane detail = in - inplane_lowpass, stored back into `lp`. */
    for (size_t i=0;i<n;i++) lp[i] = in[i] - lp[i];

    /* noise floor: if the sentinel (<0) was passed, derive it from the detail's robust
     * scale. The detail in a flat region is high-pass noise; its median-absolute-
     * deviation (MAD) is a robust noise estimate that is MATCHED to the quantity being
     * thresholded (far more reliable than mapping an absolute intensity-domain noise
     * model). sigma_noise ~ 1.4826*MAD; we core at ~1x that so sub-noise detail dies. */
    double nf = noise_floor;
    if (nf < 0) {
        /* MAD via median of |detail| (matched to the gated quantity). t2 = scratch. */
        for (size_t i=0;i<n;i++) t2[i] = (float)fabs((double)lp[i]);
        qsort(t2, n, sizeof(float), cmp_float_asc);   /* tiles are <=256^3: fine */
        double mad = t2[n/2];
        nf = 1.4826 * mad;               /* ~1 sigma of the detail-band noise */
        if (nf < 0) nf = 0;
    }

    /* out = in + gain * NOISE_GATE(detail) * DIRECTIONAL_GATE.
     *
     * NOISE_GATE -- a Wiener-style soft gate  d * d^2/(d^2 + (c*nf)^2): suppresses
     * detail QUADRATICALLY below the noise scale (flat-region noise |d|~nf is cut to
     * ~1/(1+c^2) -- far stronger rejection than a plain soft-threshold, which leaves
     * O(1) noise), while passing supra-noise texture (|d|>>nf -> ~1). c sets the knee in
     * noise-sigmas (c=2 -> detail at the noise level suppressed ~5x). This is what makes
     * texture/noise IMPROVE rather than uniformly scaling noise up.
     *
     * DIRECTIONAL_GATE -- hp^2/(hp^2 + hn^2), where hp=|in-inplane_lp| (in-plane high-
     * pass) and hn=us[i]=|in-normal_lp| (cross-sheet/normal high-pass). Genuine in-plane
     * texture has hn~0 -> gate~1 (passed); a layering edge or the leak from a slightly
     * tilted normal has hn>>hp -> gate~0 (suppressed). This is the second reason the
     * cross-sheet step is NOT amplified (beyond the in-plane steering itself). */
    double cnf2 = (2.0*nf)*(2.0*nf);
    for (size_t i=0;i<n;i++){
        double d  = (double)lp[i];
        double hp = fabs(d);
        double hn = (double)us[i];
        double ngate = (cnf2 > 0) ? (d*d)/(d*d + cnf2) : 1.0;       /* d* below */
        double dgate = (hp+hn > 0) ? (hp*hp)/(hp*hp + hn*hn) : 1.0;
        double gated = d * ngate * dgate;
        out[i] = (float)((double)in[i] + gain*gated);
    }

    free(us);free(t1);free(t2);free(J11);free(J12);free(J13);
    free(J22);free(J23);free(J33);free(lp);
    return 0;
}

/* ---- AUTO-CALIBRATION helpers ---------------------------------------------- */

/* estimate the cross-sheet LAYER spacing (voxels): first non-trivial peak of the
 * mean 1-D autocorrelation along the axis with strongest periodicity. Same idea as
 * diffusion.c's estimate_sheet_spacing (that one is file-static in its TU). */
static double tex_estimate_sheet_spacing(const float *in, int nz, int ny, int nx) {
    int maxlag = 24;
    double best_period = 0.0, best_strength = 0.0;
    for (int axis = 0; axis < 3; axis++) {
        int len = (axis==0)?nz:(axis==1)?ny:nx;
        if (len < 2*maxlag+4) continue;
        double ac[64]; for (int l=0;l<=maxlag && l<64;l++) ac[l]=0.0;
        long lines = 0;
        int s1 = (axis==0)?ny:(axis==1)?nz:nz;
        int s2 = (axis==0)?nx:(axis==1)?nx:ny;
        int step1 = s1>8 ? s1/8 : 1, step2 = s2>8 ? s2/8 : 1;
        for (int a=step1/2; a<s1; a+=step1) for (int b=step2/2; b<s2; b+=step2) {
            double mean=0;
            for (int i=0;i<len;i++){ int z,y,x;
                if(axis==0){z=i;y=a;x=b;} else if(axis==1){z=a;y=i;x=b;} else {z=a;y=b;x=i;}
                mean += in[((size_t)z*ny+y)*nx+x]; }
            mean/=len;
            for (int l=0;l<=maxlag;l++){
                double s=0; int cnt=0;
                for (int i=0;i+l<len;i++){ int z1,y1,x1,z2,y2,x2;
                    if(axis==0){z1=i;y1=a;x1=b; z2=i+l;y2=a;x2=b;}
                    else if(axis==1){z1=a;y1=i;x1=b; z2=a;y2=i+l;x2=b;}
                    else {z1=a;y1=b;x1=i; z2=a;y2=b;x2=i+l;}
                    s += (in[((size_t)z1*ny+y1)*nx+x1]-mean)*(in[((size_t)z2*ny+y2)*nx+x2]-mean);
                    cnt++;
                }
                ac[l]+= (cnt? s/cnt : 0);
            }
            lines++;
        }
        if (!lines) continue;
        for (int l=0;l<=maxlag;l++) ac[l]/=lines;
        double a0 = ac[0]>1e-9?ac[0]:1.0;
        int dipped=0;
        for (int l=2;l<=maxlag;l++){
            double r = ac[l]/a0;
            if (r<0) dipped=1;
            if (dipped && r>ac[l-1]/a0 && r>ac[l+1<=maxlag?l+1:l]/a0 && r>0.05){
                if (r>best_strength){ best_strength=r; best_period=l; }
                break;
            }
        }
    }
    return best_period;
}

/* Estimate the IN-PLANE texture scale (voxels): the in-plane spatial lag at which
 * the autocorrelation of the (mean-subtracted) signal first dips, measured WITHIN the
 * tangent plane. We orient by the GLOBAL dominant sheet normal (mean structure tensor),
 * build two in-plane basis vectors, then walk the mean in-plane autocorrelation along
 * those directions; the lag of the first minimum (~half the dominant in-plane period)
 * is the texture scale, used as the steered low-pass radius so the crackle lands in the
 * detail band. Returns a scale in voxels (clamped [1,4]), or 2.0 if no clear texture. */
static double tex_estimate_inplane_scale(const float *in, int nz, int ny, int nx,
                                         double sigma, double rho) {
    size_t n=(size_t)nz*ny*nx;
    float *us=malloc(sizeof(float)*n), *t1=malloc(sizeof(float)*n), *t2=malloc(sizeof(float)*n);
    float *J11=malloc(sizeof(float)*n),*J12=malloc(sizeof(float)*n),*J13=malloc(sizeof(float)*n);
    float *J22=malloc(sizeof(float)*n),*J23=malloc(sizeof(float)*n),*J33=malloc(sizeof(float)*n);
    if(!us||!t1||!t2||!J11||!J12||!J13||!J22||!J23||!J33){
        free(us);free(t1);free(t2);free(J11);free(J12);free(J13);free(J22);free(J23);free(J33);
        return 2.0;
    }
    tex_gauss_blur(in, us, t1, nz,ny,nx, sigma);
    for (int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        int xm=CLAMP(x-1,0,nx-1),xp=CLAMP(x+1,0,nx-1);
        int ym=CLAMP(y-1,0,ny-1),yp=CLAMP(y+1,0,ny-1);
        int zm=CLAMP(z-1,0,nz-1),zp=CLAMP(z+1,0,nz-1);
        double gx=0.5*(us[IDX(z,y,xp)]-us[IDX(z,y,xm)]);
        double gy=0.5*(us[IDX(z,yp,x)]-us[IDX(z,ym,x)]);
        double gz=0.5*(us[IDX(zp,y,x)]-us[IDX(zm,y,x)]);
        size_t i=IDX(z,y,x);
        J11[i]=(float)(gx*gx); J12[i]=(float)(gx*gy); J13[i]=(float)(gx*gz);
        J22[i]=(float)(gy*gy); J23[i]=(float)(gy*gz); J33[i]=(float)(gz*gz);
    }
    tex_box_gauss(J11,t1,t2,nz,ny,nx,rho); tex_box_gauss(J12,t1,t2,nz,ny,nx,rho);
    tex_box_gauss(J13,t1,t2,nz,ny,nx,rho); tex_box_gauss(J22,t1,t2,nz,ny,nx,rho);
    tex_box_gauss(J23,t1,t2,nz,ny,nx,rho); tex_box_gauss(J33,t1,t2,nz,ny,nx,rho);
    /* global mean structure tensor -> dominant normal -> in-plane basis */
    double m11=0,m12=0,m13=0,m22=0,m23=0,m33=0;
    for (size_t i=0;i<n;i++){ m11+=J11[i];m12+=J12[i];m13+=J13[i];m22+=J22[i];m23+=J23[i];m33+=J33[i]; }
    double inv=1.0/n; m11*=inv;m12*=inv;m13*=inv;m22*=inv;m23*=inv;m33*=inv;
    double lam[3],vec[3][3]; eig3_sym_vec(m11,m12,m13,m22,m23,m33,lam,vec);
    /* eigenvector components are (X,Y,Z) order (tensor built from gx,gy,gz). */
    double t1x=vec[1][0],t1y=vec[1][1],t1z=vec[1][2];
    double t2x=vec[2][0],t2y=vec[2][1],t2z=vec[2][2];

    /* mean in-plane autocorrelation of the signal along the two tangents. Subtract the
     * per-sample local DC (3x3x3 box mean) so the autocorrelation reflects texture, not
     * the bright-sheet pedestal. Find the lag of the first minimum on each tangent. */
    int maxlag = 8;
    double ac1[16]={0}, ac2[16]={0}; long cnt=0;
    int step = (nx>16? nx/16:1); if (step<1) step=1;
    int c0=nz/4, c1=nz-nz/4;
    for (int z=c0; z<c1; z+=step) for (int y=c0; y<c1; y+=step) for (int x=c0; x<c1; x+=step){
        /* local DC */
        double dc=0; int dcn=0;
        for(int dz=-1;dz<=1;dz++)for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
            int zz=CLAMP(z+dz,0,nz-1),yy=CLAMP(y+dy,0,ny-1),xx=CLAMP(x+dx,0,nx-1);
            dc+=in[IDX(zz,yy,xx)]; dcn++;
        }
        dc/=dcn;
        double cv = (double)in[IDX(z,y,x)] - dc;
        for (int l=0;l<=maxlag;l++){
            double a = sample_trilin(in,nz,ny,nx,z+l*t1z,y+l*t1y,x+l*t1x) - dc;
            double b = sample_trilin(in,nz,ny,nx,z+l*t2z,y+l*t2y,x+l*t2x) - dc;
            ac1[l] += cv*a; ac2[l] += cv*b;
        }
        cnt++;
    }
    free(us);free(t1);free(t2);free(J11);free(J12);free(J13);free(J22);free(J23);free(J33);
    if (cnt==0) return 2.0;
    double scale = 0.0; int found=0;
    for (int variant=0; variant<2; variant++){
        double *ac = variant? ac2 : ac1;
        double a0 = ac[0]>1e-9?ac[0]:1.0;
        for (int l=1;l<maxlag;l++){
            double r=ac[l]/a0, rp=ac[l-1]/a0, rn=ac[l+1]/a0;
            if (r<rp && r<=rn && r<0.98){ scale += l; found++; break; }
        }
    }
    if (!found) return 2.0;
    double s = scale/found;
    if (s < 1.0) s = 1.0; if (s > 4.0) s = 4.0;
    return s;
}

/* ---- AUTO-CALIBRATED in-plane texture enhancer (no-knobs vc3d entry point) ----
 * Derives parameters from the data:
 *   sigma         <- presmoothing from measured noise (fy_estimate_noise), [0.6,1.2]
 *   noise_floor   <- the measured noise std (noise_ref) -> cores sub-noise detail
 *   rho           <- ~half the measured layer spacing (integrate within one sheet)
 *   inplane_scale <- the measured in-plane texture scale (autocorrelation in plane)
 *   gain          <- strength {1=gentle,2=normal,3=strong} -> {0.8,1.5,2.5}
 * Returns 0 on success. */
int fy_texture_enhance_auto(const float *in, float *out, int nz, int ny, int nx,
                            int strength) {
    fy_noise_model nm;
    double sigma = 0.8;
    if (fy_estimate_noise(in, nz, ny, nx, 5, 10.0, 0.4, &nm) == 0 && nm.noise_ref > 0) {
        /* presmooth scale from the measured noise correlation; clamp to useful range. */
        sigma = 0.6 + 1.0 * nm.noise_ref;
        if (sigma < 0.6) sigma = 0.6;
        if (sigma > 1.2) sigma = 1.2;
    }
    /* noise_floor: pass the sentinel (<0) so fy_texture_enhance derives the gate from
     * the DETAIL's own robust scale (MAD). The detail-domain noise is the right quantity
     * to gate on, and is measured directly rather than mapped from an absolute intensity-
     * domain noise model (which is corrupted by the bright sheets / layering). */
    double noise_floor = -1.0;
    double spacing = tex_estimate_sheet_spacing(in, nz, ny, nx);
    double rho = (spacing > 1.5) ? 0.5 * spacing : 3.0;
    if (rho < 1.5) rho = 1.5;
    if (rho > 6.0) rho = 6.0;
    double inplane_scale = tex_estimate_inplane_scale(in, nz, ny, nx, sigma, rho);
    double gain = (strength <= 1) ? 0.8 : (strength == 2) ? 1.5 : 2.5;
    return fy_texture_enhance(in, out, nz, ny, nx, sigma, rho, gain, inplane_scale, noise_floor);
}

/* Recommended per-side halo (voxels) for tiled / streaming use:
 *   presmooth (3*sigma) + integration box (3*rho) + steered low-pass reach
 *   (ceil(inplane_scale)) + small safety margin. The filter is single-pass (no
 *   iteration) so there is no n_iters term. */
int fy_texture_enhance_halo(double sigma, double rho, double inplane_scale) {
    if (sigma < 0) sigma = 0.7;
    if (rho   <= 0) rho   = 3.0;
    if (inplane_scale <= 0) inplane_scale = 2.0;
    int h = (int)ceil(3.0*sigma) + 3*(int)ceil(rho) + (int)ceil(inplane_scale) + 2;
    return h;
}

#undef IDX
#undef CLAMP
