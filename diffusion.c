/* diffusion.c -- Coherence-Enhancing Anisotropic Diffusion (Weickert 1999), 3D.
 *
 * Purpose: clean papyrus SHEETS for tracing/unwrapping. Smooth ALONG the sheet
 * surface (denoise, make sheets continuous) but NOT across it -- so the THIN DARK
 * GAPS between two touching layers are PRESERVED (never merge adjacent sheets).
 * A plain Gaussian/isotropic blur would fill those gaps; CED keeps them because
 * diffusion across the sheet normal is suppressed.
 *
 * Method (Weickert, "Coherence-Enhancing Diffusion Filtering", IJCV 1999):
 *   1. STRUCTURE TENSOR  J_rho = G_rho * ( grad u_sigma (x) grad u_sigma^T ):
 *      - presmooth u with a Gaussian G_sigma (noise scale, sigma~0.5-1 vox),
 *      - gradient via central differences,
 *      - form the 6 unique components of the 3x3 outer product per voxel,
 *      - smooth each component with a Gaussian G_rho (integration / sheet-coherence
 *        scale, rho~2-4 vox) -- here a 3-pass box filter approximates the Gaussian
 *        (fast, separable, accurate to ~Gaussian by CLT).
 *   2. EIGEN-ANALYSIS of the symmetric J per voxel -> eigenvalues mu1>=mu2>=mu3 and
 *      orthonormal eigenvectors. For a SHEET (planar): grad u points across the sheet
 *      (the sheet normal) so ONE eigenvalue (mu1) is large and TWO (mu2,mu3) are small
 *      (in-plane directions, within the sheet).
 *   3. DIFFUSION TENSOR D: SAME eigenvectors, REMAPPED eigenvalues. Diffuse STRONG
 *      along the two small-mu (in-plane) directions, WEAK along the large-mu (normal)
 *      direction. Weickert's coherence-enhancing diffusivities:
 *        lam_normal = alpha                              (across the sheet: tiny)
 *        lam_plane  = alpha + (1-alpha)*exp(-C/(kappa^2)) (along the sheet: ->1 where
 *                     coherent)   with coherence kappa = (mu1-mu3)^2 ("planarity"
 *                     contrast: large on a sheet, ~0 in isotropic noise).
 *      So in a coherent sheet, lam_plane->1 (smooth a lot in-plane) while lam_normal
 *      stays = alpha (barely smooth across) -> sheets clean, gaps survive. In flat
 *      noise (kappa~0) all three ->alpha -> mild isotropic smoothing.
 *   4. EVOLVE  u_{t+1} = u_t + tau * div(D grad u), explicit Euler, stable tau
 *      (<= ~0.12 for 3D), iterate n_iters (3-15). div(D grad u) uses a central-
 *      difference flux form (Weickert's standard discretization): assemble the flux
 *      vector F = D grad u at each voxel (central-difference gradient, full 3x3 D),
 *      then take its divergence by central differences. Reflecting (Neumann) bounds.
 *
 * STREAMING / TILING (for vc3d): the whole filter is LOCAL. Per iteration the
 * stencil touches a 1-voxel neighbourhood, and the structure tensor adds the
 * presmooth + gradient + box halo. Total halo to feed for a tile is approximately
 *   halo = ceil(3*sigma) + 3*ceil(rho) + n_iters + 2   (voxels, per side).
 * (The +n_iters is the diffusion stencil's reach over the iterations; the box term
 * is the 3-pass box radius ~ rho.) See fy_coherence_diffusion_halo().
 *
 * Pure C, libc+libm only, no SIMD intrinsics, vectorizable loops. Row-major,
 * x-fastest: idx = (z*ny + y)*nx + x. float data, returns int (0=ok).
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define IDX(z,y,x) (((size_t)(z)*ny+(y))*nx+(x))
#define CLAMP(v,lo,hi) ((v)<(lo)?(lo):((v)>(hi)?(hi):(v)))

/* ---- separable Gaussian (presmooth, noise scale). Reflecting boundary. ---- */
static void gauss_blur(const float *in, float *out, float *tmp,
                       int nz, int ny, int nx, double sigma) {
    if (sigma <= 0) { memcpy(out, in, sizeof(float)*(size_t)nz*ny*nx); return; }
    int r = (int)ceil(3.0 * sigma); if (r < 1) r = 1;
    int w = 2*r+1;
    double *k = malloc(sizeof(double)*w); double sum=0;
    for (int i=-r;i<=r;i++){ k[i+r]=exp(-0.5*i*i/(sigma*sigma)); sum+=k[i+r]; }
    for (int i=0;i<w;i++) k[i]/=sum;
    float *a=out, *b=tmp;
    /* X */
    for (int z=0;z<nz;z++) for (int y=0;y<ny;y++){ size_t row=((size_t)z*ny+y)*nx;
        for (int x=0;x<nx;x++){ double s=0; for(int t=-r;t<=r;t++){int xx=CLAMP(x+t,0,nx-1);s+=k[t+r]*in[row+xx];} a[row+x]=(float)s; } }
    /* Y */
    for (int z=0;z<nz;z++) for (int x=0;x<nx;x++)
        for (int y=0;y<ny;y++){ double s=0; for(int t=-r;t<=r;t++){int yy=CLAMP(y+t,0,ny-1);s+=k[t+r]*a[IDX(z,yy,x)];} b[IDX(z,y,x)]=(float)s; }
    /* Z */
    for (int y=0;y<ny;y++) for (int x=0;x<nx;x++)
        for (int z=0;z<nz;z++){ double s=0; for(int t=-r;t<=r;t++){int zz=CLAMP(z+t,0,nz-1);s+=k[t+r]*b[IDX(zz,y,x)];} a[IDX(z,y,x)]=(float)s; }
    if (a != out) memcpy(out, a, sizeof(float)*(size_t)nz*ny*nx);
    free(k);
}

/* ---- separable box filter of radius r, reflecting boundary (one pass) ---- */
static void box_pass(const float *in, float *out, float *tmp,
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

/* 3-pass box ~ Gaussian G_rho (smooth the structure-tensor components). */
static void box_gauss(float *buf, float *t1, float *t2, int nz,int ny,int nx, double rho){
    if (rho <= 0) return;
    int r = (int)floor(rho); if (r < 1) r = 1;   /* ~ matched variance for 3 passes */
    box_pass(buf, t1, t2, nz,ny,nx, r);
    box_pass(t1, buf, t2, nz,ny,nx, r);
    box_pass(buf, t1, t2, nz,ny,nx, r);
    memcpy(buf, t1, sizeof(float)*(size_t)nz*ny*nx);
}

/* ---- symmetric 3x3 eigen-decomposition: eigenvalues sorted DESCENDING (l1>=l2>=l3)
 * with orthonormal eigenvectors. Eigenvalue part mirrors eig3_sym in sheetness.c;
 * eigenvectors obtained from the (A - lambda I) null space via cross products. ---- */
static void eig3_sym_vec(double a,double b,double c,double d,double e,double f,
                         double lam[3], double vec[3][3]) {
    /* matrix M = [[a,b,c],[b,d,e],[c,e,f]] */
    double p1 = b*b + c*c + e*e;
    if (p1 == 0.0) {
        /* already diagonal */
        double v[3]={a,d,f};
        int idx[3]={0,1,2};
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
    /* sort DESCENDING l1>=l2>=l3 */
    double ev[3]={eig1,eig2,eig3};
    for(int i=0;i<2;i++)for(int j=0;j<2-i;j++) if(ev[j]<ev[j+1]){double t=ev[j];ev[j]=ev[j+1];ev[j+1]=t;}
    lam[0]=ev[0]; lam[1]=ev[1]; lam[2]=ev[2];

    /* eigenvector for eigenvalue L: a vector in the null space of (M - L*I).
     * Rows of (M-LI): r0=(a-L,b,c), r1=(b,d-L,e), r2=(c,e,f-L). The null space is
     * spanned by the cross product of any two independent rows; pick the pair whose
     * cross product has the largest norm (most robust). */
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
        if (bestn <= 1e-30) {
            /* degenerate (repeated eigenvalue): fall back to a canonical axis,
             * orthogonalized against earlier eigenvectors below. */
            best[0]=(k==0);best[1]=(k==1);best[2]=(k==2); bestn=1;
        }
        double inv=1.0/sqrt(best[0]*best[0]+best[1]*best[1]+best[2]*best[2]);
        vec[k][0]=best[0]*inv; vec[k][1]=best[1]*inv; vec[k][2]=best[2]*inv;
    }
    /* Gram-Schmidt to guarantee orthonormality despite repeated-eigenvalue cases. */
    for (int k=1;k<3;k++){
        for (int j=0;j<k;j++){
            double dot=vec[k][0]*vec[j][0]+vec[k][1]*vec[j][1]+vec[k][2]*vec[j][2];
            vec[k][0]-=dot*vec[j][0]; vec[k][1]-=dot*vec[j][1]; vec[k][2]-=dot*vec[j][2];
        }
        double nn=vec[k][0]*vec[k][0]+vec[k][1]*vec[k][1]+vec[k][2]*vec[k][2];
        if (nn<1e-20){ /* pick any orthogonal axis */
            double ax[3]={1,0,0};
            if (fabs(vec[0][0])>0.9){ax[0]=0;ax[1]=1;}
            /* project out vec[0..k-1] */
            for (int j=0;j<k;j++){ double dot=ax[0]*vec[j][0]+ax[1]*vec[j][1]+ax[2]*vec[j][2];
                ax[0]-=dot*vec[j][0];ax[1]-=dot*vec[j][1];ax[2]-=dot*vec[j][2]; }
            nn=ax[0]*ax[0]+ax[1]*ax[1]+ax[2]*ax[2];
            vec[k][0]=ax[0];vec[k][1]=ax[1];vec[k][2]=ax[2];
        }
        double inv=1.0/sqrt(nn); vec[k][0]*=inv;vec[k][1]*=inv;vec[k][2]*=inv;
    }
}

/* Coherence-Enhancing Diffusion (Weickert), 3D.
 *   in,out     : nz*ny*nx float volumes (out may differ from in; in is not modified)
 *   sigma      : noise-scale presmoothing for the gradient (~0.5-1 vox)
 *   rho        : integration scale = sheet-coherence scale (~2-4 vox)
 *   tau        : explicit time step (<= ~0.12 for 3D stability)
 *   n_iters    : number of explicit iterations (3-15)
 *   coherence_alpha : Weickert's alpha (base diffusivity along the normal, 0<alpha<1;
 *                     ~0.001-0.01). Smaller alpha = harder gap preservation.
 * Returns 0 on success, 1 on allocation failure. */
int fy_coherence_diffusion(const float *in, float *out, int nz, int ny, int nx,
                           double sigma, double rho, double tau, int n_iters,
                           double coherence_alpha) {
    if (nz<1||ny<1||nx<1) return 1;
    if (sigma < 0) sigma = 0.7;
    if (rho   <= 0) rho   = 3.0;
    if (tau   <= 0) tau   = 0.10;
    if (n_iters < 1) n_iters = 5;
    double alpha = coherence_alpha; if (alpha <= 0) alpha = 0.001; if (alpha >= 1) alpha = 0.5;

    size_t n=(size_t)nz*ny*nx;
    float *u   = malloc(sizeof(float)*n);   /* evolving volume            */
    float *us  = malloc(sizeof(float)*n);   /* presmoothed (for gradient) */
    float *t1  = malloc(sizeof(float)*n);
    float *t2  = malloc(sizeof(float)*n);
    /* 6 unique structure-tensor / diffusion-tensor components */
    float *J11=malloc(sizeof(float)*n),*J12=malloc(sizeof(float)*n),*J13=malloc(sizeof(float)*n);
    float *J22=malloc(sizeof(float)*n),*J23=malloc(sizeof(float)*n),*J33=malloc(sizeof(float)*n);
    /* flux components Fx,Fy,Fz = D grad u */
    float *Fx=malloc(sizeof(float)*n),*Fy=malloc(sizeof(float)*n),*Fz=malloc(sizeof(float)*n);
    if(!u||!us||!t1||!t2||!J11||!J12||!J13||!J22||!J23||!J33||!Fx||!Fy||!Fz){
        free(u);free(us);free(t1);free(t2);free(J11);free(J12);free(J13);
        free(J22);free(J23);free(J33);free(Fx);free(Fy);free(Fz); return 1;
    }
    memcpy(u, in, sizeof(float)*n);

    /* coherence normalization: data-adaptive scale for the (mu1-mu3)^2 measure so
     * one alpha works across intensity ranges. Set after the first tensor build. */
    double Cnorm = 0.0;

    for (int it=0; it<n_iters; it++) {
        /* (1) presmooth at noise scale */
        gauss_blur(u, us, t1, nz,ny,nx, sigma);

        /* (1b) gradient (central diff, reflecting) -> outer-product components */
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

        /* (1c) smooth each component with G_rho (3-pass box) -> structure tensor J */
        box_gauss(J11,t1,t2,nz,ny,nx,rho); box_gauss(J12,t1,t2,nz,ny,nx,rho);
        box_gauss(J13,t1,t2,nz,ny,nx,rho); box_gauss(J22,t1,t2,nz,ny,nx,rho);
        box_gauss(J23,t1,t2,nz,ny,nx,rho); box_gauss(J33,t1,t2,nz,ny,nx,rho);

        /* coherence normalization once: median-ish scale via mean of (mu1-mu3)^2.
         * Use the mean over voxels of the planarity contrast as kappa^2 reference. */
        if (it==0) {
            double acc=0; size_t cnt=0;
            for (size_t i=0;i<n;i++){
                double lam[3],vec[3][3];
                eig3_sym_vec(J11[i],J12[i],J13[i],J22[i],J23[i],J33[i],lam,vec);
                double k2=(lam[0]-lam[2])*(lam[0]-lam[2]);
                acc+=k2; cnt++;
            }
            double mean=cnt?acc/cnt:0;
            Cnorm = mean>0 ? mean : 1.0;   /* C in exp(-C/(kappa^2)) scaled by this */
        }

        /* (2,3) per-voxel eigen-analysis -> remapped diffusion tensor D (6 comps),
         * stored back into J11..J33 in place. */
        for (size_t i=0;i<n;i++){
            double lam[3],vec[3][3];
            eig3_sym_vec(J11[i],J12[i],J13[i],J22[i],J23[i],J33[i],lam,vec);
            /* lam[0]>=lam[1]>=lam[2]; vec[0]=normal(across sheet), vec[1],vec[2]=in-plane */
            double kappa2=(lam[0]-lam[2])*(lam[0]-lam[2]);   /* planarity contrast */
            /* Weickert coherence-enhancing diffusivities */
            double lam_normal = alpha;                       /* across the sheet (vec[0]) */
            double cval = (kappa2>0) ? exp(-Cnorm/kappa2) : 0.0;
            double lam_plane  = alpha + (1.0-alpha)*cval;    /* within the sheet */
            /* Assign: vec[0] gets lam_normal (weak), vec[1],vec[2] get lam_plane (strong) */
            double d0=lam_normal, d1=lam_plane, d2=lam_plane;
            /* D = sum_k d_k * v_k v_k^T  (6 unique comps) */
            double D11=0,D12=0,D13=0,D22=0,D23=0,D33=0;
            double dk[3]={d0,d1,d2};
            for (int k=0;k<3;k++){
                double vx=vec[k][0],vy=vec[k][1],vz=vec[k][2],w=dk[k];
                D11+=w*vx*vx; D12+=w*vx*vy; D13+=w*vx*vz;
                D22+=w*vy*vy; D23+=w*vy*vz; D33+=w*vz*vz;
            }
            J11[i]=(float)D11; J12[i]=(float)D12; J13[i]=(float)D13;
            J22[i]=(float)D22; J23[i]=(float)D23; J33[i]=(float)D33;
        }

        /* (4) flux F = D grad u (central-difference gradient of the CURRENT u) */
        for (int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
            int xm=CLAMP(x-1,0,nx-1),xp=CLAMP(x+1,0,nx-1);
            int ym=CLAMP(y-1,0,ny-1),yp=CLAMP(y+1,0,ny-1);
            int zm=CLAMP(z-1,0,nz-1),zp=CLAMP(z+1,0,nz-1);
            double gx=0.5*(u[IDX(z,y,xp)]-u[IDX(z,y,xm)]);
            double gy=0.5*(u[IDX(z,yp,x)]-u[IDX(z,ym,x)]);
            double gz=0.5*(u[IDX(zp,y,x)]-u[IDX(zm,y,x)]);
            size_t i=IDX(z,y,x);
            Fx[i]=(float)(J11[i]*gx + J12[i]*gy + J13[i]*gz);
            Fy[i]=(float)(J12[i]*gx + J22[i]*gy + J23[i]*gz);
            Fz[i]=(float)(J13[i]*gx + J23[i]*gy + J33[i]*gz);
        }

        /* (4b) div(F) by central differences -> explicit update u += tau*div(F).
         * Reflecting boundary: at borders the one-sided neighbour clamps, giving
         * zero flux through the wall (Neumann), so total intensity is conserved. */
        for (int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
            int xm=CLAMP(x-1,0,nx-1),xp=CLAMP(x+1,0,nx-1);
            int ym=CLAMP(y-1,0,ny-1),yp=CLAMP(y+1,0,ny-1);
            int zm=CLAMP(z-1,0,nz-1),zp=CLAMP(z+1,0,nz-1);
            double div = 0.5*(Fx[IDX(z,y,xp)]-Fx[IDX(z,y,xm)])
                       + 0.5*(Fy[IDX(z,yp,x)]-Fy[IDX(z,ym,x)])
                       + 0.5*(Fz[IDX(zp,y,x)]-Fz[IDX(zm,y,x)]);
            t1[IDX(z,y,x)] = (float)(u[IDX(z,y,x)] + tau*div);
        }
        memcpy(u, t1, sizeof(float)*n);
    }

    memcpy(out, u, sizeof(float)*n);
    free(u);free(us);free(t1);free(t2);free(J11);free(J12);free(J13);
    free(J22);free(J23);free(J33);free(Fx);free(Fy);free(Fz);
    return 0;
}

/* Recommended per-side halo (voxels) for tiled / streaming use. See header notes:
 * presmooth (3*sigma) + integration box (3*rho) + diffusion stencil reach (n_iters)
 * + a small safety margin. */
int fy_coherence_diffusion_halo(double sigma, double rho, int n_iters) {
    if (sigma < 0) sigma = 0.7;
    if (rho   <= 0) rho   = 3.0;
    if (n_iters < 1) n_iters = 5;
    int h = (int)ceil(3.0*sigma) + 3*(int)ceil(rho) + n_iters + 2;
    return h;
}

#undef IDX
#undef CLAMP
