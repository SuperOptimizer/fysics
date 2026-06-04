/* sheetness.c -- Frangi-style plate/sheet detector for papyrus sheets.
 *
 * Papyrus is a thin wound SHEET; this responds to plate-like local geometry,
 * helping segmentation/unwrapping separate adjacent sheets (even when they touch
 * and have similar intensity -- geometry differs). NOT for ink (it emphasizes
 * sheet structure over the faint ink texture).
 *
 * Method (Frangi 1998, sheet/plate variant): at scale sigma, compute the 3x3
 * Hessian (second derivatives via Gaussian-derivative convolution), get its three
 * eigenvalues |l1|<=|l2|<=|l3|. A bright sheet on dark background has l3 strongly
 * negative, l1,l2 ~ 0. Sheetness:
 *   R_a  = |l2| / |l3|            (plate-vs-blob; small for sheet, ~1 for blob)
 *   R_b  = |l1| / sqrt(|l2 l3|)   (deviation from sheet; small for sheet & plate)
 *   S    = sqrt(l1^2+l2^2+l3^2)   (structureness; low in noise/flat)
 *   sheetness = exp(-Ra^2/2b^2) * exp(-Rb^2/2a^2) * (1-exp(-S^2/2c^2))
 * (both ratio terms PEAK at R=0 so a perfect sheet scores ~1; only the structureness
 *  gate uses 1-exp to suppress flat/noise regions.)
 * Tuning: SIGMA must match sheet thickness in voxels (the key knob); a,b shape
 * sensitivity (defaults 0.5); c is the noise/structure threshold (scale to data).
 * Detects BRIGHT sheets (set bright=1) -- papyrus is bright on dark air.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* separable Gaussian blur at sigma (used to build the smoothed volume the Hessian
 * is computed on -- Hessian-of-Gaussian = finite-diff of the blurred volume) */
static void gauss_blur(const float *in, float *out, float *tmp,
                       int nz, int ny, int nx, double sigma) {
    int r = (int)ceil(3.0 * sigma); if (r < 1) r = 1;
    int w = 2*r+1;
    double *k = malloc(sizeof(double)*w); double sum=0;
    for (int i=-r;i<=r;i++){ k[i+r]=exp(-0.5*i*i/(sigma*sigma)); sum+=k[i+r]; }
    for (int i=0;i<w;i++) k[i]/=sum;
    size_t n=(size_t)nz*ny*nx;
    const float *src=in; float *a=out, *b=tmp;
    /* X */
    for (int z=0;z<nz;z++) for (int y=0;y<ny;y++){ size_t row=((size_t)z*ny+y)*nx;
        for (int x=0;x<nx;x++){ double s=0; for(int t=-r;t<=r;t++){int xx=x+t;if(xx<0)xx=0;if(xx>=nx)xx=nx-1;s+=k[t+r]*src[row+xx];} a[row+x]=(float)s; } }
    /* Y */
    for (int z=0;z<nz;z++) for (int x=0;x<nx;x++){
        for (int y=0;y<ny;y++){ double s=0; for(int t=-r;t<=r;t++){int yy=y+t;if(yy<0)yy=0;if(yy>=ny)yy=ny-1;s+=k[t+r]*a[((size_t)z*ny+yy)*nx+x];} b[((size_t)z*ny+y)*nx+x]=(float)s; } }
    /* Z */
    for (int y=0;y<ny;y++) for (int x=0;x<nx;x++){
        for (int z=0;z<nz;z++){ double s=0; for(int t=-r;t<=r;t++){int zz=z+t;if(zz<0)zz=0;if(zz>=nz)zz=nz-1;s+=k[t+r]*b[((size_t)zz*ny+y)*nx+x];} a[((size_t)z*ny+y)*nx+x]=(float)s; } }
    memcpy(out, a, sizeof(float)*n);
    free(k);
}

/* symmetric 3x3 eigenvalues (analytic, for a real symmetric matrix) */
static void eig3_sym(double a,double b,double c,double d,double e,double f,
                     double *l1,double *l2,double *l3) {
    /* matrix [[a,b,c],[b,d,e],[c,e,f]] */
    double p1 = b*b + c*c + e*e;
    if (p1 == 0) { /* diagonal */
        double v[3]={a,d,f};
        for(int i=0;i<2;i++)for(int j=0;j<2-i;j++) if(fabs(v[j])>fabs(v[j+1])){double t=v[j];v[j]=v[j+1];v[j+1]=t;}
        *l1=v[0];*l2=v[1];*l3=v[2]; return;
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
    /* sort by ABSOLUTE value: |l1|<=|l2|<=|l3| */
    double v[3]={eig1,eig2,eig3};
    for(int i=0;i<2;i++)for(int j=0;j<2-i;j++) if(fabs(v[j])>fabs(v[j+1])){double t=v[j];v[j]=v[j+1];v[j+1]=t;}
    *l1=v[0];*l2=v[1];*l3=v[2];
}

/* Sheetness at a single scale. bright=1 detects bright sheets (papyrus on dark air).
 * alpha,beta plate/blob shape (0.5 default), c structureness (auto if <=0). */
int fy_sheetness(const float *in, float *out, int nz, int ny, int nx,
                 double sigma, double alpha, double beta, double c, int bright) {
    if (sigma <= 0) sigma = 2.0;
    if (alpha <= 0) alpha = 0.5;
    if (beta <= 0) beta = 0.5;
    size_t n=(size_t)nz*ny*nx;
    float *g=malloc(sizeof(float)*n), *tmp=malloc(sizeof(float)*n);
    if(!g||!tmp){free(g);free(tmp);return 1;}
    gauss_blur(in,g,tmp,nz,ny,nx,sigma);   /* Hessian-of-Gaussian via FD of blurred */

    /* auto structureness c: half the max Hessian magnitude (scale to data) */
    double a2=2*alpha*alpha, b2=2*beta*beta;
    /* first pass to estimate c if needed */
    double maxS=0;
    #define IDX(z,y,x) (((size_t)(z)*ny+(y))*nx+(x))
    #define CL(v,lo,hi) ((v)<(lo)?(lo):((v)>(hi)?(hi):(v)))
    for (int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        int zm=CL(z-1,0,nz-1),zp=CL(z+1,0,nz-1),ym=CL(y-1,0,ny-1),yp=CL(y+1,0,ny-1),xm=CL(x-1,0,nx-1),xp=CL(x+1,0,nx-1);
        double gc=g[IDX(z,y,x)];
        double hxx=g[IDX(z,y,xp)]-2*gc+g[IDX(z,y,xm)];
        double hyy=g[IDX(z,yp,x)]-2*gc+g[IDX(z,ym,x)];
        double hzz=g[IDX(zp,y,x)]-2*gc+g[IDX(zm,y,x)];
        double S=sqrt(hxx*hxx+hyy*hyy+hzz*hzz);
        if(S>maxS)maxS=S;
    }
    if (c <= 0) c = 0.5*maxS; if (c<=0) c=1e-6;
    double c2=2*c*c;

    for (int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        int zm=CL(z-1,0,nz-1),zp=CL(z+1,0,nz-1),ym=CL(y-1,0,ny-1),yp=CL(y+1,0,ny-1),xm=CL(x-1,0,nx-1),xp=CL(x+1,0,nx-1);
        double gc=g[IDX(z,y,x)];
        double hxx=g[IDX(z,y,xp)]-2*gc+g[IDX(z,y,xm)];
        double hyy=g[IDX(z,yp,x)]-2*gc+g[IDX(z,ym,x)];
        double hzz=g[IDX(zp,y,x)]-2*gc+g[IDX(zm,y,x)];
        double hxy=(g[IDX(z,yp,xp)]-g[IDX(z,yp,xm)]-g[IDX(z,ym,xp)]+g[IDX(z,ym,xm)])*0.25;
        double hxz=(g[IDX(zp,y,xp)]-g[IDX(zp,y,xm)]-g[IDX(zm,y,xp)]+g[IDX(zm,y,xm)])*0.25;
        double hyz=(g[IDX(zp,yp,x)]-g[IDX(zp,ym,x)]-g[IDX(zm,yp,x)]+g[IDX(zm,ym,x)])*0.25;
        double l1,l2,l3; eig3_sym(hxx,hxy,hxz,hyy,hyz,hzz,&l1,&l2,&l3);
        double v=0.0;
        /* bright sheet: largest-magnitude eigenvalue l3 must be negative */
        int ok = bright ? (l3 < 0) : (l3 > 0);
        if (ok && fabs(l3) > 1e-12) {
            /* For an ideal sheet l1~l2~0, l3<<0:
             *   Ra = |l2|/|l3|       -> 0 for a sheet, ->1 for a blob (plate- vs-blob)
             *   Rb = |l1|/sqrt(|l2 l3|) -> 0 for sheet & plate, ->1 for blob (deviation
             *        from a line/tube)  -- here we only need the plate selector
             *   S  = Frobenius norm    -> small in flat/noise, large on real structure
             * A sheet wants SMALL Ra and SMALL Rb (both -> exp(-R^2/.) peaks at 0) and
             * LARGE S (the (1-exp(-S^2/.)) structureness gate). The earlier code used
             * (1-exp) on the blob term, which zeroed a perfect sheet (Rb=0). */
            double Ra = fabs(l2)/(fabs(l3)+1e-12);          /* plate-vs-blob: small=sheet */
            double Rb = fabs(l1)/(sqrt(fabs(l2*l3))+1e-12); /* deviation from sheet */
            double S  = sqrt(l1*l1+l2*l2+l3*l3);
            v = exp(-Ra*Ra/b2) * exp(-Rb*Rb/a2) * (1.0-exp(-S*S/c2));
        }
        out[IDX(z,y,x)] = (float)v;
    }
    #undef IDX
    #undef CL
    free(g); free(tmp);
    return 0;
}

/* Multi-scale sheetness: max response over several sigmas (catches sheets of
 * varying thickness). sigmas[ns] in voxels. */
int fy_sheetness_multiscale(const float *in, float *out, int nz, int ny, int nx,
                            const double *sigmas, int ns, double alpha, double beta,
                            double c, int bright) {
    size_t n=(size_t)nz*ny*nx;
    float *s=malloc(sizeof(float)*n);
    if(!s) return 1;
    memset(out,0,sizeof(float)*n);
    for (int i=0;i<ns;i++){
        if (fy_sheetness(in,s,nz,ny,nx,sigmas[i],alpha,beta,c,bright)) { free(s); return 1; }
        for (size_t j=0;j<n;j++) if (s[j]>out[j]) out[j]=s[j];
    }
    free(s);
    return 0;
}
