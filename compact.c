/* compact.c -- anti-aliased downsampling + data-driven safe-factor recommendation.
 *
 * Fine (~1.1um) scroll volumes are ~2x oversampled: their resolved detail fits a
 * coarser grid, so an anti-aliased downsample is lossless of real detail and saves
 * ~8x storage. Coarser volumes are critically sampled -- not compactible. The safe
 * factor is measured per-volume from the worst-case textured region (oversampling
 * varies spatially). Deblur (if any) must happen at full res BEFORE downsampling.
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* separable Gaussian blur (reflect edges) -- anti-alias before decimation */
static void gauss3(const float *in, float *out, float *tmp,
                   int nz, int ny, int nx, double sigma) {
    int r = (int)ceil(3.0 * sigma); if (r < 1) r = 1;
    int w = 2*r+1;
    double *k = malloc(sizeof(double)*w), sum = 0;
    for (int i=-r;i<=r;i++){ k[i+r]=exp(-0.5*i*i/(sigma*sigma)); sum+=k[i+r]; }
    for (int i=0;i<w;i++) k[i]/=sum;
    size_t n=(size_t)nz*ny*nx;
    /* X -> out */
    for (int z=0;z<nz;z++) for (int y=0;y<ny;y++){ size_t row=((size_t)z*ny+y)*nx;
        for (int x=0;x<nx;x++){ double s=0; for(int t=-r;t<=r;t++){int xx=x+t; if(xx<0)xx=-xx-1; if(xx>=nx)xx=2*nx-xx-1; s+=k[t+r]*in[row+xx];} out[row+x]=(float)s; } }
    /* Y -> tmp */
    for (int z=0;z<nz;z++) for (int x=0;x<nx;x++)
        for (int y=0;y<ny;y++){ double s=0; for(int t=-r;t<=r;t++){int yy=y+t; if(yy<0)yy=-yy-1; if(yy>=ny)yy=2*ny-yy-1; s+=k[t+r]*out[((size_t)z*ny+yy)*nx+x];} tmp[((size_t)z*ny+y)*nx+x]=(float)s; }
    /* Z -> out */
    for (int y=0;y<ny;y++) for (int x=0;x<nx;x++)
        for (int z=0;z<nz;z++){ double s=0; for(int t=-r;t<=r;t++){int zz=z+t; if(zz<0)zz=-zz-1; if(zz>=nz)zz=2*nz-zz-1; s+=k[t+r]*tmp[((size_t)zz*ny+y)*nx+x];} out[((size_t)z*ny+y)*nx+x]=(float)s; }
    (void)n; free(k);
}

/* trilinear sample of a volume at fractional coord (z,y,x), clamped */
static float sample(const float *v, int nz, int ny, int nx, double z, double y, double x) {
    if (z<0)z=0; if (z>nz-1)z=nz-1; if (y<0)y=0; if (y>ny-1)y=ny-1; if (x<0)x=0; if (x>nx-1)x=nx-1;
    int z0=(int)z, y0=(int)y, x0=(int)x;
    int z1=z0+1<nz?z0+1:z0, y1=y0+1<ny?y0+1:y0, x1=x0+1<nx?x0+1:x0;
    double fz=z-z0, fy=y-y0, fx=x-x0;
    #define V(zz,yy,xx) v[((size_t)(zz)*ny+(yy))*nx+(xx)]
    double c00=V(z0,y0,x0)*(1-fx)+V(z0,y0,x1)*fx;
    double c01=V(z0,y1,x0)*(1-fx)+V(z0,y1,x1)*fx;
    double c10=V(z1,y0,x0)*(1-fx)+V(z1,y0,x1)*fx;
    double c11=V(z1,y1,x0)*(1-fx)+V(z1,y1,x1)*fx;
    #undef V
    double c0=c00*(1-fy)+c01*fy, c1=c10*(1-fy)+c11*fy;
    return (float)(c0*(1-fz)+c1*fz);
}

int fy_downsample(const float *in, float *out, int nz, int ny, int nx,
                  double factor, int *onz, int *ony, int *onx) {
    if (factor <= 1.0) {  /* no shrink: copy */
        memcpy(out, in, sizeof(float)*(size_t)nz*ny*nx);
        if(onz)*onz=nz; if(ony)*ony=ny; if(onx)*onx=nx; return 0;
    }
    int dz=(int)ceil(nz/factor), dy=(int)ceil(ny/factor), dx=(int)ceil(nx/factor);
    if (dz<1)dz=1; if(dy<1)dy=1; if(dx<1)dx=1;
    size_t n=(size_t)nz*ny*nx;
    float *blur=malloc(sizeof(float)*n), *tmp=malloc(sizeof(float)*n);
    if(!blur||!tmp){ free(blur); free(tmp); return 1; }
    gauss3(in, blur, tmp, nz, ny, nx, 0.5*factor);  /* anti-alias */
    for (int z=0;z<dz;z++) for (int y=0;y<dy;y++) for (int x=0;x<dx;x++)
        out[((size_t)z*dy+y)*dx+x] = sample(blur, nz,ny,nx, z*factor, y*factor, x*factor);
    if(onz)*onz=dz; if(ony)*ony=dy; if(onx)*onx=dx;
    free(blur); free(tmp);
    return 0;
}

/* round-trip L2 error of a down+up cycle, computed only over TEXTURED blocks
 * (std > thr). Returns the WORST (max) per-block relative error. */
static double roundtrip_textured_err(const float *in, int nz, int ny, int nx,
                                     double factor) {
    int dz,dy,dx;
    size_t n=(size_t)nz*ny*nx;
    float *small=malloc(sizeof(float)*n), *back=malloc(sizeof(float)*n);
    if(!small||!back){ free(small); free(back); return 1.0; }
    fy_downsample(in, small, nz, ny, nx, factor, &dz, &dy, &dx);
    /* upsample back to full size by trilinear */
    for (int z=0;z<nz;z++) for (int y=0;y<ny;y++) for (int x=0;x<nx;x++)
        back[((size_t)z*ny+y)*nx+x] = sample(small, dz,dy,dx, z/factor, y/factor, x/factor);
    /* per-block worst relative error over textured blocks */
    int bs=16; double worst=0; int found=0;
    for (int z=0;z+bs<=nz;z+=bs) for (int y=0;y+bs<=ny;y+=bs) for (int x=0;x+bs<=nx;x+=bs) {
        double mean=0,m2=0; int c=0;
        double num=0,den=0;
        for (int dzz=0;dzz<bs;dzz++) for (int dyy=0;dyy<bs;dyy++) for (int dxx=0;dxx<bs;dxx++) {
            size_t idx=((size_t)(z+dzz)*ny+(y+dyy))*nx+(x+dxx);
            double a=in[idx], b=back[idx];
            mean+=a; m2+=a*a; c++;
            num+=(a-b)*(a-b); den+=a*a;
        }
        double var=m2/c-(mean/c)*(mean/c);
        double std=var>0?sqrt(var):0;
        if (std < 0.05) continue;           /* skip flat/air blocks (in [0,1] units) */
        found=1;
        double rel = den>0 ? sqrt(num/den) : 0;
        if (rel>worst) worst=rel;
    }
    free(small); free(back);
    return found ? worst : 0.0;   /* no textured blocks -> nothing to lose */
}

double fy_recommend_downsample(const float *in, int nz, int ny, int nx,
                               double err_budget) {
    if (err_budget <= 0) err_budget = 0.03;
    /* try larger factors first; return the largest under budget */
    const double cand[] = {2.0, 1.75, 1.5, 1.25};
    for (int i = 0; i < 4; i++) {
        double e = roundtrip_textured_err(in, nz, ny, nx, cand[i]);
        if (e <= err_budget) return cand[i];
    }
    return 1.0;  /* critically sampled: no safe shrink */
}
