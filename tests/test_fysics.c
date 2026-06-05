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


static void test_nlm_denoises(void) {
    int nz=16,ny=24,nx=24,n=nz*ny*nx;
    float *clean=malloc(4*n),*noisy=malloc(4*n),*out=malloc(4*n);
    /* smooth structure + noise */
    unsigned s=12345;
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        double v=0.5+0.4*sin(x*0.4)*cos(y*0.3);
        clean[(z*ny+y)*nx+x]=(float)v;
        s=s*1103515245u+12345u; double nse=((s>>16)&0x7fff)/32768.0-0.5;
        noisy[(z*ny+y)*nx+x]=(float)(v+0.15*nse);
    }
    fy_nlm_denoise(noisy,out,nz,ny,nx,0.15,4,1);
    /* denoised should be closer to clean than noisy was */
    double en=0,eo=0;
    for(int i=0;i<n;i++){double a=noisy[i]-clean[i],b=out[i]-clean[i];en+=a*a;eo+=b*b;}
    CHECK(eo<en, "NLM reduces error vs clean (denoises)");
    printf("      (noisy MSE=%.5f -> denoised MSE=%.5f)\n", en/n, eo/n);
    free(clean);free(noisy);free(out);
}
static void test_bilateral_denoises(void){
    int nz=16,ny=24,nx=24,n=nz*ny*nx;
    float *clean=malloc(4*n),*noisy=malloc(4*n),*out=malloc(4*n);
    unsigned s=999;
    /* smooth spatial structure so neighbors are genuinely similar */
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        double v=0.5+0.3*sin(x*0.25)*sin(y*0.2);
        clean[(z*ny+y)*nx+x]=(float)v;
        s=s*1103515245u+12345u;double nse=((s>>16)&0x7fff)/32768.0-0.5;
        noisy[(z*ny+y)*nx+x]=(float)(v+0.08*nse);
    }
    /* sigma_range >= noise amplitude so the filter averages across noise */
    fy_bilateral_denoise(noisy,out,nz,ny,nx,1.5,0.10,2);
    double en=0,eo=0;for(int i=0;i<n;i++){double a=noisy[i]-clean[i],b=out[i]-clean[i];en+=a*a;eo+=b*b;}
    CHECK(eo<en,"bilateral reduces error vs clean");
    printf("      (noisy MSE=%.5f -> denoised MSE=%.5f)\n", en/n, eo/n);
    free(clean);free(noisy);free(out);
}


static void test_process_recipe(void){
    fy_physics p={1000,78,220,2.4,1.2,4.0};
    int nz=16,ny=64,nx=64,n=nz*ny*nx;
    float*in=malloc(4*n),*out=malloc(4*n);
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        double v=(y%20<10)?0.6+0.1*sin(x*0.5):0.05; // sheets + air
        in[(z*ny+y)*nx+x]=(float)v;
    }
    fy_recipe r=fy_recipe_default();
    int rc=fy_process(in,out,nz,ny,nx,&p,&r);
    CHECK(rc==0,"fy_process default runs");
    int finite=1; for(int i=0;i<n;i++) if(!isfinite(out[i])) finite=0;
    CHECK(finite,"fy_process output finite");
    fy_recipe d=fy_recipe_default();
    CHECK(d.deconv_reg>0 && d.air_thresh>0,"default recipe sharpens + masks");
    free(in);free(out);
}


static void test_streaming_global(void){
    /* chunked two-pass GLCAE-global == whole-volume (the 20TB design) */
    int n=8*32*32; unsigned char*vol=malloc(n);
    for(int i=0;i<n;i++) vol[i]=(unsigned char)((i*7)%256);
    fy_hist_state st; fy_hist_init(&st);
    for(int z=0;z<8;z+=2) fy_hist_accumulate_u8(&st, vol+z*32*32, 2*32*32);
    int mp[256]; fy_glcae_global_finalize(&st,mp);
    float*oc=malloc(4*n);
    for(int z=0;z<8;z+=2) fy_glcae_global_apply_u8(vol+z*32*32, oc+z*32*32, 2*32*32, mp);
    fy_hist_state s2; fy_hist_init(&s2); fy_hist_accumulate_u8(&s2,vol,n);
    int mp2[256]; fy_glcae_global_finalize(&s2,mp2);
    float*ow=malloc(4*n); fy_glcae_global_apply_u8(vol,ow,n,mp2);
    double md=0; for(int i=0;i<n;i++){double d=fabs(oc[i]-ow[i]);if(d>md)md=d;}
    CHECK(md<1e-6,"streaming GLCAE-global == whole-volume (chunked correctness)");
    free(vol);free(oc);free(ow);
}


static void test_gureyev_deconv(void){
    fy_physics p={1000,77,220,2.403,1.2,4.0,0.6};
    int nz=24,ny=24,nx=24,n=nz*ny*nx;
    float*in=malloc(4*n),*out=malloc(4*n);
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        double r=(x-12.0)*(x-12.0)+(y-12.0)*(y-12.0)+(z-12.0)*(z-12.0);
        in[(z*ny+y)*nx+x]=(float)exp(-r/30.0);
    }
    int rc=fy_deconvolve_gureyev(in,out,nz,ny,nx,&p,0.02);
    CHECK(rc==0,"gureyev deconv runs");
    int finite=1; for(int i=0;i<n;i++) if(!isfinite(out[i])) finite=0;
    CHECK(finite,"gureyev output finite");
    double si=0,so=0,mi=0,mo=0;
    for(int i=0;i<n;i++){mi+=in[i];mo+=out[i];} mi/=n;mo/=n;
    for(int i=0;i<n;i++){si+=(in[i]-mi)*(in[i]-mi);so+=(out[i]-mo)*(out[i]-mo);}
    CHECK(so>si,"gureyev sharpens (raises contrast vs input)");
}


static void test_fsc(void){
    int nz=32,ny=32,nx=32,n=nz*ny*nx;
    float*s=malloc(4*n),*b=malloc(4*n),*fr=malloc(4*24),*fc=malloc(4*24);
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++)
        s[(z*ny+y)*nx+x]=(float)(0.5+0.3*sin(x*0.3)*cos(y*0.25)+0.15*sin(x*1.1));
    /* FSC of a volume with itself = ~1 everywhere (perfect correlation) */
    fy_fsc(s,s,nz,ny,nx,24,fr,fc);
    int high=1; for(int k=1;k<12;k++) if(fc[k]<0.9f) high=0;
    CHECK(high,"FSC(vol,vol) ~ 1 (self-correlation)");
    /* blurred resolves <= structured */
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        double a=0;int c=0;for(int dx=-2;dx<=2;dx++){int xx=x+dx;if(xx<0||xx>=nx)continue;a+=s[(z*ny+y)*nx+xx];c++;}
        b[(z*ny+y)*nx+x]=(float)(a/c);}
    float rs=-1; int rc=fy_fsc_self(s,nz,ny,nx,12,0.5f,&rs,NULL,NULL);
    CHECK(rc==0 && rs>=0.0f && rs<=1.0f,"FSC self returns valid resolution fraction");
    free(s);free(b);free(fr);free(fc);
}


static void test_zdrift(void){
    int nz=80,ny=24,nx=24,n=nz*ny*nx;
    float*v=malloc(4*n);
    for(int z=0;z<nz;z++){float d=1.0f-0.13f*z/(nz-1);
        for(int i=0;i<ny*nx;i++) v[(size_t)z*ny*nx+i]=0.6f*d;}
    fy_correct_zdrift(v,nz,ny,nx,0.1f);
    double m0=0,mN=0; for(int i=0;i<ny*nx;i++){m0+=v[i];mN+=v[(size_t)(nz-1)*ny*nx+i];}
    m0/=ny*nx; mN/=ny*nx;
    CHECK(fabs(m0-mN)/m0 < 0.03,"z-drift removed (slice0 ~ sliceN after)");
    free(v);
}


static void test_auto_thresh(void){
    fy_hist_state s; fy_hist_init(&s);
    for(int i=0;i<256;i++){
        long air=(long)(100000*exp(-(i-25.0)*(i-25.0)/200.0));
        long pap=(long)(120000*exp(-(i-160.0)*(i-160.0)/2000.0));
        s.hist[i]=air+pap; s.total+=air+pap;
    }
    float th=fy_auto_air_thresh(&s);
    CHECK(th>0.1f && th<0.5f,"Otsu auto air-threshold lands in the valley");
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

static void test_compact(void){
    int nz=48,ny=48,nx=48; size_t n=(size_t)nz*ny*nx;
    float *v=malloc(4*n), *out=malloc(4*n);
    /* OVERSAMPLED volume: only low frequencies -> downsample 2x should be ~lossless */
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++)
        v[((size_t)z*ny+y)*nx+x]=(float)(0.5+0.3*sin(x*0.15)*cos(y*0.13)*sin(z*0.12));
    int oz,oy,ox;
    int rc=fy_downsample(v,out,nz,ny,nx,2.0,&oz,&oy,&ox);
    CHECK(rc==0 && oz==24 && oy==24 && ox==24, "downsample 2x gives half-size grid");
    /* oversampled (smooth) volume -> recommends a shrink */
    double f_smooth=fy_recommend_downsample(v,nz,ny,nx,0.03);
    CHECK(f_smooth>1.0, "oversampled volume -> recommends downsample >1x");
    /* HIGH-frequency volume (detail at Nyquist) -> NOT compactible */
    float *hf=malloc(4*n);
    unsigned s=77;
    for(int i=0;i<(int)n;i++){ s=s*1103515245u+12345u; hf[i]=0.5f+0.4f*(((s>>16)&1)?1:-1); } /* checkerboard-ish */
    double f_hf=fy_recommend_downsample(hf,nz,ny,nx,0.03);
    CHECK(f_hf < f_smooth, "high-freq volume less compactible than smooth one");
    free(v);free(out);free(hf);
}

static void test_gat_and_quality(void){
    /* GAT forward->inverse round-trips exactly */
    size_t n=1000; float *in=malloc(4*n),*t=malloc(4*n),*back=malloc(4*n);
    for(size_t i=0;i<n;i++) in[i]=0.01f+0.0009f*i;  /* spread of intensities */
    double g=0.02,b=1e-4;
    fy_gat_forward(in,t,n,g,b);
    fy_gat_inverse(t,back,n,g,b);
    double maxe=0; for(size_t i=0;i<n;i++){double e=fabs(back[i]-in[i]); if(e>maxe)maxe=e;}
    CHECK(maxe<1e-4, "GAT forward->inverse round-trips");
    free(in);free(t);free(back);
    /* quality denoise reduces error vs clean on a noisy textured volume */
    int nz=24,ny=24,nx=24; size_t m=(size_t)nz*ny*nx;
    float *clean=malloc(4*m),*noisy=malloc(4*m),*out=malloc(4*m);
    unsigned s=4242;
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        double v=0.4+0.25*sin(x*0.5)*cos(y*0.4);
        clean[((size_t)z*ny+y)*nx+x]=(float)v;
        s=s*1103515245u+12345u; double nse=((s>>16)&0x7fff)/32768.0-0.5;
        noisy[((size_t)z*ny+y)*nx+x]=(float)(v+0.06*nse);
    }
    int rc=fy_denoise_quality(noisy,out,nz,ny,nx);
    CHECK(rc==0,"fy_denoise_quality runs");
    double en=0,eo=0; for(size_t i=0;i<m;i++){double a=noisy[i]-clean[i],bb=out[i]-clean[i];en+=a*a;eo+=bb*bb;}
    CHECK(eo<en,"quality denoise reduces error vs clean");
    free(clean);free(noisy);free(out);
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

static void test_sheetness(void){
    int nz=32,ny=32,nx=32; size_t n=(size_t)nz*ny*nx;
    float*in=malloc(4*n),*out=malloc(4*n);
    /* thin bright sheet (single z-plane) on dark bg -> after blur, a plate */
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++)
        in[((size_t)z*ny+y)*nx+x]=(z==16)?0.8f:0.05f;
    fy_sheetness(in,out,nz,ny,nx,2.0,0.5,0.5,-1,1);
    float on=out[((size_t)16*ny+16)*nx+16], off=out[((size_t)8*ny+16)*nx+16];
    CHECK(on>0.5f && off<0.05f, "sheetness lights up plate, dark off it");
    /* selectivity: a tube and a blob must score far lower than a sheet */
    for(size_t i=0;i<n;i++)in[i]=0.05f;
    for(int x=0;x<nx;x++) in[((size_t)16*ny+16)*nx+x]=0.8f;
    fy_sheetness(in,out,nz,ny,nx,2.0,0.5,0.5,-1,1);
    float tube=out[((size_t)16*ny+16)*nx+16];
    for(size_t i=0;i<n;i++)in[i]=0.05f;
    in[((size_t)16*ny+16)*nx+16]=0.8f;
    fy_sheetness(in,out,nz,ny,nx,2.0,0.5,0.5,-1,1);
    float blob=out[((size_t)16*ny+16)*nx+16];
    CHECK(on>2.0f*tube && on>2.0f*blob, "sheetness selective (sheet >> tube,blob)");
    free(in);free(out);
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

static void test_downsample2x(void) {
    int nz=32,ny=32,nx=32,n=nz*ny*nx;
    float *in=malloc(4*n);
    make_struct_vol(in,nz,ny,nx);
    int dz,dy,dx;
    float *out=malloc(4*(n/8));
    int rc=fy_downsample2x(in,out,nz,ny,nx,&dz,&dy,&dx);
    CHECK(rc==0 && dz==16 && dy==16 && dx==16, "downsample2x halves dims");
    /* range preserved (anti-alias shouldn't blow up or zero the signal) */
    float mn=1e9f,mx=-1e9f;
    for(int i=0;i<dz*dy*dx;i++){if(out[i]<mn)mn=out[i];if(out[i]>mx)mx=out[i];}
    CHECK(mx>0.5f && mn>=0.0f && mx<=1.01f, "downsample2x preserves dynamic range");
    free(in);free(out);
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

/* ============ LANDMARK affine fit + multi-resolution fusion ============== */

static double det33(const double *M) { /* linear part of a 3x4, leading 3 cols */
    return M[0]*(M[5]*M[10]-M[6]*M[9]) - M[1]*(M[4]*M[10]-M[6]*M[8])
         + M[2]*(M[4]*M[9]-M[5]*M[8]);
}

static void test_affine_from_points_exact(void) {
    /* known full affine: dst = A p + t, exact fit must recover it to ~0 residual */
    double A[9] = {0.47, -0.01, 0.02,  0.015, 0.46, -0.008,  -0.01, 0.012, 0.48};
    double t[3] = {120.0, -33.0, 57.0};
    int n = 12;
    double src[36], dst[36];
    unsigned int s = 7u;
    for (int i = 0; i < n; i++) {
        for (int a = 0; a < 3; a++) { s = s*1103515245u+12345u; src[i*3+a] = (double)((s>>9)%2000); }
        for (int r = 0; r < 3; r++)
            dst[i*3+r] = A[r*3+0]*src[i*3+0]+A[r*3+1]*src[i*3+1]+A[r*3+2]*src[i*3+2]+t[r];
    }
    double M[12], rms = -1;
    int rc = fy_affine_from_points(src, dst, n, 0, 0, 0.0, M, NULL, &rms);
    double aerr = 0;
    for (int r = 0; r < 3; r++) {
        aerr += fabs(M[r*4+0]-A[r*3+0])+fabs(M[r*4+1]-A[r*3+1])+fabs(M[r*4+2]-A[r*3+2]);
        aerr += fabs(M[r*4+3]-t[r]) * 1e-3;
    }
    printf("     [affine_from_points] rms=%.3e  coeff_err=%.3e\n", rms, aerr);
    CHECK(rc==0, "affine_from_points returns success");
    CHECK(rms < 1e-6, "affine_from_points exact fit residual ~0");
    CHECK(aerr < 1e-4, "affine_from_points recovers the known matrix");
}

static void test_similarity_from_points(void) {
    /* known similarity: scale*R*p + t, with a real rotation. Fit must recover it
     * AND keep equal singular values (no shear) -- the multi-res scan model. */
    double th = 0.21, s = 0.4701;
    double R[9] = { cos(th),-sin(th),0,  sin(th),cos(th),0,  0,0,1 };
    double tt[3] = {200.0, 88.0, -45.0};
    int n = 10;
    double src[30], dst[30];
    unsigned int sd = 99u;
    for (int i = 0; i < n; i++) {
        for (int a=0;a<3;a++){ sd=sd*1103515245u+12345u; src[i*3+a]=(double)((sd>>9)%1500); }
        for (int r=0;r<3;r++)
            dst[i*3+r]= s*(R[r*3+0]*src[i*3+0]+R[r*3+1]*src[i*3+1]+R[r*3+2]*src[i*3+2]) + tt[r];
    }
    double M[12], rms=-1;
    int rc = fy_affine_from_points(src, dst, n, 1, 0, 0.0, M, NULL, &rms);
    double d = det33(M);
    double scale_est = cbrt(fabs(d));
    printf("     [similarity] rms=%.3e  est_scale=%.5f (true %.5f)\n", rms, scale_est, s);
    CHECK(rc==0, "similarity fit returns success");
    CHECK(rms < 1e-5, "similarity exact fit residual ~0");
    CHECK(fabs(scale_est - s) < 1e-3, "similarity recovers the isotropic scale");
}

static void test_affine_ransac_outliers(void) {
    /* 16 good correspondences + 6 gross outliers. RANSAC must reject outliers and
     * recover the clean transform; plain LS would be corrupted. */
    double A[9] = {0.5,0,0, 0,0.5,0, 0,0,0.5};
    double t[3] = {10,20,30};
    int n = 22;
    double src[66], dst[66];
    unsigned int s = 3u;
    for (int i = 0; i < n; i++) {
        for (int a=0;a<3;a++){ s=s*1103515245u+12345u; src[i*3+a]=(double)((s>>9)%1000); }
        for (int r=0;r<3;r++)
            dst[i*3+r]=A[r*3+0]*src[i*3+0]+A[r*3+1]*src[i*3+1]+A[r*3+2]*src[i*3+2]+t[r];
        if (i >= 16) { /* corrupt the last 6 with large random offsets */
            for (int r=0;r<3;r++){ s=s*1103515245u+12345u; dst[i*3+r]+=(double)((int)((s>>9)%800)-400);}
        }
    }
    double Mls[12], Mr[12], rms_ls=-1, rms_r=-1; int inl[22];
    fy_affine_from_points(src, dst, n, 0, 0, 0.0, Mls, NULL, &rms_ls);  /* plain LS */
    int rc = fy_affine_from_points(src, dst, n, 0, 200, 2.0, Mr, inl, &rms_r); /* RANSAC */
    int ninl = 0; for (int i=0;i<n;i++) ninl+=inl[i];
    /* coeff error of RANSAC vs truth */
    double aerr=0; for (int r=0;r<3;r++) for(int c=0;c<3;c++) aerr+=fabs(Mr[r*4+c]-A[r*3+c]);
    printf("     [ransac] LS_rms=%.2f  RANSAC_rms=%.3e  inliers=%d/22  coeff_err=%.2e\n",
           rms_ls, rms_r, ninl, aerr);
    CHECK(rc==0, "ransac affine returns success");
    CHECK(ninl==16, "ransac identifies exactly the 16 inliers");
    CHECK(rms_r < 1e-6, "ransac residual ~0 on the clean inliers");
    CHECK(aerr < 1e-4, "ransac recovers the clean transform despite outliers");
    CHECK(rms_ls > 10.0, "plain least-squares IS corrupted by the outliers (control)");
}

static void test_fuse_denoises(void) {
    /* Synthetic: a smooth low-freq "structure" + a fine high-freq detail. Make two
     * INDEPENDENT noisy realizations sharing the same low band (fine: low+high+noise1;
     * coarse: low+noise2, NO high band). Fusion must (a) lower the low-band noise vs
     * the fine scan alone, and (b) keep the fine high-freq detail. */
    int nz=48,ny=48,nx=48; size_t N=(size_t)nz*ny*nx;
    float *low=malloc(4*N),*high=malloc(4*N);
    float *fine=malloc(4*N),*coarse=malloc(4*N),*fused=malloc(4*N);
    /* ground-truth low (smooth) and high (fine ripples) */
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        size_t i=(z*ny+y)*nx+x;
        double fz=(double)z/nz,fy=(double)y/ny,fx=(double)x/nx;
        low[i]=(float)(0.5+0.25*sin(3.0*fx*2*M_PI)*cos(2.0*fy*2*M_PI)*sin(1.5*fz*2*M_PI+0.4));
        high[i]=(float)(0.12*sin(10.0*fx*2*M_PI)*sin(9.0*fy*2*M_PI+0.7));
    }
    /* two independent noise fields */
    unsigned int s1=11u,s2=777u;
    double sig=0.04;
    for(size_t i=0;i<N;i++){
        s1=s1*1103515245u+12345u; double n1=((double)((s1>>9)%10000)/10000.0-0.5)*2*1.732*sig;
        s2=s2*1103515245u+12345u; double n2=((double)((s2>>9)%10000)/10000.0-0.5)*2*1.732*sig;
        fine[i]=(float)(low[i]+high[i]+n1);
        coarse[i]=(float)(low[i]+n2);   /* same low band, independent noise, no detail */
    }
    /* fuse: equal noise variances -> ~0.5/0.5 low-band average */
    double v=sig*sig;
    fy_fuse_multiscale(fine, coarse, NULL, nz,ny,nx, 3.0, v, v, 1.0, fused);

    /* Measure LOW-BAND noise the way fusion defines it: low band = the same gaussian
     * split (sigma=3) used inside the kernel, applied here as a wide box-blur proxy
     * (radius 4 ~ sigma 2-3). noise = std of (lowpass(volume) - true_low_signal).
     * Fusion averages two independent low bands -> ~1/sqrt(2) the noise of the fine
     * low band alone. We compare fine-alone vs fused low-band error to true_low. */
    /* Measure LOW-BAND noise: take (result - true_signal) -- this is pure noise since
     * truth is noise-free -- and LOW-PASS it (the high band is undenoised by design, so
     * we look only at the shared low band where fusion acts). fine's low-band noise is
     * noise1; fused's is the 0.5/0.5 average of two INDEPENDENT noises -> lower std.
     * Low-pass via 3 box-blur passes (radius 2 ~ gaussian sigma~2.8, matching split). */
    float *rf=malloc(4*N),*rg=malloc(4*N),*tb=malloc(4*N);
    for(size_t i=0;i<N;i++){ double truth=(double)low[i]+(double)high[i];
        rf[i]=(float)(fine[i]-truth); rg[i]=(float)(fused[i]-truth); }
    for(int pass=0;pass<2;pass++){
        float *buf=(pass==0)?rf:rg;
        for(int it=0;it<3;it++){
            int rr=2;
            for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
                double a=0;int cc=0;
                for(int dz=-rr;dz<=rr;dz++)for(int dy=-rr;dy<=rr;dy++)for(int dx=-rr;dx<=rr;dx++){
                    int zz=z+dz,yy=y+dy,xx=x+dx; if(zz<0||yy<0||xx<0||zz>=nz||yy>=ny||xx>=nx)continue;
                    a+=buf[(zz*ny+yy)*nx+xx]; cc++; }
                tb[(z*ny+y)*nx+x]=(float)(a/cc);
            }
            memcpy(buf,tb,N*4);
        }
    }
    int R=6;
    double err_fine=0, err_fused=0; long c=0;
    for(int z=R;z<nz-R;z++)for(int y=R;y<ny-R;y++)for(int x=R;x<nx-R;x++){
        size_t i=(z*ny+y)*nx+x;
        err_fine+=(double)rf[i]*rf[i]; err_fused+=(double)rg[i]*rg[i]; c++;
    }
    err_fine=sqrt(err_fine/c); err_fused=sqrt(err_fused/c);
    free(rf);free(rg);free(tb);
    /* high-freq retention: the high band the kernel keeps is fine - gaussian(fine).
     * Measure correlation of (fused minus its OWN wide blur) with the true high band. */
    float *hb=malloc(4*N);
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        size_t i=(z*ny+y)*nx+x; double a=0; int cc=0;
        for(int dz=-R;dz<=R;dz++)for(int dy=-R;dy<=R;dy++)for(int dx=-R;dx<=R;dx++){
            int zz=z+dz,yy=y+dy,xx=x+dx; if(zz<0||yy<0||xx<0||zz>=nz||yy>=ny||xx>=nx)continue;
            a+=fused[(zz*ny+yy)*nx+xx]; cc++; }
        hb[i]=(float)(fused[i]-a/cc);
    }
    double hr=pearson(hb, high, N);
    printf("     [fuse] low-band noise vs truth: fine-alone=%.5f  fused=%.5f  (ratio %.2f, ideal 0.71)  high-retention=%.3f\n",
           err_fine, err_fused, err_fused/err_fine, hr);
    CHECK(err_fused < err_fine*0.90, "fusion lowers low-band noise vs fine-alone (>10%)");
    CHECK(hr > 0.5, "fusion retains the fine high-freq detail");
    free(low);free(high);free(fine);free(coarse);free(fused);free(hb);
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

/* Coherence-enhancing diffusion: two parallel bright sheets separated by a thin
 * dark gap, plus noise. CED must (a) drop the noise on the sheets AND (b) keep the
 * dark gap dark (NOT merge the layers). A plain isotropic blur would fill the gap. */
static void test_coherence_diffusion_gap(void) {
    int nz=48,ny=48,nx=48,n=nz*ny*nx;
    float *clean=malloc(4*n), *noisy=malloc(4*n), *ced=malloc(4*n), *gauss=malloc(4*n);
    /* two bright sheets perpendicular to x: a plane at x=22 and x=26 (3-vox gap at
     * x=23,24,25 stays dark). Sheets span all y,z (planar -> coherent). */
    int s1=22, s2=26;
    for (int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        float v=0.1f;                         /* dark background */
        if (x==s1-1||x==s1||x==s1+1) v=0.9f;  /* sheet 1 (3 vox thick) */
        if (x==s2-1||x==s2||x==s2+1) v=0.9f;  /* sheet 2 */
        clean[((size_t)z*ny+y)*nx+x]=v;
    }
    /* add noise */
    unsigned int seed=1234567u;
    for (int i=0;i<n;i++){ seed=seed*1103515245u+12345u;
        float r=((float)((seed>>9)&1023)/1023.0f - 0.5f)*0.30f;
        noisy[i]=clean[i]+r; }

    double sigma=0.7, rho=3.0, tau=0.10, alpha=0.001; int iters=10;
    int rc=fy_coherence_diffusion(noisy,ced,nz,ny,nx,sigma,rho,tau,iters,alpha);
    CHECK(rc==0, "fy_coherence_diffusion returns 0");

    /* matched-smoothing Gaussian blur for comparison (fills the gap -> the bad case).
     * sigma chosen ~ along-sheet smoothing reach; reuse fy_sheetness path? just do a
     * separable gaussian via the public guided? simplest: small explicit gaussian.   */
    /* approximate a matched isotropic gaussian using repeated 3-pt smoothing */
    memcpy(gauss,noisy,4*n);
    for (int pass=0; pass<8; pass++){
        float *tmp=malloc(4*n); memcpy(tmp,gauss,4*n);
        for (int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
            int xm=x>0?x-1:0,xp=x<nx-1?x+1:nx-1;
            int ym=y>0?y-1:0,yp=y<ny-1?y+1:ny-1;
            int zm=z>0?z-1:0,zp=z<nz-1?z+1:nz-1;
            size_t i=((size_t)z*ny+y)*nx+x;
            gauss[i]=(tmp[((size_t)z*ny+y)*nx+xm]+tmp[((size_t)z*ny+y)*nx+xp]
                     +tmp[((size_t)z*ny+ym)*nx+x]+tmp[((size_t)z*ny+yp)*nx+x]
                     +tmp[((size_t)zm*ny+y)*nx+x]+tmp[((size_t)zp*ny+y)*nx+x]
                     +tmp[i])/7.0f;
        }
        free(tmp);
    }

    /* (a) sheet-interior noise: std over the interior of sheet 1 (x=s1), central yz. */
    #define INTERIOR(buf,xc) ({ double m=0,m2=0; int c=0; \
        for(int z=8;z<nz-8;z++)for(int y=8;y<ny-8;y++){ \
            double v=buf[((size_t)z*ny+y)*nx+(xc)]; m+=v;m2+=v*v;c++;} \
        m/=c; sqrt(m2/c-m*m); })
    double noisy_std=INTERIOR(noisy,s1);
    double ced_std  =INTERIOR(ced,s1);
    printf("     [ced] sheet-interior std: noisy=%.4f -> CED=%.4f\n", noisy_std, ced_std);
    CHECK(ced_std < 0.6*noisy_std, "CED drops sheet-interior noise substantially (>40%)");

    /* (b) GAP PRESERVATION: gap center is x=24. Contrast = sheet_mean - gap_min over
     * the central region. CED must keep the gap dark; Gaussian fills it. */
    #define GAPMIN(buf) ({ double mn=1e9; \
        for(int z=12;z<nz-12;z++)for(int y=12;y<ny-12;y++){ \
            double v=buf[((size_t)z*ny+y)*nx+24]; if(v<mn)mn=v;} mn; })
    #define GAPMEAN(buf) ({ double m=0;int c=0; \
        for(int z=12;z<nz-12;z++)for(int y=12;y<ny-12;y++){ \
            m+=buf[((size_t)z*ny+y)*nx+24];c++;} m/c; })
    #define SHEETMEAN(buf,xc) ({ double m=0;int c=0; \
        for(int z=12;z<nz-12;z++)for(int y=12;y<ny-12;y++){ \
            m+=buf[((size_t)z*ny+y)*nx+(xc)];c++;} m/c; })
    double clean_gap=GAPMEAN(clean), clean_sheet=SHEETMEAN(clean,s1);
    double ced_gap=GAPMEAN(ced),     ced_sheet=SHEETMEAN(ced,s1);
    double gau_gap=GAPMEAN(gauss),   gau_sheet=SHEETMEAN(gauss,s1);
    double clean_contrast=clean_sheet-clean_gap;
    double ced_contrast  =ced_sheet-ced_gap;
    double gau_contrast  =gau_sheet-gau_gap;
    printf("     [ced] gap-mean: clean=%.3f CED=%.3f gauss=%.3f\n",clean_gap,ced_gap,gau_gap);
    printf("     [ced] sheet-vs-gap contrast: clean=%.3f CED=%.3f(%.0f%%) gauss=%.3f(%.0f%%)\n",
           clean_contrast, ced_contrast, 100*ced_contrast/clean_contrast,
           gau_contrast, 100*gau_contrast/clean_contrast);
    /* CED retains most of the gap contrast (>70% of clean) */
    CHECK(ced_contrast > 0.70*clean_contrast, "CED PRESERVES the inter-sheet gap (>70% contrast retained)");
    /* and is decisively better at it than matched gaussian blur */
    CHECK(ced_contrast > 1.3*gau_contrast, "CED preserves the gap far better than a matched Gaussian blur");

    free(clean);free(noisy);free(ced);free(gauss);
}

static void test_coherence_diffusion_auto(void) {
    /* auto path runs end-to-end on a two-sheet+gap volume, denoises, keeps the gap */
    int nz=40,ny=40,nx=40; size_t n=(size_t)nz*ny*nx;
    float *in=malloc(4*n), *out=malloc(4*n);
    unsigned s=271;
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){
        /* two bright x-y sheets at z=16 and z=22 (gap at z=19) */
        double v=(z==16||z==22)?0.8:0.1;
        s=s*1103515245u+12345u; double nse=((s>>16)&0x7fff)/32768.0-0.5;
        in[((size_t)z*ny+y)*nx+x]=(float)(v+0.05*nse);
    }
    int rc=fy_coherence_diffusion_auto(in,out,nz,ny,nx,2);
    CHECK(rc==0,"fy_coherence_diffusion_auto runs");
    int finite=1; for(size_t i=0;i<n;i++) if(!isfinite(out[i])) finite=0;
    CHECK(finite,"auto CED output finite");
    /* gap (z=19) stays darker than sheets (z=16) */
    double gap=0,sheet=0; int c=0;
    for(int y=0;y<ny;y++)for(int x=0;x<nx;x++){gap+=out[((size_t)19*ny+y)*nx+x];sheet+=out[((size_t)16*ny+y)*nx+x];c++;}
    CHECK(gap/c < 0.7*(sheet/c), "auto CED keeps the inter-sheet gap (gap<<sheet)");
    free(in);free(out);
}

int main(void) {
    test_fft_vs_dft();
    test_fft_vs_dft();
    test_fft_roundtrip();
    test_paganin_transfer();
    test_deconvolve_sharpens();
    test_halo_reasonable();
    test_nlm_denoises();
    test_bilateral_denoises();
    test_process_recipe();
    test_streaming_global();
    test_gureyev_deconv();
    test_fsc();
    test_zdrift();
    test_auto_thresh();
    test_sheetness();
    test_dewindow();
    test_estimate_noise();
    test_compact();
    test_gat_and_quality();
    test_deltabeta_scale();
    test_warp_identity();
    test_warp_translation();
    test_warp_roundtrip();
    test_downsample2x();
    test_ncc_self();
    test_register_rigid();
    test_register_affine();
    test_register_contrast();
    test_warp_field_identity();
    test_demons_known_warp();
    test_demons_regularization();
    test_register_full_affine_plus_demons();
    test_affine_from_points_exact();
    test_similarity_from_points();
    test_affine_ransac_outliers();
    test_fuse_denoises();
    test_phase_correlate_subvoxel();
    test_mutual_information_peaks();
    test_coherence_diffusion_gap();
    test_coherence_diffusion_auto();
    printf("\n%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
