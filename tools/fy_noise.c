/* fy_noise.c -- noise / randomness analysis for a raw-u8 zarr (level 0/).
 *
 * Answers: are the low N bits of the (material) voxels TRUE white noise, or fine
 * signal? Noise is spatially UNCORRELATED; signal is correlated. Per-voxel bit
 * frequency (~50%) is necessary but not sufficient -- the decisive test is
 * spatial autocorrelation per bit-plane.
 *
 * Reports, over material (nonzero) voxels:
 *   - value entropy
 *   - per-bit lag-1 spatial autocorrelation (x-axis); ~0 => that bit is white
 *   - the local high-pass residual std (the noise floor, u8 units)
 *   - N_RANDOM: the count of low bits consistent with white noise
 *   - a SUGGESTED quantization quality q ~ noise floor (discard noise, keep signal)
 *
 *   fy_noise <zarr_root> [max_chunks]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static long jint(const char *s, const char *key, int idx) {
    const char *p = strstr(s, key); if (!p) return -1; p += strlen(key);
    for (int i = 0; i <= idx; i++) {
        while (*p && (*p < '0' || *p > '9') && *p != '-') p++;
        if (i < idx) while (*p >= '0' && *p <= '9') p++;
    }
    return strtol(p, NULL, 10);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <zarr_root> [max_chunks]\n", argv[0]); return 2; }
    const char *root = argv[1];
    long max_chunks = argc > 2 ? atol(argv[2]) : 64;
    char path[2048]; snprintf(path, sizeof path, "%s/0/.zarray", root);
    FILE *zf = fopen(path, "rb"); if (!zf) { fprintf(stderr, "no %s\n", path); return 1; }
    char buf[4096]; size_t zn = fread(buf, 1, sizeof buf - 1, zf); buf[zn] = 0; fclose(zf);
    long SZ=jint(buf,"\"shape\"",0), SY=jint(buf,"\"shape\"",1), SX=jint(buf,"\"shape\"",2);
    long CZ=jint(buf,"\"chunks\"",0), CY=jint(buf,"\"chunks\"",1), CX=jint(buf,"\"chunks\"",2);
    if (SZ<=0||CZ<=0){ fprintf(stderr,"bad .zarray\n"); return 1; }
    long ncz=(SZ+CZ-1)/CZ, ncy=(SY+CY-1)/CY, ncx=(SX+CX-1)/CX, ntot=ncz*ncy*ncx;

    // accumulators (over material = nonzero voxels)
    double hist[256]={0}; double material=0, total=0;
    // per-bit lag-1 x-autocorr: need E[Xi], E[Xi*Xi+1] over adjacent material pairs
    double p1[8]={0}, p11[8]={0}; double npair=0;
    // noise floor: high-pass residual r = v[x] - (v[x-1]+v[x+1])/2 on flat material runs
    double sumr2=0, sumr=0; double nres=0;  double sumr_r1=0;  double prev_r=0; int have_prev=0;

    unsigned char *cb = malloc((size_t)CZ*CY*CX);
    long seen=0;
    // sample chunks on a stride so we cover the volume, not just a corner
    long stride = ntot>max_chunks ? ntot/max_chunks : 1;
    for (long ci=0; ci<ntot && seen<max_chunks; ci+=stride) {
        long cz=ci/(ncy*ncx), r=ci%(ncy*ncx), cy=r/ncx, cx=r%ncx;
        long ez=CZ<SZ-cz*CZ?CZ:SZ-cz*CZ, ey=CY<SY-cy*CY?CY:SY-cy*CY, ex=CX<SX-cx*CX?CX:SX-cx*CX;
        snprintf(path,sizeof path,"%s/0/%ld/%ld/%ld",root,cz,cy,cx);
        FILE *f=fopen(path,"rb"); if(!f) continue;       // absent chunk = all air, skip
        size_t got=fread(cb,1,(size_t)ez*ey*ex,f); fclose(f);
        if(got<(size_t)ez*ey*ex) continue;
        seen++;
        for(long z=0;z<ez;z++)for(long y=0;y<ey;y++){
            const unsigned char *row=cb+((size_t)z*ey+y)*ex;
            for(long x=0;x<ex;x++){
                total++; unsigned v=row[x];
                if(!v) continue;
                material++; hist[v]++;
                if(x+1<ex && row[x+1]){       // adjacent material pair (x, x+1)
                    npair++;
                    for(int b=0;b<8;b++){ int bi=(v>>b)&1, bj=(row[x+1]>>b)&1;
                        p1[b]+=bi; p11[b]+=bi*bj; }
                }
                if(x>0 && x+1<ex && row[x-1] && row[x+1]){   // flat-ish: both neighbors material
                    double res = (double)v - 0.5*((double)row[x-1]+(double)row[x+1]);
                    sumr+=res; sumr2+=res*res; nres++;
                    if(have_prev) sumr_r1 += res*prev_r;     // residual lag-1 autocorr
                    prev_r=res; have_prev=1;
                } else have_prev=0;
            }
        }
    }
    free(cb);
    if(material<1000){ fprintf(stderr,"too little material (%.0f voxels)\n",material); return 1; }

    // value entropy (material)
    double ent=0; for(int i=1;i<256;i++){ if(hist[i]>0){ double pp=hist[i]/material; ent-=pp*log2(pp);} }
    // residual noise floor (std of high-pass) and its whiteness
    double rmean=sumr/nres, rvar=sumr2/nres-rmean*rmean, rstd=rvar>0?sqrt(rvar):0;
    // the high-pass r = v-(left+right)/2 has gain 1.5 on white noise -> divide
    double noise_std = rstd/1.2247449;   // sqrt(1+0.25+0.25)=sqrt(1.5)
    double res_ac1 = (sumr_r1/nres - rmean*rmean)/(rvar>0?rvar:1);

    printf("=== %s ===\n", root);
    printf("voxels %.1fM, material %.0f%% (%.1fM), chunks sampled %ld\n",
           total/1e6, 100*material/total, material/1e6, seen);
    printf("value entropy (material): %.2f / 8.00 bits\n", ent);
    printf("per-bit lag-1 x-autocorrelation (material pairs, n=%.0fM):\n", npair/1e6);
    int n_random=0; int counting=1;
    for(int b=0;b<8;b++){
        double P1=p1[b]/npair, P11=p11[b]/npair;
        double ac = (P11 - P1*P1)/(P1*(1-P1)+1e-12);
        const char *tag = fabs(ac)<0.04 ? "WHITE (random)" : ac>0.2 ? "structured (signal)" : "weak";
        printf("  bit%d: P(1)=%.3f  autocorr=%+.4f  %s\n", b, P1, ac, tag);
        if(counting && fabs(ac)<0.04) n_random++; else counting=0;   // contiguous low random bits
    }
    printf("residual noise floor: %.2f u8  (residual lag-1 ac=%+.3f %s)\n",
           noise_std, res_ac1, fabs(res_ac1)<0.1?"~white":"correlated");
    printf("N_RANDOM (contiguous low white bits): %d\n", n_random);
    // suggested q: set the DC quant step ~ the noise floor so quantization discards
    // noise but not signal (HF steps are already larger via the freq weighting).
    double q_suggest = noise_std>0.5 ? noise_std : 0.5;
    printf("SUGGESTED quality (q ~ noise floor): %.1f\n", q_suggest);
    return 0;
}
