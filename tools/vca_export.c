/* vca_export.c -- FUSED single-pass export: uncompressed zarr (local or s3://) -> preprocess
 * (fysics BM18 pipeline: denoise + air-zero, gated norm/zdrift) -> v2/v3 VCA archive (.v3).
 * No intermediate zarr. The air-zero output feeds the v3 archive's mask-from-zeros directly.
 *
 * Pipeline: fy_calibrate ONCE on the input -> for each 128^3 chunk (OpenMP), fy_process_chunk
 * with halo -> hand the processed 128^3 to the v3 vsrc -> v3_build_from_vsrc encodes all LODs.
 *
 * Usage: vca_export <in_zarr|s3url> <out.v3> [--profile conservative|aggressive] [--dim N]
 *        [--quality Q] [--meta PATH] [--tile 128]
 */
#define _GNU_SOURCE
#include "fysics.h"
#include "../vendor/vca/v3archive_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static void apply_meta(fy_pipeline_cfg *c, const char *key, double v) {
    if      (!strcmp(key,"delta_beta"))            c->delta_beta = v;
    else if (!strcmp(key,"energy_kev"))            c->energy_kev = v;
    else if (!strcmp(key,"distance_mm"))           c->distance_mm = v;
    else if (!strcmp(key,"pixel_um"))              c->pixel_um = v;
    else if (!strcmp(key,"unsharp_sigma"))         c->unsharp_sigma = v;
    else if (!strcmp(key,"unsharp_coeff"))         c->unsharp_coeff = v;
    else if (!strcmp(key,"machine_current_start")) c->machine_current_start = v;
    else if (!strcmp(key,"machine_current_stop"))  c->machine_current_stop = v;
    else if (!strcmp(key,"window_lo"))             c->window_lo = v;
    else if (!strcmp(key,"window_hi"))             c->window_hi = v;
}

int main(int argc, char **argv){
    if (argc < 3) {
        fprintf(stderr,"usage: %s <in_zarr|s3url> <out.v3> [--profile P] [--dim N] [--quality Q]"
                       " [--meta PATH] [--tile T]\n", argv[0]);
        return 2;
    }
    const char *in = argv[1], *out = argv[2];
    const char *profile = "conservative";
    int dim = 1024, tile = 128;
    float quality = 8.0f;
    char meta_path[PATH_MAX]; meta_path[0]=0;
    for (int i=3;i<argc;i++){
        if      (!strcmp(argv[i],"--profile") && i+1<argc) profile = argv[++i];
        else if (!strcmp(argv[i],"--dim") && i+1<argc)     dim = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--quality") && i+1<argc) quality = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--tile") && i+1<argc)    tile = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--meta") && i+1<argc)    snprintf(meta_path,sizeof meta_path,"%s",argv[++i]);
        else { fprintf(stderr,"unknown arg: %s\n",argv[i]); return 2; }
    }
    if (!meta_path[0]) snprintf(meta_path,sizeof meta_path,"%s/metadata.json", in);

    /* ---- BM18 defaults + profile ---- */
    fy_pipeline_cfg cfg; memset(&cfg,0,sizeof cfg);
    cfg.delta_beta=1000.0; cfg.energy_kev=78.0; cfg.distance_mm=220.0; cfg.pixel_um=2.4;
    cfg.unsharp_sigma=1.2; cfg.unsharp_coeff=4.0;
    cfg.do_deconv=0; cfg.use_matched_deconv=1; cfg.psf_sigma_vox=1.0; cfg.deconv_tikhonov=0.05;
    cfg.guided_eps=0.0;
    cfg.do_air_zero=1; cfg.air_cut_u8=-1; cfg.air_cut_band=8; cfg.air_thresh=0.05;
    cfg.scratch_passes=5;
    cfg.do_normalize=1; cfg.norm_lo=-1; cfg.norm_hi=-1;
    cfg.do_zdrift=1; cfg.zdrift_factor=NULL;
    cfg.do_musica=0; cfg.musica_p=0.6; cfg.musica_levels=4; cfg.musica_core=0.0;
    if      (!strcmp(profile,"conservative")) { cfg.air_cut_aggr=0.0; cfg.denoise_k=3.0; }
    else if (!strcmp(profile,"aggressive"))   { cfg.air_cut_aggr=1.0; cfg.denoise_k=4.2; }
    else { fprintf(stderr,"unknown profile %s\n",profile); return 2; }

    /* ---- metadata via the python reader (same locator as fysics_process) ---- */
    char script[PATH_MAX]={0}, exe[PATH_MAX]; ssize_t en=readlink("/proc/self/exe",exe,sizeof exe-1);
    char exedir[PATH_MAX]={0}; if(en>0){ exe[en]=0; snprintf(exedir,sizeof exedir,"%s",dirname(exe)); }
    const char *cands[4]; int nc=0; char c0[PATH_MAX],c1[PATH_MAX];
    const char *env=getenv("FYSICS_READMETA"); if(env) cands[nc++]=env;
    if(exedir[0]){ snprintf(c0,sizeof c0,"%s/read_meta.py",exedir); cands[nc++]=c0;
                   snprintf(c1,sizeof c1,"%s/../tools/read_meta.py",exedir); cands[nc++]=c1; }
    cands[nc++]="read_meta.py";
    for(int i=0;i<nc;i++) if(access(cands[i],R_OK)==0){ snprintf(script,sizeof script,"%s",cands[i]); break; }
    if(!script[0]) snprintf(script,sizeof script,"read_meta.py");
    char cmd[PATH_MAX*2]; snprintf(cmd,sizeof cmd,"python3 '%s' '%s' 2>/dev/null",script,meta_path);
    FILE *pp=popen(cmd,"r"); int meta_lines=0;
    if(pp){ char line[256]; while(fgets(line,sizeof line,pp)){ char*eq=strchr(line,'='); if(!eq)continue;
            *eq=0; apply_meta(&cfg,line,atof(eq+1)); meta_lines++; } pclose(pp); }
    /* window -> physical air threshold (same as fysics_process) */
    if(cfg.window_hi>cfg.window_lo){ double af=(0.0-cfg.window_lo)/(cfg.window_hi-cfg.window_lo);
        if(af<0)af=0; if(af>1)af=1; cfg.air_thresh=af; }

    fprintf(stderr,"vca_export: in=%s out=%s profile=%s dim=%d q=%.1f (%d meta keys)\n",
            in,out,profile,dim,quality,meta_lines);

    /* ---- 1. CALIBRATE once on the input (local or s3://) ---- */
    if (fy_calibrate(in, &cfg, tile, 1) != 0) { fprintf(stderr,"calibrate failed\n"); return 1; }

    /* ---- 2. open the input zarr, process every 128^3 chunk into the v3 vsrc ---- */
    fy_zarr zin; if (fy_zarr_open(&zin, in) != 0) { fprintf(stderr,"open %s failed\n",in); return 1; }
    long V = zin.shape[0];
    if ((int)V != dim) { fprintf(stderr,"[note] zarr dim %ld != --dim %d; using zarr dim\n",V,dim); dim=(int)V; }
    void *vs = v3_vsrc_alloc(dim);
    if (!vs) return 1;
    int G = dim/128;
    long nproc=0, nair=0;
    #pragma omp parallel for collapse(3) schedule(dynamic) reduction(+:nproc,nair)
    for (int cz=0; cz<G; cz++) for (int cy=0; cy<G; cy++) for (int cx=0; cx<G; cx++){
        unsigned char *buf = malloc((size_t)128*128*128);
        if (!buf) continue;
        int tz,ty,tx,air=0;
        fy_process_chunk(&zin, &cfg, (long)cz*128,(long)cy*128,(long)cx*128, 128, buf, &tz,&ty,&tx,&air);
        if (air) { free(buf); nair++; }            /* absent chunk */
        else { v3_vsrc_set_chunk(vs, cz,cy,cx, buf); nproc++; }   /* vsrc owns buf */
    }
    fprintf(stderr,"vca_export: processed %ld chunks (%ld all-air) of %d^3\n", nproc, nair, G*G*G);
    free(cfg.zdrift_factor);

    /* ---- 3. encode all LODs to the .v3 archive ---- */
    int rc = v3_build_from_vsrc(vs, out, dim, quality, NULL, 0);   /* consumes vs */
    if (rc==0) fprintf(stderr,"vca_export: wrote %s\n", out);
    return rc;
}
