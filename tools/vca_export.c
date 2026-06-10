/* vca_export.c -- STREAMING single-pass export: uncompressed OME-zarr (local or s3://)
 * -> fysics preprocess -> matter-compressor archive (.mc), all LODs, bounded RAM.
 *
 * Architecture ported from volume-compressor's vc_export_stream (the optimized tiled
 * S3 tool), fused with the fysics preprocessing chain:
 *
 *   - OCCUPANCY from a coarse pyramid level: one tiny download replaces ~10k HEADs;
 *     fully-absent bands are skipped outright (absent mc chunks decode to zero).
 *   - DOWNLOADER POOL -> bounded queue -> COMPUTE POOL: producers run back-to-back
 *     GETs (saturating S3 concurrency), consumers process already-resident bands --
 *     the latency-bound download never blocks compute.
 *   - WORK UNIT = (XY tile SB x SB, Z band of BAND): chunk-aligned at L0 and L1, so
 *     units are fully independent. Each unit: assemble halo'd raw band -> fysics
 *     process (fy_process_buffer per 128^3 tile) -> append L0 256^3 chunks -> 2x box
 *     downscale -> append L1 chunks. mc's appends are lock-free across threads.
 *   - COARSE TAIL (L2..): built FROM THE ARCHIVE (mc_archive_read_region on L-1,
 *     downscale, append) -- no scratch files; the tail is ~1.6% of the data.
 *
 * Calibration (fy_calibrate) runs once up front on the input zarr.
 *
 * Usage: vca_export <in_zarr|s3://...> <out.mc> [--profile conservative|aggressive]
 *        [--quality Q] [--meta PATH] [--threads N] [--io-threads M] [--queue C]
 *        [--sb SB] [--band B] [--no-process]
 */
#define _GNU_SOURCE
#include "fysics.h"
#include "../vendor/matter/mc_archive_api.h"
#include "../vendor/libs3/libs3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <libgen.h>
#include <limits.h>

typedef uint8_t u8;
#define MCC 256                 /* mc chunk edge */

/* ------------------------------------------------------------------ source I/O */
static s3_client *g_s3 = NULL;
static int is_s3(const char *p){ return strncmp(p,"s3://",5)==0; }
static s3_status cred_provider(void *ud, s3_credentials *out){ (void)ud; return s3_credentials_load(NULL,out); }
static void s3_init_once(void){
    if(g_s3) return;
    s3_config cfg; memset(&cfg,0,sizeof cfg);
    cfg.max_retries=5;
    s3_credentials probe; memset(&probe,0,sizeof probe);
    if(s3_credentials_load(NULL,&probe)==S3_OK){
        cfg.cred_provider=cred_provider;
        if(probe.region&&probe.region[0]) cfg.region=strdup(probe.region);
        s3_credentials_free(&probe);
    }
    g_s3=s3_client_new(&cfg);
    if(!g_s3){ fprintf(stderr,"s3_client_new failed\n"); exit(1); }
}
/* GET whole object/file into a malloc'd buffer; NULL on absent. *err=1 on HARD error
 * (S3 failure that is not a 404) so absent-vs-failed is never conflated. */
static u8 *src_get(const char *path, size_t *len, int *err){
    if(err)*err=0;
    if(is_s3(path)){
        s3_response r; memset(&r,0,sizeof r);
        s3_status st=s3_get(g_s3,path,&r);
        u8 *out=NULL;
        if(st==S3_OK && r.status==200 && r.body){
            /* steal the response body (libs3 frees it with plain free()) -- the
             * stream fetches ~TBs through here; no copy. */
            out=(u8*)r.body; if(len)*len=r.body_len; r.body=NULL;
        } else if(r.status!=404) { if(err)*err=1; }   /* 404 = absent chunk, not an error */
        s3_response_free(&r);
        return out;
    }
    int fd=open(path,O_RDONLY);
    if(fd<0) return NULL;
    struct stat st;
    if(fstat(fd,&st)||!S_ISREG(st.st_mode)){close(fd);return NULL;}
    size_t n=(size_t)st.st_size; u8 *b=malloc(n?n:1);
    if(!b){close(fd);if(err)*err=1;return NULL;}
    ssize_t g=read(fd,b,n); close(fd);
    if(g!=(ssize_t)n){free(b);if(err)*err=1;return NULL;}
    if(len)*len=n;
    return b;
}

/* tiny .zarray parse (same minimal int-grabber as zarr_io.c) */
static long jint_at(const char *s,const char *key,int idx){
    const char *p=strstr(s,key); if(!p)return -1; p+=strlen(key);
    for(int i=0;i<=idx;i++){ while(*p&&(*p<'0'||*p>'9')&&*p!='-')p++;
        if(i<idx){ while(*p>='0'&&*p<='9')p++; } }
    return strtol(p,NULL,10);
}
typedef struct { char dir[1100]; long sz,sy,sx, cz,cy,cx; } zlvl;
static int parse_lvl(const char *zarr,int L,zlvl *lv){
    char p[1300]; snprintf(lv->dir,sizeof lv->dir,"%s/%d",zarr,L);
    snprintf(p,sizeof p,"%s/.zarray",lv->dir);
    size_t n=0; u8 *t=src_get(p,&n,NULL); if(!t)return -1;
    char *s=realloc(t,n+1); s[n]=0;
    lv->sz=jint_at(s,"\"shape\"",0); lv->sy=jint_at(s,"\"shape\"",1); lv->sx=jint_at(s,"\"shape\"",2);
    lv->cz=jint_at(s,"\"chunks\"",0); lv->cy=jint_at(s,"\"chunks\"",1); lv->cx=jint_at(s,"\"chunks\"",2);
    free(s);
    return (lv->sz>0&&lv->cz>0)?0:-1;
}

/* ---------------------------------------------------- occupancy (coarse level) */
typedef struct { u8 *present; long ncz,ncy,ncx; } chunkmap;
/* parallel coarse-level fetch state (startup was serial-GET bound) */
static struct { const zlvl *cl; u8 *buf; long nclz,ncly,nclx; atomic_long next; } g_occ;
static void *occ_fetch_worker(void *arg){
    (void)arg;
    const zlvl *cl=g_occ.cl;
    long total=g_occ.nclz*g_occ.ncly*g_occ.nclx;
    for(;;){
        long i=atomic_fetch_add(&g_occ.next,1);
        if(i>=total) break;
        long qz=i/(g_occ.ncly*g_occ.nclx), r=i%(g_occ.ncly*g_occ.nclx), qy=r/g_occ.nclx, qx=r%g_occ.nclx;
        char p[1500]; snprintf(p,sizeof p,"%s/%ld/%ld/%ld",cl->dir,qz,qy,qx);
        size_t gn=0; u8 *got=src_get(p,&gn,NULL);
        if(!got) continue;
        long gz0=qz*cl->cz, gy0=qy*cl->cy, gx0=qx*cl->cx;
        for(long z=0;z<cl->cz&&gz0+z<cl->sz;++z)for(long y=0;y<cl->cy&&gy0+y<cl->sy;++y){
            long w=cl->cx; if(gx0+w>cl->sx)w=cl->sx-gx0;
            size_t so=((size_t)z*cl->cy+y)*cl->cx;
            if(so<gn) memcpy(g_occ.buf+((size_t)(gz0+z)*cl->sy+(gy0+y))*cl->sx+gx0, got+so,
                             (size_t)w<gn-so?(size_t)w:gn-so);
        }
        free(got);
    }
    return NULL;
}
static int cm_from_coarse(chunkmap *cm,const char *zarr,const zlvl *z0){
    int bestL=-1; zlvl cl; long f=1;
    for(int L=5;L>=1;--L){
        long ff=1L<<L;
        if(z0->cz%ff||z0->cy%ff||z0->cx%ff) continue;
        zlvl t; if(parse_lvl(zarr,L,&t)!=0) continue;
        cl=t; f=ff; bestL=L; break;
    }
    if(bestL<0) return -1;
    long vcz=z0->cz/f, vcy=z0->cy/f, vcx=z0->cx/f;
    size_t voxn=(size_t)cl.sz*cl.sy*cl.sx;
    u8 *buf=calloc(voxn,1); if(!buf) return -1;
    long nclz=(cl.sz+cl.cz-1)/cl.cz, ncly=(cl.sy+cl.cy-1)/cl.cy, nclx=(cl.sx+cl.cx-1)/cl.cx;
    /* PARALLEL fetch of the coarse level (16 downloaders; each chunk scatters to a
     * disjoint region of buf, so no locking) -- this was a serial startup bottleneck. */
    {
        g_occ.cl=&cl; g_occ.buf=buf; g_occ.nclz=nclz; g_occ.ncly=ncly; g_occ.nclx=nclx;
        atomic_store(&g_occ.next,0);
        pthread_t th[16];
        for(int i=0;i<16;i++) pthread_create(&th[i],NULL,occ_fetch_worker,NULL);
        for(int i=0;i<16;i++) pthread_join(th[i],NULL);
    }
    for(long cz=0;cz<cm->ncz;++cz)for(long cy=0;cy<cm->ncy;++cy)for(long cx=0;cx<cm->ncx;++cx){
        long z0v=cz*vcz,y0v=cy*vcy,x0v=cx*vcx; int nz=0;
        for(long z=z0v;z<z0v+vcz&&z<cl.sz&&!nz;++z)
          for(long y=y0v;y<y0v+vcy&&y<cl.sy&&!nz;++y)
            for(long x=x0v;x<x0v+vcx&&x<cl.sx;++x)
              if(buf[((size_t)z*cl.sy+y)*cl.sx+x]){nz=1;break;}
        if(nz) cm->present[((size_t)cz*cm->ncy+cy)*cm->ncx+cx]=1;
    }
    free(buf);
    fprintf(stderr,"occupancy from coarse level %d/ (factor %ld)\n",bestL,f);
    return 0;
}
static int cm_build(chunkmap *cm,const char *zarr,const zlvl *z0){
    cm->ncz=(z0->sz+z0->cz-1)/z0->cz; cm->ncy=(z0->sy+z0->cy-1)/z0->cy; cm->ncx=(z0->sx+z0->cx-1)/z0->cx;
    cm->present=calloc((size_t)cm->ncz*cm->ncy*cm->ncx,1);
    if(!cm->present) return -1;
    if(cm_from_coarse(cm,zarr,z0)==0) return 0;
    fprintf(stderr,"no coarse level; assuming all chunks present\n");
    memset(cm->present,1,(size_t)cm->ncz*cm->ncy*cm->ncx);
    return 0;
}

/* ------------------------------------------------------- shared chunk cache
 * Refcounted LRU keyed by chunk coord. Units overlap at their halo boundaries:
 * z-neighbor bands run consecutively and +x-neighbor columns run nbz units
 * later, so a modest cache turns the ~1.4x halo re-fetch into ~1.05x. */
typedef struct centry { long key; u8 *bytes; size_t len; int refs; long lru;
                        struct centry *next; } centry;
typedef struct {
    centry **bk; long nbk; size_t bytes, cap; long tick;
    long hits, misses;
    pthread_mutex_t m;
} ccache;
static ccache g_cc;
static void cc_init(size_t cap){ memset(&g_cc,0,sizeof g_cc); g_cc.cap=cap;
    g_cc.nbk=1<<16; g_cc.bk=calloc(g_cc.nbk,sizeof(centry*)); pthread_mutex_init(&g_cc.m,NULL); }
static long cc_key(long z,long y,long x){ return ((z*1000003L)+y)*1000003L+x; }
static void cc_evict_locked(void){
    while(g_cc.bytes>g_cc.cap){
        centry *best=NULL,**bp=NULL;
        for(long i=0;i<g_cc.nbk;i++) for(centry **p=&g_cc.bk[i];*p;p=&(*p)->next)
            if((*p)->refs==0 && (!best||(*p)->lru<best->lru)){ best=*p; bp=p; }
        if(!best) break;
        *bp=best->next; g_cc.bytes-=best->len; free(best->bytes); free(best);
    }
}
/* lookup: returns entry with a ref taken, or NULL */
static centry *cc_get(long z,long y,long x){
    long k=cc_key(z,y,x); centry *e;
    pthread_mutex_lock(&g_cc.m);
    for(e=g_cc.bk[(unsigned long)k%g_cc.nbk];e;e=e->next) if(e->key==k) break;
    if(e){ e->refs++; e->lru=++g_cc.tick; g_cc.hits++; } else g_cc.misses++;
    pthread_mutex_unlock(&g_cc.m);
    return e;
}
/* insert fetched bytes (takes ownership), returns entry with one ref */
static centry *cc_put(long z,long y,long x,u8 *bytes,size_t len){
    long k=cc_key(z,y,x);
    centry *e=malloc(sizeof *e); e->key=k; e->bytes=bytes; e->len=len; e->refs=1;
    pthread_mutex_lock(&g_cc.m);
    e->lru=++g_cc.tick;
    unsigned long b=(unsigned long)k%g_cc.nbk; e->next=g_cc.bk[b]; g_cc.bk[b]=e;
    g_cc.bytes+=len; cc_evict_locked();
    pthread_mutex_unlock(&g_cc.m);
    return e;
}
static void cc_release(centry *e){
    pthread_mutex_lock(&g_cc.m); e->refs--; pthread_mutex_unlock(&g_cc.m);
}

/* ------------------------------------------------------------ band queue */
typedef struct { long ccz,ccy,ccx; u8 *bytes; size_t len; centry *ce; int fetched; } cchunk;
typedef struct { long ti; cchunk *chunks; int nch; int empty; } bandbuf;
typedef struct {
    bandbuf *slot; int cap,head,tail,count,closed,active_prod;
    pthread_mutex_t m; pthread_cond_t not_full,not_empty;
} bandq;
static void bq_init(bandq *q,int cap,int nprod){
    q->slot=calloc(cap,sizeof *q->slot); q->cap=cap; q->head=q->tail=q->count=0;
    q->closed=0; q->active_prod=nprod;
    pthread_mutex_init(&q->m,NULL); pthread_cond_init(&q->not_full,NULL); pthread_cond_init(&q->not_empty,NULL);
}
static void bq_push(bandq *q,bandbuf b){
    pthread_mutex_lock(&q->m);
    while(q->count==q->cap) pthread_cond_wait(&q->not_full,&q->m);
    q->slot[q->tail]=b; q->tail=(q->tail+1)%q->cap; q->count++;
    pthread_cond_signal(&q->not_empty); pthread_mutex_unlock(&q->m);
}
static int bq_pop(bandq *q,bandbuf *out){
    pthread_mutex_lock(&q->m);
    while(q->count==0&&!q->closed) pthread_cond_wait(&q->not_empty,&q->m);
    if(q->count==0&&q->closed){pthread_mutex_unlock(&q->m);return 0;}
    *out=q->slot[q->head]; q->head=(q->head+1)%q->cap; q->count--;
    pthread_cond_signal(&q->not_full); pthread_mutex_unlock(&q->m);
    return 1;
}
static void bq_prod_done(bandq *q){
    pthread_mutex_lock(&q->m);
    if(--q->active_prod==0){q->closed=1;pthread_cond_broadcast(&q->not_empty);}
    pthread_mutex_unlock(&q->m);
}

/* ------------------------------------------------------------ scheduler */
typedef struct {
    const zlvl *z0; const chunkmap *cm;
    const fy_pipeline_cfg *cfg; int process;
    mc_archive *arc;
    long SB,BAND,halo;
    long ntx,nty,nbz,nunits;
    long pz,py,px;              /* padded dims (multiples of 2*MCC for L1 alignment) */
    atomic_long next, fail, skipped, l0app, l1app;
    atomic_long io_open, io_miss, io_bytes;
    /* in-flight fetched-bytes budget: downloaders block before starting a new band
     * while over budget (bounded RAM regardless of io-thread count; the queue cap
     * alone does NOT bound the bands being fetched). */
    atomic_long inflight;
    long inflight_cap;
    pthread_mutex_t im; pthread_cond_t icv;
    bandq q;
} sched;

static void *download_worker(void *arg){
    sched *sc=arg; const zlvl *z0=sc->z0;
    for(;;){
        long ti=atomic_fetch_add(&sc->next,1);
        if(ti>=sc->nunits) break;
        long band_bytes=0;   /* gate per CHUNK below (per-band gating lets N
                              * downloaders overshoot the cap N-fold before any
                              * accounting); band_bytes>0 guarantees progress */
        long bz=ti%sc->nbz, txy=ti/sc->nbz, ty=txy/sc->ntx, tx=txy%sc->ntx;
        long vx0=tx*sc->SB, vy0=ty*sc->SB, vz0=bz*sc->BAND;
        long vx1=vx0+sc->SB, vy1=vy0+sc->SB, vz1=vz0+sc->BAND;
        /* halo-expanded chunk footprint, clamped to the true volume */
        long h=sc->halo;
        long fx0=vx0-h<0?0:vx0-h, fy0=vy0-h<0?0:vy0-h, fz0=vz0-h<0?0:vz0-h;
        long fx1=vx1+h>z0->sx?z0->sx:vx1+h, fy1=vy1+h>z0->sy?z0->sy:vy1+h, fz1=vz1+h>z0->sz?z0->sz:vz1+h;
        cchunk *chunks=NULL; int nch=0,cap=0,any=0;
        if(fx0<fx1&&fy0<fy1&&fz0<fz1){
            long cxa=fx0/z0->cx,cxb=(fx1-1)/z0->cx, cya=fy0/z0->cy,cyb=(fy1-1)/z0->cy;
            long cza=fz0/z0->cz,czb=(fz1-1)/z0->cz;
            for(long qz=cza;qz<=czb;++qz)for(long qy=cya;qy<=cyb;++qy)for(long qx=cxa;qx<=cxb;++qx){
                if(!sc->cm->present[((size_t)qz*sc->cm->ncy+qy)*sc->cm->ncx+qx]) continue;
                /* `any` tracks the CORE (the unit's own chunks), not halo chunks */
                int core = (qz>=vz0/z0->cz && qz*z0->cz<vz1 && qy>=vy0/z0->cy && qy*z0->cy<vy1
                            && qx>=vx0/z0->cx && qx*z0->cx<vx1);
                centry *ce=cc_get(qz,qy,qx);
                u8 *got; size_t gn; int fetched=0;
                if(ce){ got=ce->bytes; gn=ce->len; if(core) any=1; }
                else {
                    pthread_mutex_lock(&sc->im);
                    while(band_bytes>0 && atomic_load(&sc->inflight)>sc->inflight_cap)
                        pthread_cond_wait(&sc->icv,&sc->im);
                    pthread_mutex_unlock(&sc->im);
                    fetched=1;
                    char p[1500]; snprintf(p,sizeof p,"%s/%ld/%ld/%ld",z0->dir,qz,qy,qx);
                    atomic_fetch_add(&sc->io_open,1);
                    gn=0; int err=0; got=src_get(p,&gn,&err);
                    if(err){ atomic_fetch_add(&sc->fail,1); }
                    if(!got){ atomic_fetch_add(&sc->io_miss,1); continue; }
                    if(core) any=1;
                    atomic_fetch_add(&sc->io_bytes,(long)gn);
                    atomic_fetch_add(&sc->inflight,(long)gn);
                    band_bytes+=(long)gn;
                    ce=cc_put(qz,qy,qx,got,gn);
                }
                if(nch==cap){cap=cap?cap*2:32;chunks=realloc(chunks,cap*sizeof *chunks);}
                chunks[nch].ccz=qz;chunks[nch].ccy=qy;chunks[nch].ccx=qx;
                chunks[nch].bytes=got;chunks[nch].len=gn;chunks[nch].ce=ce;chunks[nch].fetched=fetched;nch++;
            }
        }
        bandbuf b={ti,chunks,nch,any?0:1};
        bq_push(&sc->q,b);
    }
    bq_prod_done(&sc->q);
    return NULL;
}

/* 2x2x2 box downscale (zero-preserving rounding like the zarr pyramid) */
static void down2x(const u8 *in,long dz,long dy,long dx,u8 *out,long oz,long oy,long ox){
    for(long z=0;z<oz;++z)for(long y=0;y<oy;++y)for(long x=0;x<ox;++x){
        int s=0,c=0;
        for(int a=0;a<2;a++)for(int b=0;b<2;b++)for(int d=0;d<2;d++){
            long zz=2*z+a,yy=2*y+b,xx=2*x+d;
            if(zz<dz&&yy<dy&&xx<dx){s+=in[((size_t)zz*dy+yy)*dx+xx];c++;}
        }
        out[((size_t)z*oy+y)*ox+x]=(u8)(c?(s+c/2)/c:0);
    }
}

static void *compute_worker(void *arg){
    sched *sc=arg; const zlvl *z0=sc->z0;
    long SB=sc->SB, BAND=sc->BAND, h=sc->halo;
    size_t rawcap=(size_t)(SB+2*h)*(SB+2*h)*(BAND+2*h);
    u8 *raw=malloc(rawcap);                                   /* halo'd input band */
    u8 *proc=malloc((size_t)SB*SB*BAND);                      /* processed band    */
    u8 *l1=malloc((size_t)(SB/2)*(SB/2)*(BAND/2));            /* downscaled        */
    u8 *cbuf=malloc((size_t)MCC*MCC*MCC);                     /* chunk gather      */
    u8 *tin=malloc((size_t)(128+2*h)*(128+2*h)*(128+2*h));    /* fysics tile in    */
    u8 *tout=malloc((size_t)128*128*128);                     /* fysics tile out   */
    if(!raw||!proc||!l1||!cbuf||!tin||!tout){ fprintf(stderr,"compute alloc failed\n"); exit(1); }
    bandbuf bb;
    while(bq_pop(&sc->q,&bb)){
        long ti=bb.ti;
        long bz=ti%sc->nbz, txy=ti/sc->nbz, ty=txy/sc->ntx, tx=txy%sc->ntx;
        long vx0=tx*SB, vy0=ty*SB, vz0=bz*BAND;
        if(bb.empty){ atomic_fetch_add(&sc->skipped,1); goto release; }
        {
        /* assemble the halo'd raw band (clamped to the volume) */
        long rz0=vz0-h<0?0:vz0-h, ry0=vy0-h<0?0:vy0-h, rx0=vx0-h<0?0:vx0-h;
        long rz1=vz0+BAND+h>z0->sz?z0->sz:vz0+BAND+h;
        long ry1=vy0+SB+h>z0->sy?z0->sy:vy0+SB+h;
        long rx1=vx0+SB+h>z0->sx?z0->sx:vx0+SB+h;
        long hz=rz1-rz0, hy=ry1-ry0, hx=rx1-rx0;
        memset(raw,0,(size_t)hz*hy*hx);
        for(int i=0;i<bb.nch;++i){
            const cchunk *c=&bb.chunks[i];
            long gz0=c->ccz*z0->cz, gy0=c->ccy*z0->cy, gx0=c->ccx*z0->cx;
            long oz0=gz0>rz0?gz0:rz0, oz1=gz0+z0->cz<rz1?gz0+z0->cz:rz1;
            long oy0=gy0>ry0?gy0:ry0, oy1=gy0+z0->cy<ry1?gy0+z0->cy:ry1;
            long ox0=gx0>rx0?gx0:rx0, ox1=gx0+z0->cx<rx1?gx0+z0->cx:rx1;
            /* TRUE stored extents (edge chunks are stored smaller) */
            long ez=z0->sz-gz0<z0->cz?z0->sz-gz0:z0->cz;
            long ey=z0->sy-gy0<z0->cy?z0->sy-gy0:z0->cy;
            long ex=z0->sx-gx0<z0->cx?z0->sx-gx0:z0->cx;
            for(long z=oz0;z<oz1;++z)for(long y=oy0;y<oy1;++y){
                size_t so=((size_t)(z-gz0)*ey+(y-gy0))*ex+(ox0-gx0);
                size_t want=ox1-ox0;
                u8 *dst=raw+((size_t)(z-rz0)*hy+(y-ry0))*hx+(ox0-rx0);
                if(so>=c->len) continue;
                size_t avail=c->len-so, n=want<avail?want:avail;
                memcpy(dst,c->bytes+so,n);
            }
        }
        /* process (or copy) 128^3 tiles into the processed band */
        memset(proc,0,(size_t)SB*SB*BAND);
        for(long tz0=vz0; tz0<vz0+BAND; tz0+=128)
        for(long ty0=vy0; ty0<vy0+SB; ty0+=128)
        for(long tx0=vx0; tx0<vx0+SB; tx0+=128){
            if(tz0>=z0->sz||ty0>=z0->sy||tx0>=z0->sx) continue;
            long tzn=z0->sz-tz0<128?z0->sz-tz0:128;
            long tyn=z0->sy-ty0<128?z0->sy-ty0:128;
            long txn=z0->sx-tx0<128?z0->sx-tx0:128;
            /* halo'd tile box (clamped), copied contiguous from the raw band */
            long az0=tz0-h<rz0?rz0:tz0-h, ay0=ty0-h<ry0?ry0:ty0-h, ax0=tx0-h<rx0?rx0:tx0-h;
            long az1=tz0+tzn+h>rz1?rz1:tz0+tzn+h;
            long ay1=ty0+tyn+h>ry1?ry1:ty0+tyn+h;
            long ax1=tx0+txn+h>rx1?rx1:tx0+txn+h;
            long bzn=az1-az0, byn=ay1-ay0, bxn=ax1-ax0;
            for(long z=0;z<bzn;++z)for(long y=0;y<byn;++y)
                memcpy(tin+((size_t)z*byn+y)*bxn,
                       raw+((size_t)(az0-rz0+z)*hy+(ay0-ry0+y))*hx+(ax0-rx0), bxn);
            int air=0;
            if(sc->process){
                if(fy_process_buffer(sc->cfg,tin,az0,ay0,ax0,bzn,byn,bxn,
                                     tz0,ty0,tx0,tzn,tyn,txn,z0->sy,z0->sx,tout,&air)!=0){
                    atomic_fetch_add(&sc->fail,1); continue;
                }
            } else {
                for(long z=0;z<tzn;++z)for(long y=0;y<tyn;++y)
                    memcpy(tout+((size_t)z*tyn+y)*txn,
                           tin+((size_t)(tz0-az0+z)*byn+(ty0-ay0+y))*bxn+(tx0-ax0), txn);
            }
            for(long z=0;z<tzn;++z)for(long y=0;y<tyn;++y)
                memcpy(proc+((size_t)(tz0-vz0+z)*SB+(ty0-vy0+y))*SB+(tx0-vx0),
                       tout+((size_t)z*tyn+y)*txn, txn);
        }
        /* append L0 chunks */
        for(long az=0;az<BAND/MCC;++az)for(long ay=0;ay<SB/MCC;++ay)for(long ax=0;ax<SB/MCC;++ax){
            for(long z=0;z<MCC;++z)for(long y=0;y<MCC;++y)
                memcpy(cbuf+((size_t)z*MCC+y)*MCC,
                       proc+((size_t)(az*MCC+z)*SB+(ay*MCC+y))*SB+ax*MCC, MCC);
            if(mc_archive_append_chunk_raw(sc->arc,0,vz0/MCC+az,vy0/MCC+ay,vx0/MCC+ax,cbuf)!=0)
                atomic_fetch_add(&sc->fail,1);
            else atomic_fetch_add(&sc->l0app,1);
        }
        /* L1: 2x downscale the processed band, append */
        down2x(proc,BAND,SB,SB,l1,BAND/2,SB/2,SB/2);
        for(long ay=0;ay<SB/2/MCC;++ay)for(long ax=0;ax<SB/2/MCC;++ax){
            for(long z=0;z<MCC;++z)for(long y=0;y<MCC;++y)
                memcpy(cbuf+((size_t)z*MCC+y)*MCC,
                       l1+((size_t)z*(SB/2)+(ay*MCC+y))*(SB/2)+ax*MCC, MCC);
            if(mc_archive_append_chunk_raw(sc->arc,1,vz0/2/MCC,vy0/2/MCC+ay,vx0/2/MCC+ax,cbuf)!=0)
                atomic_fetch_add(&sc->fail,1);
            else atomic_fetch_add(&sc->l1app,1);
        }
        }
release:
        { long rel=0;
          for(int i=0;i<bb.nch;++i){ if(bb.chunks[i].fetched) rel+=(long)bb.chunks[i].len; cc_release(bb.chunks[i].ce); }
          atomic_fetch_sub(&sc->inflight,rel);
          pthread_mutex_lock(&sc->im); pthread_cond_broadcast(&sc->icv); pthread_mutex_unlock(&sc->im); }
        free(bb.chunks);
    }
    free(raw);free(proc);free(l1);free(cbuf);free(tin);free(tout);
    return NULL;
}

/* coarse tail: level L from level L-1 via the archive itself */
typedef struct { mc_archive *arc; int L; long ncz,ncy,ncx; atomic_long next, fail; } tailctx;
static void *tail_worker(void *arg){
    tailctx *t=arg;
    u8 *src=malloc((size_t)2*MCC*2*MCC*2*MCC), *dst=malloc((size_t)MCC*MCC*MCC);
    long n=t->ncz*t->ncy*t->ncx;
    for(;;){
        long i=atomic_fetch_add(&t->next,1);
        if(i>=n) break;
        long cz=i/(t->ncy*t->ncx), r=i%(t->ncy*t->ncx), cy=r/t->ncx, cx=r%t->ncx;
        mc_archive_read_region(t->arc,t->L-1,cz*2L*MCC,cy*2L*MCC,cx*2L*MCC,
                               2*MCC,2*MCC,2*MCC,src,(size_t)2*MCC*2*MCC,(size_t)2*MCC,1);
        down2x(src,2*MCC,2*MCC,2*MCC,dst,MCC,MCC,MCC);
        if(mc_archive_append_chunk_raw(t->arc,t->L,cz,cy,cx,dst)!=0)
            atomic_fetch_add(&t->fail,1);
    }
    free(src);free(dst);
    return NULL;
}

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
    if(argc<3){
        fprintf(stderr,"usage: %s <in_zarr|s3://...> <out.mc> [--profile P] [--quality Q]"
                       " [--meta PATH] [--threads N] [--io-threads M] [--queue C]"
                       " [--sb SB] [--band B] [--no-process]\n",argv[0]);
        return 2;
    }
    const char *in=argv[1], *out=argv[2], *profile="conservative";
    float quality=8.0f;
    long SB=1024, BAND=512;
    int nthreads=0, niothreads=0, qcap=0, process=1; double cache_gb=12.0, mem_gb=24.0;
    char meta_path[PATH_MAX]; meta_path[0]=0;
    for(int i=3;i<argc;i++){
        if      (!strcmp(argv[i],"--profile")&&i+1<argc)   profile=argv[++i];
        else if (!strcmp(argv[i],"--quality")&&i+1<argc)   quality=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--meta")&&i+1<argc)      snprintf(meta_path,sizeof meta_path,"%s",argv[++i]);
        else if (!strcmp(argv[i],"--threads")&&i+1<argc)   nthreads=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--io-threads")&&i+1<argc)niothreads=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--queue")&&i+1<argc)     qcap=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--sb")&&i+1<argc)        SB=atol(argv[++i]);
        else if (!strcmp(argv[i],"--band")&&i+1<argc)      BAND=atol(argv[++i]);
        else if (!strcmp(argv[i],"--no-process"))          process=0;
        else if (!strcmp(argv[i],"--cache-gb")&&i+1<argc)  cache_gb=atof(argv[++i]);
        else if (!strcmp(argv[i],"--mem-gb")&&i+1<argc)    mem_gb=atof(argv[++i]);
        else { fprintf(stderr,"unknown arg: %s\n",argv[i]); return 2; }
    }
    if(SB%512||SB<512){fprintf(stderr,"--sb must be a multiple of 512\n");return 2;}
    if(BAND%512||BAND<512){fprintf(stderr,"--band must be a multiple of 512\n");return 2;}
    if(!meta_path[0]) snprintf(meta_path,sizeof meta_path,"%s/metadata.json",in);
    if(is_s3(in)) s3_init_once();

    /* ---- fysics config: BM18 defaults + profile + metadata ---- */
    fy_pipeline_cfg cfg; memset(&cfg,0,sizeof cfg);
    cfg.delta_beta=1000.0; cfg.energy_kev=78.0; cfg.distance_mm=220.0; cfg.pixel_um=2.4;
    cfg.unsharp_sigma=1.2; cfg.unsharp_coeff=4.0;
    cfg.do_deconv=0; cfg.use_matched_deconv=1; cfg.psf_sigma_vox=1.0; cfg.deconv_tikhonov=0.05;
    cfg.do_air_zero=1; cfg.air_cut_u8=-1; cfg.air_cut_band=8; cfg.air_thresh=0.05;
    cfg.scratch_passes=5;
    cfg.do_normalize=1; cfg.norm_lo=-1; cfg.norm_hi=-1;
    cfg.do_zdrift=1; cfg.do_dering=1; cfg.dering_cy=-1; cfg.dering_cx=-1;
    if      (!strcmp(profile,"conservative")) { cfg.air_cut_aggr=0.0; cfg.denoise_k=3.0; }
    else if (!strcmp(profile,"aggressive"))   { cfg.air_cut_aggr=1.0; cfg.denoise_k=4.2; }
    else { fprintf(stderr,"unknown profile %s\n",profile); return 2; }
    /* metadata via the python reader (same locator as fysics_process) */
    {
        char script[PATH_MAX]={0}, exe[PATH_MAX]; ssize_t en=readlink("/proc/self/exe",exe,sizeof exe-1);
        char exedir[PATH_MAX]={0}; if(en>0){exe[en]=0;snprintf(exedir,sizeof exedir,"%s",dirname(exe));}
        const char *cands[4]; int nc=0; char c0[PATH_MAX],c1[PATH_MAX];
        const char *env=getenv("FYSICS_READMETA"); if(env)cands[nc++]=env;
        if(exedir[0]){snprintf(c0,sizeof c0,"%s/read_meta.py",exedir);cands[nc++]=c0;
                      snprintf(c1,sizeof c1,"%s/../tools/read_meta.py",exedir);cands[nc++]=c1;}
        cands[nc++]="read_meta.py";
        for(int i=0;i<nc;i++) if(access(cands[i],R_OK)==0){snprintf(script,sizeof script,"%s",cands[i]);break;}
        if(!script[0]) snprintf(script,sizeof script,"read_meta.py");
        char cmd[PATH_MAX*2]; snprintf(cmd,sizeof cmd,"python3 '%s' '%s' 2>/dev/null",script,meta_path);
        FILE *pp=popen(cmd,"r"); int nmeta=0;
        if(pp){char line[256];while(fgets(line,sizeof line,pp)){char*eq=strchr(line,'=');if(!eq)continue;
               *eq=0;apply_meta(&cfg,line,atof(eq+1));nmeta++;}pclose(pp);}
        if(cfg.window_hi>cfg.window_lo){double af=(0.0-cfg.window_lo)/(cfg.window_hi-cfg.window_lo);
            if(af<0)af=0;if(af>1)af=1;cfg.air_thresh=af;}
        fprintf(stderr,"vca_export: %d meta keys from %s\n",nmeta,meta_path);
    }

    /* ---- parse level 0 + occupancy ---- */
    zlvl z0;
    if(parse_lvl(in,0,&z0)!=0){fprintf(stderr,"cannot parse %s/0/.zarray\n",in);return 1;}
    fprintf(stderr,"input %ldx%ldx%ld chunks %ldx%ldx%ld\n",z0.sz,z0.sy,z0.sx,z0.cz,z0.cy,z0.cx);
    chunkmap cm;
    if(cm_build(&cm,in,&z0)!=0){fprintf(stderr,"occupancy build failed\n");return 1;}

    /* ---- calibrate (reads the zarr through the fysics library) ---- */
    if(process){
        if(fy_calibrate(in,&cfg,128,1)!=0){fprintf(stderr,"calibrate failed\n");return 1;}
    }

    /* ---- archive + schedule ---- */
    mc_codec_init();
    mc_archive *arc=mc_archive_open_dims(out,(int)z0.sx,(int)z0.sy,(int)z0.sz,quality);
    if(!arc){fprintf(stderr,"cannot open %s\n",out);return 1;}
    sched sc; memset(&sc,0,sizeof sc);
    sc.z0=&z0; sc.cm=&cm; sc.cfg=&cfg; sc.process=process; sc.arc=arc;
    sc.SB=SB; sc.BAND=BAND; sc.halo=cfg.halo>0?cfg.halo:8;
    sc.px=((z0.sx+2*MCC-1)/(2*MCC))*(2*MCC);
    sc.py=((z0.sy+2*MCC-1)/(2*MCC))*(2*MCC);
    sc.pz=((z0.sz+2*MCC-1)/(2*MCC))*(2*MCC);
    sc.ntx=(sc.px+SB-1)/SB; sc.nty=(sc.py+SB-1)/SB; sc.nbz=(sc.pz+BAND-1)/BAND;
    sc.nunits=sc.ntx*sc.nty*sc.nbz;
    long ncpu=sysconf(_SC_NPROCESSORS_ONLN); if(ncpu<1)ncpu=4;
    int nc_=nthreads>0?nthreads:(int)(ncpu<8?ncpu:8);
    int ni_=niothreads>0?niothreads:(is_s3(in)?nc_*2:(nc_<4?nc_:4));
    int qc_=qcap>0?qcap:nc_+2;
    cc_init((size_t)(cache_gb*1e9));
    sc.inflight_cap=(long)(mem_gb*1e9); atomic_store(&sc.inflight,0);
    pthread_mutex_init(&sc.im,NULL); pthread_cond_init(&sc.icv,NULL);
    bq_init(&sc.q,qc_,ni_);
    fprintf(stderr,"units %ld (%ldx%ld tiles x %ld bands), SB=%ld BAND=%ld halo=%ld, "
                   "%d compute + %d io threads, queue %d\n",
            sc.nunits,sc.ntx,sc.nty,sc.nbz,SB,BAND,sc.halo,nc_,ni_,qc_);

    pthread_t *iot=malloc(ni_*sizeof *iot), *cot=malloc(nc_*sizeof *cot);
    for(int i=0;i<ni_;i++) pthread_create(&iot[i],NULL,download_worker,&sc);
    for(int i=0;i<nc_;i++) pthread_create(&cot[i],NULL,compute_worker,&sc);
    for(int i=0;i<ni_;i++) pthread_join(iot[i],NULL);
    for(int i=0;i<nc_;i++) pthread_join(cot[i],NULL);
    free(iot);free(cot);
    fprintf(stderr,"chunk cache: %ld hits / %ld misses (%.0f%% re-fetch avoided)\n",
            g_cc.hits,g_cc.misses,g_cc.hits+g_cc.misses?100.0*g_cc.hits/(g_cc.hits+g_cc.misses):0.0);
    fprintf(stderr,"stream done: %ld L0 + %ld L1 chunks, %ld empty units skipped, "
                   "%ld GETs (%ld absent, %.1f GB)\n",
            atomic_load(&sc.l0app),atomic_load(&sc.l1app),atomic_load(&sc.skipped),
            atomic_load(&sc.io_open),atomic_load(&sc.io_miss),atomic_load(&sc.io_bytes)/1e9);
    if(atomic_load(&sc.fail)){
        fprintf(stderr,"ERROR: %ld unit failures -- archive is INCOMPLETE\n",atomic_load(&sc.fail));
        mc_archive_close(arc); return 1;
    }

    /* ---- coarse tail L2.. from the archive ---- */
    for(int L=2;L<8;L++){
        long dz=(z0.sz>>L), dy=(z0.sy>>L), dx=(z0.sx>>L);
        if(dz<1&&dy<1&&dx<1) break;
        tailctx t={arc,L,(dz+MCC-1)/MCC?:(long)1,(dy+MCC-1)/MCC?:(long)1,(dx+MCC-1)/MCC?:(long)1,0,0};
        long n=t.ncz*t.ncy*t.ncx;
        int tw=(int)(n<nc_?(n>0?n:1):nc_);
        pthread_t *tt=malloc(tw*sizeof *tt);
        for(int i=0;i<tw;i++) pthread_create(&tt[i],NULL,tail_worker,&t);
        for(int i=0;i<tw;i++) pthread_join(tt[i],NULL);
        free(tt);
        fprintf(stderr,"tail L%d: %ld chunks\n",L,n);
        if(atomic_load(&t.fail)){fprintf(stderr,"ERROR: tail failures\n");mc_archive_close(arc);return 1;}
    }
    mc_archive_close(arc);
    free(cfg.zdrift_factor);
    if(cfg.dering){fy_dering_free(cfg.dering);free(cfg.dering);}
    cm.present?free(cm.present):(void)0;
    fprintf(stderr,"vca_export: wrote %s\n",out);
    return 0;
}
