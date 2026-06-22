/* mca_export.c -- STREAMING single-pass export: uncompressed OME-zarr (local or s3://)
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
 * Usage: mca_export <in_zarr|s3://...> <out.mc> [--profile conservative|aggressive]
 *        [--quality Q] [--meta PATH] [--threads N] [--io-threads M] [--queue C]
 *        [--sb SB] [--band B] [--no-process]
 */
#define _GNU_SOURCE
#include "fysics.h"
#include "matter_compressor.h"
#include "libs3.h"
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
#include <malloc.h>
#include <limits.h>
#include <time.h>
#include <signal.h>
#include <execinfo.h>

/* crash diagnostic: dump a backtrace on fatal signals (the streaming pipeline
 * runs many threads; a silent SIGSEGV otherwise leaves no trace). async-signal
 * unsafe in the strict sense, but acceptable for a post-mortem stderr dump. */
static void fy_crash_handler(int sig){
    void *bt[64]; int n=backtrace(bt,64);
    fprintf(stderr,"\n==== FATAL SIGNAL %d (%s) ====\n",sig,strsignal(sig));
    backtrace_symbols_fd(bt,n,2);
    signal(sig,SIG_DFL); raise(sig);
}

typedef uint8_t u8;
#define MCC 256                 /* mc chunk edge */

/* ------------------------------------------------------------------ source I/O */
static s3_client *g_s3 = NULL;
static int is_s3(const char *p){ return strncmp(p,"s3://",5)==0; }
/* per-batch connection count (env MCA_BATCH, default 16). Total in-flight S3
 * connections ~= io_threads * this; keep it modest so connections are REUSED
 * (keep-alive) rather than churned -- churn triggers handshake resets. */
static int mca_batch_conc(void){ static int v=0; if(!v){const char*e=getenv("MCA_BATCH"); v=e?atoi(e):16; if(v<1)v=16;} return v; }
static void s3_init_once(void){
    if(g_s3) return;
    s3_config cfg; memset(&cfg,0,sizeof cfg);
    cfg.max_retries=5;
    /* Vesuvius open-data buckets are PUBLIC: use ANONYMOUS (unsigned) requests --
     * no cred_provider, so libs3 issues unsigned GETs (avoids SSO / expired-cred
     * signing failures on a public bucket). Just pin the region. */
    const char *reg = getenv("AWS_REGION"); if(!reg||!reg[0]) reg = getenv("AWS_DEFAULT_REGION");
    cfg.region = (reg && reg[0]) ? reg : "us-east-1";
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
/* content bbox (world LOD0, hi EXCLUSIVE) into bb[6]={zlo,zhi,ylo,yhi,xlo,xhi};
 * filled by cm_from_coarse from the coarse-level nonzero extent. NULL = don't compute. */
static int cm_from_coarse(chunkmap *cm,const char *zarr,const zlvl *z0,long *bb,int base){
    int bestL=-1; zlvl cl; long f=1;
    /* search ABSOLUTE pyramid levels COARSER than the base (base+1..base+5); the
     * occupancy/ROI factor is RELATIVE to the base level. */
    for(int L=base+5;L>base;--L){
        long ff=1L<<(L-base);
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
    if(bb){
        /* tight nonzero bbox in coarse voxels, scaled to LOD0 (1 coarse voxel = f LOD0). */
        long czmin=cl.sz,czmax=-1,cymin=cl.sy,cymax=-1,cxmin=cl.sx,cxmax=-1;
        for(long z=0;z<cl.sz;++z)for(long y=0;y<cl.sy;++y)for(long x=0;x<cl.sx;++x)
            if(buf[((size_t)z*cl.sy+y)*cl.sx+x]){
                if(z<czmin)czmin=z; if(z>czmax)czmax=z;
                if(y<cymin)cymin=y; if(y>cymax)cymax=y;
                if(x<cxmin)cxmin=x; if(x>cxmax)cxmax=x;
            }
        if(czmax<0){ bb[0]=bb[1]=bb[2]=bb[3]=bb[4]=bb[5]=0; }   /* empty volume */
        else {
            bb[0]=czmin*f; bb[1]=(czmax+1)*f>z0->sz?z0->sz:(czmax+1)*f;
            bb[2]=cymin*f; bb[3]=(cymax+1)*f>z0->sy?z0->sy:(cymax+1)*f;
            bb[4]=cxmin*f; bb[5]=(cxmax+1)*f>z0->sx?z0->sx:(cxmax+1)*f;
        }
    }
    free(buf);
    fprintf(stderr,"occupancy from coarse level %d/ (factor %ld)\n",bestL,f);
    return 0;
}
static int cm_build(chunkmap *cm,const char *zarr,const zlvl *z0,long *bb,int base){
    cm->ncz=(z0->sz+z0->cz-1)/z0->cz; cm->ncy=(z0->sy+z0->cy-1)/z0->cy; cm->ncx=(z0->sx+z0->cx-1)/z0->cx;
    cm->present=calloc((size_t)cm->ncz*cm->ncy*cm->ncx,1);
    if(!cm->present) return -1;
    if(cm_from_coarse(cm,zarr,z0,bb,base)==0) return 0;
    fprintf(stderr,"no coarse level; assuming all chunks present (no ROI trim)\n");
    memset(cm->present,1,(size_t)cm->ncz*cm->ncy*cm->ncx);
    if(bb){ bb[0]=0;bb[1]=z0->sz; bb[2]=0;bb[3]=z0->sy; bb[4]=0;bb[5]=z0->sx; }  /* full volume = no trim */
    return 0;
}

/* ------------------------------------------------------- shared chunk cache
 * Refcounted LRU keyed by chunk coord. Units overlap at their halo boundaries:
 * z-neighbor bands run consecutively and +x-neighbor columns run nbz units
 * later, so a modest cache turns the ~1.4x halo re-fetch into ~1.05x. */
typedef struct centry { long key; u8 *bytes; size_t len; int refs; long lru;
                        int ready;          /* 0 = fetch in flight (single-flight) */
                        struct centry *next; } centry;
typedef struct {
    centry **bk; long nbk; size_t bytes, cap; long tick;   /* cap = RESIDENT budget:
        bytes is the truth (maintained under the lock for every fill/evict), so the
        fetch gate uses it directly -- no parallel counter to fall out of sync with
        shared/single-flight entries (the previous counters leaked exactly there). */
    long hits, misses;
    pthread_mutex_t m;
    pthread_cond_t cv;                      /* broadcast on every cc_fill */
} ccache;
static ccache g_cc;
static void cc_init(size_t cap){ memset(&g_cc,0,sizeof g_cc); g_cc.cap=cap;
    g_cc.nbk=1<<16; g_cc.bk=calloc(g_cc.nbk,sizeof(centry*));
    pthread_mutex_init(&g_cc.m,NULL); pthread_cond_init(&g_cc.cv,NULL); }
static long cc_key(long z,long y,long x){ return ((z*1000003L)+y)*1000003L+x; }
static void cc_evict_target_locked(size_t target){
    while(g_cc.bytes>target){
        centry *best=NULL,**bp=NULL;
        /* never evict a referenced OR pending entry (a pending entry is the
         * single-flight rendezvous point; freeing it would strand waiters) */
        for(long i=0;i<g_cc.nbk;i++) for(centry **p=&g_cc.bk[i];*p;p=&(*p)->next)
            if((*p)->refs==0 && (*p)->ready && (!best||(*p)->lru<best->lru)){ best=*p; bp=p; }
        if(!best) break;
        *bp=best->next; g_cc.bytes-=best->len; free(best->bytes); free(best);
    }
}
static void cc_evict_locked(void){ cc_evict_target_locked(g_cc.cap); }
static centry *cc_find_locked(long k){
    centry *e;
    for(e=g_cc.bk[(unsigned long)k%g_cc.nbk];e;e=e->next) if(e->key==k) break;
    return e;
}
/* lookup (phase 1): returns a READY entry with a ref taken, or NULL. Pending
 * entries are NOT returned -- the caller queues the chunk as 'needed' and
 * rendezvous happens in cc_reserve/cc_wait_ready. */
static centry *cc_get(long z,long y,long x){
    long k=cc_key(z,y,x); centry *e;
    pthread_mutex_lock(&g_cc.m);
    e=cc_find_locked(k);
    if(e&&e->ready){ e->refs++; e->lru=++g_cc.tick; g_cc.hits++; }
    else e=NULL;                /* absent or pending: counted at cc_reserve */
    pthread_mutex_unlock(&g_cc.m);
    return e;
}
/* SINGLE-FLIGHT reserve: exactly one caller per key becomes the owner.
 *   ready entry exists  -> ref'd entry, *owner=0 (use it directly)
 *   pending entry exists-> ref'd entry, *owner=0 (caller cc_wait_ready's)
 *   no entry            -> insert pending (refs=1, ready=0), *owner=1 (caller fetches+fills) */
static centry *cc_reserve(long z,long y,long x,int *owner){
    long k=cc_key(z,y,x); centry *e;
    pthread_mutex_lock(&g_cc.m);
    e=cc_find_locked(k);
    if(e){
        e->refs++; e->lru=++g_cc.tick; g_cc.hits++; *owner=0;
    } else {
        e=malloc(sizeof *e); e->key=k; e->bytes=NULL; e->len=0; e->refs=1; e->ready=0;
        e->lru=++g_cc.tick;
        unsigned long b=(unsigned long)k%g_cc.nbk; e->next=g_cc.bk[b]; g_cc.bk[b]=e;
        g_cc.misses++; *owner=1;
    }
    pthread_mutex_unlock(&g_cc.m);
    return e;
}
/* owner publishes the fetch result (takes ownership of bytes; bytes==NULL marks
 * the chunk absent/failed) and wakes every waiter */
static void cc_fill(centry *e,u8 *bytes,size_t len){
    pthread_mutex_lock(&g_cc.m);
    e->bytes=bytes; e->len=len; e->ready=1; e->lru=++g_cc.tick;
    g_cc.bytes+=len; cc_evict_locked();
    pthread_cond_broadcast(&g_cc.cv);
    pthread_mutex_unlock(&g_cc.m);
}
/* non-owner: wait for the owner's cc_fill. After return, e->bytes==NULL means
 * the owner's fetch failed/was absent (owner already did the accounting). */
/* bounded: returns 1 when ready; 0 on timeout (the owner may itself be gated --
 * the caller then fetches a private copy rather than wait forever). */
static int cc_wait_ready(centry *e){
    struct timespec ts; clock_gettime(CLOCK_REALTIME,&ts); ts.tv_sec+=2;
    pthread_mutex_lock(&g_cc.m);
    while(!e->ready)
        if(pthread_cond_timedwait(&g_cc.cv,&g_cc.m,&ts)!=0){ pthread_mutex_unlock(&g_cc.m); return 0; }
    pthread_mutex_unlock(&g_cc.m);
    return 1;
}
static void cc_release(centry *e){
    pthread_mutex_lock(&g_cc.m);
    e->refs--;
    if(g_cc.bytes>g_cc.cap) cc_evict_locked();
    pthread_cond_broadcast(&g_cc.cv);   /* wake fetch gates: resident bytes may have dropped */
    pthread_mutex_unlock(&g_cc.m);
}
/* fetch gate with a PROGRESS TOKEN. The per-band "first fetch free" rule is not
 * enough: with every downloader mid-band, their partial bands can collectively
 * pin resident bytes at the cap and ALL of them gate -> deadlock (observed live:
 * 88 sleeping threads, rx 0, RSS at the plateau). When over budget, exactly one
 * downloader claims the token and runs cap-EXEMPT until its band is queued --
 * compute then consumes it, releases bytes, and wakes the rest. */
static pthread_mutex_t g_tok = PTHREAD_MUTEX_INITIALIZER;
/* gate AND pre-reserve: the group's expected bytes are added to the budget BEFORE
 * the transfers start (the in-progress curl bodies were the last unaccounted
 * consumer: 64 threads x ~64MB in-flight responses OOM'd on top of a full budget).
 * Caller settles the reservation after the fetch via cc_settle(). */
static int cc_gate_reserve(size_t est, int have_tok){
    pthread_mutex_lock(&g_cc.m);
    if(!have_tok){
        while(g_cc.bytes+est>g_cc.cap){
            /* EVICT before waiting: unreferenced cached entries are instantly
             * reclaimable and must never block fetches (observed: cache filled
             * the cap, 57/64 downloaders gated behind the eviction trickle,
             * 6 Gbit on a 15 Gbit NIC). Only truly PINNED bytes may make us wait. */
            size_t before=g_cc.bytes;
            cc_evict_target_locked(g_cc.cap>est?g_cc.cap-est:0);
            if(g_cc.bytes+est<=g_cc.cap) break;
            if(g_cc.bytes<before) continue;       /* progress: re-check */
            if(pthread_mutex_trylock(&g_tok)==0){ have_tok=1; break; }
            pthread_cond_wait(&g_cc.cv,&g_cc.m);
        }
    }
    g_cc.bytes+=est;
    pthread_mutex_unlock(&g_cc.m);
    return have_tok;
}
static void cc_settle(size_t est){   /* remove the reservation (actuals were added by cc_fill) */
    pthread_mutex_lock(&g_cc.m);
    g_cc.bytes-=est;
    pthread_cond_broadcast(&g_cc.cv);
    pthread_mutex_unlock(&g_cc.m);
}
static void cc_tok_release(void){
    pthread_mutex_unlock(&g_tok);
    pthread_mutex_lock(&g_cc.m);
    pthread_cond_broadcast(&g_cc.cv);
    pthread_mutex_unlock(&g_cc.m);
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
    mc_archive *arc; float quality;
    long SB,BAND,halo;
    long ntx,nty,nbz,nunits;
    long pz,py,px;              /* padded dims (multiples of 2*MCC for L1 alignment) */
    long roz,roy,rox;          /* ROI origin in WORLD voxels (512-aligned): the archive's
                                * (0,0,0) maps to world (roz,roy,rox). Reads add it back;
                                * archive chunk coords are world-minus-origin. 0 = no trim. */
    atomic_long next, fail, skipped, l0app, l1app;
    atomic_long dl_fetch, dl_gate, wk_busy, units_done;   /* live telemetry gauges */
    unsigned char *done_bm; int prog_fd; atomic_long resumed;   /* resume journal */
    atomic_long io_open, io_miss, io_bytes;
    bandq q;
} sched;

static void *download_worker(void *arg){
    sched *sc=arg; const zlvl *z0=sc->z0;
    for(;;){
        long ti=atomic_fetch_add(&sc->next,1);
        if(ti>=sc->nunits) break;
        if(sc->done_bm && (sc->done_bm[ti>>3]>>(ti&7)&1)){
            atomic_fetch_add(&sc->units_done,1); atomic_fetch_add(&sc->resumed,1); continue;
        }
        int have_tok=0;
        long band_bytes=0;   /* gate per CHUNK below (per-band gating lets N
                              * downloaders overshoot the cap N-fold before any
                              * accounting); band_bytes>0 guarantees progress */
        long bz=ti%sc->nbz, txy=ti/sc->nbz, ty=txy/sc->ntx, tx=txy%sc->ntx;
        /* WORLD read position = trimmed band position + ROI origin */
        long vx0=tx*sc->SB+sc->rox, vy0=ty*sc->SB+sc->roy, vz0=bz*sc->BAND+sc->roz;
        long vx1=vx0+sc->SB, vy1=vy0+sc->SB, vz1=vz0+sc->BAND;
        /* halo-expanded chunk footprint, clamped to the true volume */
        long h=sc->halo;
        long fx0=vx0-h<0?0:vx0-h, fy0=vy0-h<0?0:vy0-h, fz0=vz0-h<0?0:vz0-h;
        long fx1=vx1+h>z0->sx?z0->sx:vx1+h, fy1=vy1+h>z0->sy?z0->sy:vy1+h, fz1=vz1+h>z0->sz?z0->sz:vz1+h;
        cchunk *chunks=NULL; int nch=0,cap=0,any=0;
        if(fx0<fx1&&fy0<fy1&&fz0<fz1){
            long cxa=fx0/z0->cx,cxb=(fx1-1)/z0->cx, cya=fy0/z0->cy,cyb=(fy1-1)/z0->cy;
            long cza=fz0/z0->cz,czb=(fz1-1)/z0->cz;
            /* phase 1: resolve cache hits + collect the chunks that need fetching */
            long need_cap=(czb-cza+1)*(cyb-cya+1)*(cxb-cxa+1);
            long *needq=malloc(need_cap*3*sizeof(long)); long nneed=0;
            for(long qz=cza;qz<=czb;++qz)for(long qy=cya;qy<=cyb;++qy)for(long qx=cxa;qx<=cxb;++qx){
                if(!sc->cm->present[((size_t)qz*sc->cm->ncy+qy)*sc->cm->ncx+qx]) continue;
                int core = (qz>=vz0/z0->cz && qz*z0->cz<vz1 && qy>=vy0/z0->cy && qy*z0->cy<vy1
                            && qx>=vx0/z0->cx && qx*z0->cx<vx1);
                centry *ce=cc_get(qz,qy,qx);
                if(ce){
                    if(!ce->bytes){ cc_release(ce); continue; }  /* negative-cached: chunk absent */
                    if(core) any=1;
                    if(nch==cap){cap=cap?cap*2:32;chunks=realloc(chunks,cap*sizeof *chunks);}
                    chunks[nch].ccz=qz;chunks[nch].ccy=qy;chunks[nch].ccx=qx;
                    chunks[nch].bytes=ce->bytes;chunks[nch].len=ce->len;chunks[nch].ce=ce;chunks[nch].fetched=0;nch++;
                } else {
                    needq[nneed*3]=qz; needq[nneed*3+1]=qy; needq[nneed*3+2]=qx; nneed++;
                }
            }
            /* phase 2: BATCHED fetch (s3_get_batch multiplexes transfers over pooled
             * connections -- a serial GET pays TLS+TTFB per object). Groups of 32,
             * budget-gated between groups; local paths use the serial reader.
             * SINGLE-FLIGHT: every needed chunk is cc_reserve'd first; exactly one
             * worker (owner) fetches a given chunk, everyone else waits on its fill. */
            for(long g0=0; g0<nneed; g0+=32){
                long gN=nneed-g0<32?nneed-g0:32;
                size_t gest=(size_t)gN*(size_t)z0->cz*z0->cy*z0->cx;
                atomic_fetch_add(&sc->dl_gate,1);
                have_tok=cc_gate_reserve(gest,have_tok);
                atomic_fetch_sub(&sc->dl_gate,1);
                atomic_fetch_add(&sc->dl_fetch,1);
                char paths[32][1500]; s3_range_req reqs[32]; s3_response rsp[32];
                centry *ces[32]; int owner[32]; long ownidx[32]; long nown=0;
                memset(rsp,0,sizeof rsp);
                for(long i=0;i<gN;++i){
                    long qz=needq[(g0+i)*3],qy=needq[(g0+i)*3+1],qx=needq[(g0+i)*3+2];
                    ces[i]=cc_reserve(qz,qy,qx,&owner[i]);
                    if(owner[i]){
                        snprintf(paths[i],sizeof paths[i],"%s/%ld/%ld/%ld",z0->dir,qz,qy,qx);
                        reqs[nown].url=paths[i]; reqs[nown].offset=0; reqs[nown].length=0;
                        ownidx[nown++]=i;
                    }
                }
                int use_batch = is_s3(z0->dir) && nown>1;
                if(use_batch && s3_get_batch(g_s3,reqs,nown,mca_batch_conc(),rsp)!=S3_OK){
                    for(long i=0;i<nown;++i) s3_response_free(&rsp[i]);
                    use_batch=0;                      /* transport failure -> serial (retries) */
                }
                /* pass 1: OWNERS fetch + cc_fill. All fills happen before any wait
                 * below, so cross-thread waits can never form a cycle. */
                for(long oi=0;oi<nown;++oi){
                    long i=ownidx[oi];
                    long qz=needq[(g0+i)*3],qy=needq[(g0+i)*3+1],qx=needq[(g0+i)*3+2];
                    int core = (qz>=vz0/z0->cz && qz*z0->cz<vz1 && qy>=vy0/z0->cy && qy*z0->cy<vy1
                                && qx>=vx0/z0->cx && qx*z0->cx<vx1);
                    u8 *got=NULL; size_t gn=0;
                    atomic_fetch_add(&sc->io_open,1);
                    if(use_batch){
                        if(rsp[oi].status==200 && rsp[oi].body){
                            got=(u8*)rsp[oi].body; gn=rsp[oi].body_len; rsp[oi].body=NULL;
                        } else if(rsp[oi].status!=404 && rsp[oi].status!=0){
                            atomic_fetch_add(&sc->fail,1);
                        }
                        s3_response_free(&rsp[oi]);
                    } else {
                        int err=0; got=src_get(paths[i],&gn,&err);
                        if(err) atomic_fetch_add(&sc->fail,1);
                    }
                    if(!got){
                        atomic_fetch_add(&sc->io_miss,1);
                        cc_fill(ces[i],NULL,0);       /* publish 'absent' to waiters */
                        cc_release(ces[i]);
                        continue;
                    }
                    if(core) any=1;
                    atomic_fetch_add(&sc->io_bytes,(long)gn);
                    band_bytes+=(long)gn;
                    cc_fill(ces[i],got,gn);
                    if(nch==cap){cap=cap?cap*2:32;chunks=realloc(chunks,cap*sizeof *chunks);}
                    chunks[nch].ccz=qz;chunks[nch].ccy=qy;chunks[nch].ccx=qx;
                    chunks[nch].bytes=got;chunks[nch].len=gn;chunks[nch].ce=ces[i];chunks[nch].fetched=1;nch++;
                }
                cc_settle(gest);
                atomic_fetch_sub(&sc->dl_fetch,1);
                /* pass 2: NON-OWNERS rendezvous with the concurrent fetch and share
                 * its bytes (fetched=0: resident bytes are tracked by the cache itself -- the owner's band
                 * carries those bytes). bytes==NULL = owner saw absent/failed; the
                 * owner already counted it, so just drop the ref. */
                for(long i=0;i<gN;++i){
                    if(owner[i]) continue;
                    long qz=needq[(g0+i)*3],qy=needq[(g0+i)*3+1],qx=needq[(g0+i)*3+2];
                    int core = (qz>=vz0/z0->cz && qz*z0->cz<vz1 && qy>=vy0/z0->cy && qy*z0->cy<vy1
                                && qx>=vx0/z0->cx && qx*z0->cx<vx1);
                    if(!cc_wait_ready(ces[i])){
                        /* owner stuck (likely gated): fetch a PRIVATE copy so this band
                         * can finish -- rare duplicate GET beats a wedged pipeline */
                        cc_release(ces[i]);
                        char p[1500]; snprintf(p,sizeof p,"%s/%ld/%ld/%ld",z0->dir,qz,qy,qx);
                        atomic_fetch_add(&sc->io_open,1);
                        size_t gn=0; int err=0; u8 *got=src_get(p,&gn,&err);
                        if(err) atomic_fetch_add(&sc->fail,1);
                        if(!got){ atomic_fetch_add(&sc->io_miss,1); continue; }
                        if(core) any=1;
                        atomic_fetch_add(&sc->io_bytes,(long)gn);
                        if(nch==cap){cap=cap?cap*2:32;chunks=realloc(chunks,cap*sizeof *chunks);}
                        chunks[nch].ccz=qz;chunks[nch].ccy=qy;chunks[nch].ccx=qx;
                        chunks[nch].bytes=got;chunks[nch].len=gn;
                        chunks[nch].ce=NULL;chunks[nch].fetched=2;nch++;   /* private: freed on release */
                        continue;
                    }
                    if(!ces[i]->bytes){ cc_release(ces[i]); continue; }
                    if(core) any=1;
                    if(nch==cap){cap=cap?cap*2:32;chunks=realloc(chunks,cap*sizeof *chunks);}
                    chunks[nch].ccz=qz;chunks[nch].ccy=qy;chunks[nch].ccx=qx;
                    chunks[nch].bytes=ces[i]->bytes;chunks[nch].len=ces[i]->len;
                    chunks[nch].ce=ces[i];chunks[nch].fetched=0;nch++;
                }
            }
            free(needq);
        }
        bandbuf b={ti,chunks,nch,any?0:1};
        bq_push(&sc->q,b);
        if(have_tok){ cc_tok_release(); have_tok=0; }
    }
    bq_prod_done(&sc->q);
    return NULL;
}

/* 2x downscale via the volume-compressor kernels: DS_BOX (plain mean) or DS_CBOX
 * (contrast-maintaining: mean pushed toward the cell's max-deviation voxel so thin
 * sheets/gaps survive coarse zoom). Strictly within-cell -> no halo, zero-preserving. */
static fy_ds_method g_ds = FY_DS_CBOX; static float g_alpha = 0.5f;
static void down2x(const u8 *in,long dz,long dy,long dx,u8 *out,long oz,long oy,long ox){
    int ox_,oy_,oz_;
    fy_downscale2x(in,(int)dx,(int)dy,(int)dz,out,&ox_,&oy_,&oz_,g_ds,g_alpha);
    (void)oz;(void)oy;(void)ox;
}

static void *compute_worker(void *arg){
    sched *sc=arg; const zlvl *z0=sc->z0;
    long SB=sc->SB, BAND=sc->BAND, h=sc->halo;
    size_t rawcap=(size_t)(SB+2*h)*(SB+2*h)*(BAND+2*h);
    u8 *raw=malloc(rawcap);                                   /* halo'd input band */
    u8 *proc=malloc((size_t)SB*SB*BAND);                      /* processed band    */
    u8 *cbuf=malloc((size_t)MCC*MCC*MCC);                     /* chunk gather      */
    u8 *tin=malloc((size_t)(128+2*h)*(128+2*h)*(128+2*h));    /* fysics tile in    */
    u8 *tout=malloc((size_t)128*128*128);                     /* fysics tile out   */
    if(!raw||!proc||!cbuf||!tin||!tout){ fprintf(stderr,"compute alloc failed\n"); exit(1); }
    mc_codec_ctx *mctx=mc_codec_ctx_new();                    /* one encoder ctx per worker */
    if(!mctx){ fprintf(stderr,"compute ctx alloc failed\n"); exit(1); }
    mc_codec_ctx_set_quality(mctx,sc->quality);
    mc_codec_ctx_set_max_error(mctx,(int)(2.0f*sc->quality+0.5f)); /* tau = 2q (locked rule: clips the heavy DCT tail ~free) */
    bandbuf bb;
    while(bq_pop(&sc->q,&bb)){
        long ti=bb.ti;
        long bz=ti%sc->nbz, txy=ti/sc->nbz, ty=txy/sc->ntx, tx=txy%sc->ntx;
        /* WORLD band position (reads/assembly); archive coords subtract the ROI origin */
        long vx0=tx*SB+sc->rox, vy0=ty*SB+sc->roy, vz0=bz*BAND+sc->roz;
        if(bb.empty){ atomic_fetch_add(&sc->skipped,1); atomic_fetch_add(&sc->units_done,1);
            if(sc->prog_fd>=0){ long v=ti; if(write(sc->prog_fd,&v,8)!=8){} }
            goto release; }
        atomic_fetch_add(&sc->wk_busy,1);
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
        /* append L0 chunks (LOD1+ are built independently in the tail). Clamp to the
         * archive's L0 chunk grid: BAND=512 over a 256-padded dim can overshoot by one
         * chunk at the edge, and that chunk is all-zero -> just skip it. */
        long ncz=(sc->pz+MCC-1)/MCC, ncy=(sc->py+MCC-1)/MCC, ncx=(sc->px+MCC-1)/MCC;
        for(long az=0;az<BAND/MCC;++az)for(long ay=0;ay<SB/MCC;++ay)for(long ax=0;ax<SB/MCC;++ax){
            long cz=(vz0-sc->roz)/MCC+az, cy=(vy0-sc->roy)/MCC+ay, cx=(vx0-sc->rox)/MCC+ax;
            if(cz>=ncz||cy>=ncy||cx>=ncx) continue;   /* edge overshoot (all-zero) */
            for(long z=0;z<MCC;++z)for(long y=0;y<MCC;++y)
                memcpy(cbuf+((size_t)z*MCC+y)*MCC,
                       proc+((size_t)(az*MCC+z)*SB+(ay*MCC+y))*SB+ax*MCC, MCC);
            if(mc_archive_append_chunk_ctx(sc->arc,mctx,0,cz,cy,cx,cbuf)!=0)
                atomic_fetch_add(&sc->fail,1);
            else atomic_fetch_add(&sc->l0app,1);
        }
        }
        atomic_fetch_sub(&sc->wk_busy,1);
        atomic_fetch_add(&sc->units_done,1);
        /* journal AFTER all of this unit's chunks are appended (archive is
         * crash-safe per append; the journal trails it, so a crash loses at
         * most the un-journaled units -- they re-process, never skip wrongly) */
        if(sc->prog_fd>=0){ long v=ti; if(write(sc->prog_fd,&v,8)!=8){} }
release:
        for(int i=0;i<bb.nch;++i){
            if(bb.chunks[i].ce) cc_release(bb.chunks[i].ce);
            else free(bb.chunks[i].bytes);          /* private timeout copy */
        }
        free(bb.chunks);
    }
    free(raw);free(proc);free(cbuf);free(tin);free(tout);
    mc_codec_ctx_free(mctx);
    return NULL;
}

/* live telemetry: one line every 10s -- where are the bands? */
static void *stats_worker(void *arg){
    sched *sc=arg;
    for(;;){
        sleep(10);
        long done=atomic_load(&sc->units_done);
        if(done>=sc->nunits) break;
        pthread_mutex_lock(&g_cc.m);
        size_t res=g_cc.bytes, cap=g_cc.cap; long hits=g_cc.hits, miss=g_cc.misses;
        pthread_mutex_unlock(&g_cc.m);
        pthread_mutex_lock(&sc->q.m); int qn=sc->q.count, qc=sc->q.cap; pthread_mutex_unlock(&sc->q.m);
        malloc_trim(0);   /* return freed heap to the OS: glibc retention crept
                           * +10GB over hours of band churn (observed 47->58GB at
                           * a fixed working set) */
        fprintf(stderr,"[t] units %ld/%ld (skip %ld) | q %d/%d | res %.1f/%.1fGB | dl fetch %ld gate %ld | wk busy %ld | cache %ld/%ld\n",
            done, sc->nunits, atomic_load(&sc->skipped), qn, qc, res/1e9, cap/1e9,
            atomic_load(&sc->dl_fetch), atomic_load(&sc->dl_gate), atomic_load(&sc->wk_busy), hits, hits+miss);
    }
    return NULL;
}

/* coarse tail: level L from level L-1 via the archive itself */
typedef struct { mc_archive *arc; float quality; int L; long ncz,ncy,ncx; atomic_long next, fail; } tailctx;
static void *tail_worker(void *arg){
    tailctx *t=arg;
    u8 *src=malloc((size_t)2*MCC*2*MCC*2*MCC), *dst=malloc((size_t)MCC*MCC*MCC);
    mc_codec_ctx *mctx=mc_codec_ctx_new();
    if(!mctx){ fprintf(stderr,"tail ctx alloc failed\n"); exit(1); }
    mc_codec_ctx_set_quality(mctx,t->quality);
    mc_codec_ctx_set_max_error(mctx,(int)(2.0f*t->quality+0.5f)); /* tau = 2q (locked rule) */
    long n=t->ncz*t->ncy*t->ncx;
    for(;;){
        long i=atomic_fetch_add(&t->next,1);
        if(i>=n) break;
        long cz=i/(t->ncy*t->ncx), r=i%(t->ncy*t->ncx), cy=r/t->ncx, cx=r%t->ncx;
        mc_archive_read_region(t->arc,t->L-1,cz*2L*MCC,cy*2L*MCC,cx*2L*MCC,
                               2*MCC,2*MCC,2*MCC,src,(size_t)2*MCC*2*MCC,(size_t)2*MCC,1);
        down2x(src,2*MCC,2*MCC,2*MCC,dst,MCC,MCC,MCC);
        if(mc_archive_append_chunk_ctx(t->arc,mctx,t->L,cz,cy,cx,dst)!=0)
            atomic_fetch_add(&t->fail,1);
    }
    free(src);free(dst);
    mc_codec_ctx_free(mctx);
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


/* ---- provenance: stamp the export's full parameterization into the archive's
 * user-metadata carveout (the [256B..128KB) region every .mca reserves), so the
 * archive is self-describing: profile, every resolved parameter (CLI knobs +
 * the post-calibration pipeline config), and the source metadata.json inlined
 * verbatim. Read back with mc_archive_metadata()/mc_metadata(). ---- */
static size_t prov_escape(char *d, size_t cap, const char *s){
    size_t n=0;
    for(; *s && n+8<cap; ++s){
        unsigned char ch=(unsigned char)*s;
        if(ch=='"'||ch=='\\'){ d[n++]='\\'; d[n++]=(char)ch; }
        else if(ch<0x20) n+=(size_t)snprintf(d+n,cap-n,"\\u%04x",ch);
        else d[n++]=(char)ch;
    }
    d[n]=0; return n;
}
static void stamp_provenance(mc_archive *arc, const char *in, const char *out,
                             const char *profile, float quality, int process,
                             const char *meta_path, const fy_pipeline_cfg *c,
                             long SB, long BAND, int nthreads, int niothreads, int qcap,
                             double mem_gb, double cache_gb,
                             long roz, long roy, long rox,            /* ROI origin in source/world voxels */
                             long wz, long wy, long wx){              /* source (untrimmed) world dims */
    size_t cap=120*1024;
    char *j=malloc(cap); if(!j) return;
    char ein[PATH_MAX*2], eout[PATH_MAX*2], emp[PATH_MAX*2];
    prov_escape(ein,sizeof ein,in); prov_escape(eout,sizeof eout,out);
    prov_escape(emp,sizeof emp,meta_path);
    char ts[32]; { time_t now=time(NULL); struct tm tm; gmtime_r(&now,&tm);
                   strftime(ts,sizeof ts,"%Y-%m-%dT%H:%M:%SZ",&tm); }
    size_t n=(size_t)snprintf(j,cap,
        "{\n"
        " \"tool\": \"mca_export\",\n"
        " \"created_utc\": \"%s\",\n"
        " \"input\": \"%s\",\n"
        " \"output\": \"%s\",\n"
        " \"profile\": \"%s\",\n"
        " \"quality\": %.4g,\n"
        " \"process\": %d,\n"
        " \"cli\": {\"sb\": %ld, \"band\": %ld, \"threads\": %d, \"io_threads\": %d,"
        " \"queue\": %d, \"mem_gb\": %.3g, \"cache_gb\": %.3g,"
        " \"downscale\": \"%s\", \"alpha\": %.4g},\n"
        " \"roi\": {\"origin\": [%ld, %ld, %ld], \"source_dims\": [%ld, %ld, %ld]},\n",
        ts,ein,eout,profile,(double)quality,process,SB,BAND,nthreads,niothreads,qcap,
        mem_gb,cache_gb,g_ds==FY_DS_BOX?"box":"cbox",(double)g_alpha,
        roz,roy,rox,wz,wy,wx);
    n+=(size_t)snprintf(j+n,cap-n,
        " \"pipeline\": {\n"
        "  \"delta_beta\": %.6g, \"energy_kev\": %.6g, \"distance_mm\": %.6g,"
        " \"pixel_um\": %.6g,\n"
        "  \"unsharp_sigma\": %.6g, \"unsharp_coeff\": %.6g,\n"
        "  \"machine_current_start\": %.6g, \"machine_current_stop\": %.6g,\n"
        "  \"window_lo\": %.6g, \"window_hi\": %.6g,\n"
        "  \"do_deconv\": %d, \"use_matched_deconv\": %d, \"psf_sigma_vox\": %.6g,"
        " \"deconv_tikhonov\": %.6g,\n"
        "  \"guided_eps\": %.8g, \"guided_subsample\": %d, \"denoise_k\": %.6g,\n"
        "  \"do_air_zero\": %d, \"air_cut_u8\": %d, \"air_cut_band\": %d,"
        " \"air_thresh\": %.6g, \"air_cut_aggr\": %.6g,\n"
        "  \"scratch_passes\": %d,\n"
        "  \"do_normalize\": %d, \"norm_lo\": %d, \"norm_hi\": %d,\n"
        "  \"do_zdrift\": %d, \"do_dering\": %d, \"dering_cy\": %.6g, \"dering_cx\": %.6g,\n"
        "  \"calib_budget_gb\": %.6g,\n"
        "  \"do_musica\": %d, \"musica_p\": %.6g, \"musica_levels\": %d, \"musica_core\": %.6g,\n"
        "  \"no_dither\": %d, \"halo\": %d\n"
        " },\n"
        " \"calibration\": {\n"
        "  \"have_norm\": %d, \"norm_lo\": %d, \"norm_hi\": %d,\n"
        "  \"have_zdrift\": %d, \"vol_z\": %ld,\n"
        "  \"have_dering\": %d,\n"
        "  \"psf_p5\": %.6g, \"psf_med\": %.6g, \"psf_z_ratio\": %.6g,\n"
        "  \"have_eps_r\": %d, \"eps_fn_a\": %.8g, \"eps_fn_b\": %.8g, \"eps_fn_med\": %.8g,\n"
        "  \"have_dec_range\": %d, \"dec_lo\": %.8g, \"dec_hi\": %.8g\n"
        " },\n"
        " \"source_metadata_path\": \"%s\",\n",
        c->delta_beta,c->energy_kev,c->distance_mm,c->pixel_um,
        c->unsharp_sigma,c->unsharp_coeff,
        c->machine_current_start,c->machine_current_stop,
        c->window_lo,c->window_hi,
        c->do_deconv,c->use_matched_deconv,c->psf_sigma_vox,c->deconv_tikhonov,
        c->guided_eps,c->guided_subsample,c->denoise_k,
        c->do_air_zero,c->air_cut_u8,c->air_cut_band,c->air_thresh,c->air_cut_aggr,
        c->scratch_passes,
        c->do_normalize,c->norm_lo,c->norm_hi,
        c->do_zdrift,c->do_dering,c->dering_cy,c->dering_cx,
        c->calib_budget_gb,
        c->do_musica,c->musica_p,c->musica_levels,c->musica_core,
        c->no_dither,c->halo,
        c->have_norm,c->norm_lo,c->norm_hi,
        c->have_zdrift,c->vol_z,
        c->have_dering,
        c->psf_p5,c->psf_med,c->psf_z_ratio,
        c->have_eps_r,c->eps_fn_a,c->eps_fn_b,c->eps_fn_med,
        c->have_dec_range,c->dec_lo,c->dec_hi,
        emp);
    /* inline the source metadata.json verbatim when it reads as a JSON value;
     * keep ~32KB headroom under the region cap for it. */
    int inlined=0;
    FILE *mf=fopen(meta_path,"rb");
    if(mf){
        fseek(mf,0,SEEK_END); long ml=ftell(mf); fseek(mf,0,SEEK_SET);
        if(ml>0 && (size_t)ml<cap-n-64){
            char *mb=malloc((size_t)ml+1);
            if(mb && fread(mb,1,(size_t)ml,mf)==(size_t)ml){
                mb[ml]=0; char *p=mb; while(*p==' '||*p=='\n'||*p=='\r'||*p=='\t')p++;
                if(*p=='{'||*p=='['){
                    n+=(size_t)snprintf(j+n,cap-n," \"source_metadata\": %s\n}\n",p);
                    inlined=1;
                }
            }
            free(mb);
        }
        fclose(mf);
    }
    if(!inlined) n+=(size_t)snprintf(j+n,cap-n," \"source_metadata\": null\n}\n");
    if(mc_archive_set_metadata(arc,j,n)==0)
        fprintf(stderr,"provenance: %zu B stamped into archive metadata (source %s)\n",
                n,inlined?"inlined":"absent");
    else
        fprintf(stderr,"provenance: WARNING metadata stamp failed (%zu B)\n",n);
    free(j);
}

/* ---- calibration persistence: stash the resolved calibration into <out>.calib
 * so a resume skips the (deterministic but ~minutes-long) recalibration. Keyed on
 * geometry + profile; pointer members (zdrift_factor[Z], dering rings) are
 * serialized inline. ---- */
#define CALIB_MAGIC 0x424C4143u   /* "CALB" */
static int resume_has_progress(const char *out, long SB, long BAND, const zlvl *z0){
    char pp[2200]; snprintf(pp,sizeof pp,"%s.progress",out);
    FILE *f=fopen(pp,"rb"); if(!f) return 0;
    long h[6]={0}, want[6]={0x4d435052,SB,BAND,z0->sz,z0->sy,z0->sx};
    int ok = fread(h,sizeof h,1,f)==1 && memcmp(h,want,sizeof h)==0;
    if(ok){ fseek(f,0,SEEK_END); ok = ftell(f) > 48; }   /* header + >=1 unit */
    fclose(f); return ok;
}
static void calib_save(const char *out, const fy_pipeline_cfg *c, long SB, long BAND,
                       const zlvl *z0, const char *profile){
    char p[PATH_MAX]; snprintf(p,sizeof p,"%s.calib",out);
    FILE *f=fopen(p,"wb"); if(!f) return;
    uint32_t magic=CALIB_MAGIC, ver=1;
    long hdr[6]={SB,BAND,z0->sz,z0->sy,z0->sx,(long)profile[0]};
    fwrite(&magic,4,1,f); fwrite(&ver,4,1,f); fwrite(hdr,sizeof hdr,1,f);
    fwrite(c,sizeof *c,1,f);
    int hz = c->have_zdrift && c->zdrift_factor;
    fwrite(&hz,sizeof hz,1,f);
    if(hz) fwrite(c->zdrift_factor,sizeof(float),(size_t)z0->sz,f);
    int hd = c->have_dering && c->dering && c->dering->ring;
    fwrite(&hd,sizeof hd,1,f);
    if(hd){ const fy_dering *d=c->dering; fwrite(d,sizeof *d,1,f);
            fwrite(d->ring,sizeof(float),(size_t)d->nslab*d->nr,f); }
    fclose(f);
}
static int calib_load(const char *out, fy_pipeline_cfg *c, long SB, long BAND,
                      const zlvl *z0, const char *profile){
    char p[PATH_MAX]; snprintf(p,sizeof p,"%s.calib",out);
    FILE *f=fopen(p,"rb"); if(!f) return -1;
    uint32_t magic=0,ver=0; long hdr[6]={0};
    long want[6]={SB,BAND,z0->sz,z0->sy,z0->sx,(long)profile[0]};
    if(fread(&magic,4,1,f)!=1||fread(&ver,4,1,f)!=1||fread(hdr,sizeof hdr,1,f)!=1||
       magic!=CALIB_MAGIC||ver!=1||memcmp(hdr,want,sizeof hdr)!=0){ fclose(f); return -1; }
    if(fread(c,sizeof *c,1,f)!=1){ fclose(f); return -1; }
    c->occ=NULL; c->zdrift_factor=NULL; c->dering=NULL;   /* re-bound below / unused */
    int hz=0; if(fread(&hz,sizeof hz,1,f)!=1){ fclose(f); return -1; }
    if(hz){ float *zf=malloc((size_t)z0->sz*sizeof(float));
        if(!zf||fread(zf,sizeof(float),(size_t)z0->sz,f)!=(size_t)z0->sz){ free(zf); fclose(f); return -1; }
        c->zdrift_factor=zf; c->have_zdrift=1; }
    int hd=0; if(fread(&hd,sizeof hd,1,f)!=1){ fclose(f); return -1; }
    if(hd){ fy_dering *d=malloc(sizeof *d);
        if(!d||fread(d,sizeof *d,1,f)!=1){ free(d); fclose(f); return -1; }
        d->sum=NULL; d->cnt=NULL; d->ring=NULL;
        size_t rn=(size_t)d->nslab*d->nr;
        d->ring=malloc(rn*sizeof(float));
        if(!d->ring||fread(d->ring,sizeof(float),rn,f)!=rn){ free(d->ring); free(d); fclose(f); return -1; }
        c->dering=d; c->have_dering=1; }
    fclose(f);
    return 0;
}

int main(int argc, char **argv){
    signal(SIGSEGV,fy_crash_handler); signal(SIGABRT,fy_crash_handler);
    signal(SIGBUS,fy_crash_handler);  signal(SIGFPE,fy_crash_handler);
    if(argc<3){
        fprintf(stderr,"usage: %s <in_zarr|s3://...> <out.mc> [--profile P] [--quality Q|auto]"
                       " [--meta PATH] [--threads N] [--io-threads M] [--queue C]"
                       " [--sb SB] [--band B] [--no-process] [--no-dither]\n"
                       "       (--quality auto sets the DC quant step to the calibrated noise floor)\n"
                       "       [--downscale box|cbox] [--alpha A]   (LOD kernel; default cbox 0.5)\n"
                       "       [--air-cut]        (enable the value air-cut; OFF by default -- masked input zeros are kept by the codec)\n"
                       "       [--air-min-comp N] (despeckle when --air-cut: drop interior material islands < N voxels; default 64)\n"
                       "       [--air-aggr A]     (air-cut dial: 0=physics floor, 1=histogram valley, >1=past valley; overrides --profile)\n",argv[0]);
        return 2;
    }
    const char *in=argv[1], *out=argv[2], *profile="conservative";
    float quality=8.0f;
    long SB=512, BAND=512;   /* SB=512: small bands keep the pinned working set far
                              * under the fetch budget (SB=1024 bands serialized the
                              * downloaders through the progress token) and shrink
                              * worker buffers so compute can run FULL-WIDTH */
    int nthreads=0, niothreads=0, qcap=0, process=1, no_dither=0, auto_q=0; double cache_gb=12.0, mem_gb=24.0, calib_gb=10.0;
    int air_min_comp=64;       /* despeckle: drop interior material islands < N voxels */
    double air_aggr=-1.0;      /* air-cut dial override: 0=floor,1=valley,>1=past valley; <0 = use profile */
    int do_air=0;              /* air-cut OFF by default (--air-cut to enable); the masked
                                * input zeros are preserved by the codec, no value-cut needed */
    int base_level=0;          /* source pyramid level to treat as this archive's LOD0
                                * (--level 5 streams the 32x scroll for a small coarse archive;
                                * progressively 5->4->3->2->1 for finer detail) */
    char meta_path[PATH_MAX]; meta_path[0]=0;
    for(int i=3;i<argc;i++){
        if      (!strcmp(argv[i],"--profile")&&i+1<argc)   profile=argv[++i];
        else if (!strcmp(argv[i],"--quality")&&i+1<argc)   { const char*qa=argv[++i]; if(!strcmp(qa,"auto")) auto_q=1; else quality=(float)atof(qa); }
        else if (!strcmp(argv[i],"--meta")&&i+1<argc)      snprintf(meta_path,sizeof meta_path,"%s",argv[++i]);
        else if (!strcmp(argv[i],"--threads")&&i+1<argc)   nthreads=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--io-threads")&&i+1<argc)niothreads=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--queue")&&i+1<argc)     qcap=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--sb")&&i+1<argc)        SB=atol(argv[++i]);
        else if (!strcmp(argv[i],"--band")&&i+1<argc)      BAND=atol(argv[++i]);
        else if (!strcmp(argv[i],"--no-process"))          process=0;
        else if (!strcmp(argv[i],"--no-dither"))           no_dither=1;
        else if (!strcmp(argv[i],"--downscale")&&i+1<argc) g_ds=strcmp(argv[++i],"box")?FY_DS_CBOX:FY_DS_BOX;
        else if (!strcmp(argv[i],"--alpha")&&i+1<argc)     g_alpha=(float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--cache-gb")&&i+1<argc)  cache_gb=atof(argv[++i]);
        else if (!strcmp(argv[i],"--calib-gb")&&i+1<argc)  calib_gb=atof(argv[++i]);
        else if (!strcmp(argv[i],"--air-min-comp")&&i+1<argc) air_min_comp=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--air-aggr")&&i+1<argc)     air_aggr=atof(argv[++i]);
        else if (!strcmp(argv[i],"--air-cut"))                do_air=1;
        else if (!strcmp(argv[i],"--level")&&i+1<argc)     base_level=atoi(argv[++i]);
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
    cfg.do_air_zero=do_air; cfg.air_cut_u8=-1; cfg.air_cut_band=8; cfg.air_thresh=0.05;
    cfg.air_min_component=air_min_comp;
    cfg.scratch_passes=5;
    cfg.no_dither=no_dither;   /* --no-dither: round-to-nearest (dither is redundant under the noise floor) */
    cfg.do_normalize=0; cfg.norm_lo=-1; cfg.norm_hi=-1;   /* no recenter/stretch by default */
    cfg.calib_budget_gb=calib_gb;   /* S3 calibration sample budget (default 10 GB; was 200) */
    cfg.do_zdrift=1; cfg.do_dering=1; cfg.dering_cy=-1; cfg.dering_cx=-1;
    /* denoise_k=-1 => guided denoise OFF in all profiles (white noise floor ~1 u8: nothing
     * to remove, and smoothing risks fine sheet detail). Profiles now differ only in air_cut.
     * dering+zdrift stay on. Re-enable with --denoise N if ever needed. */
    if      (!strcmp(profile,"conservative")) { cfg.air_cut_aggr=0.0; cfg.denoise_k=-1.0f; }
    else if (!strcmp(profile,"aggressive"))   { cfg.air_cut_aggr=1.0; cfg.denoise_k=-1.0f; }
    else { fprintf(stderr,"unknown profile %s\n",profile); return 2; }
    if (air_aggr >= 0) cfg.air_cut_aggr = air_aggr;   /* --air-aggr overrides the profile dial */
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
        fprintf(stderr,"mca_export: %d meta keys from %s\n",nmeta,meta_path);
    }

    /* ---- parse the base source level + occupancy ---- */
    zlvl z0;
    if(parse_lvl(in,base_level,&z0)!=0){fprintf(stderr,"cannot parse %s/%d/.zarray\n",in,base_level);return 1;}
    fprintf(stderr,"source level %d/: input %ldx%ldx%ld chunks %ldx%ldx%ld\n",base_level,z0.sz,z0.sy,z0.sx,z0.cz,z0.cy,z0.cx);
    chunkmap cm; long bb[6]={0};
    if(cm_build(&cm,in,&z0,bb,base_level)!=0){fprintf(stderr,"occupancy build failed\n");return 1;}
    /* ROI TRIM: crop to the content bbox, translate to (0,0,0), pad to MCC (256).
     * Origin snaps DOWN to 256 and the end UP to 256 -> LOD0 is chunk-tight. Each
     * coarser LOD is built independently in the tail (its own dims = L0>>L, storage
     * rounded to 256), so L0 need NOT be a 512 multiple. Origin is 256-aligned, which
     * keeps every LOD's 2x downscale phase-consistent with the source pyramid. */
    long ROIA=MCC;
    long roz=(bb[0]/ROIA)*ROIA, roy=(bb[2]/ROIA)*ROIA, rox=(bb[4]/ROIA)*ROIA;
    long rez=((bb[1]+ROIA-1)/ROIA)*ROIA, rey=((bb[3]+ROIA-1)/ROIA)*ROIA, rex=((bb[5]+ROIA-1)/ROIA)*ROIA;
    long tsz=rez-roz, tsy=rey-roy, tsx=rex-rox;
    if(tsz<=0||tsy<=0||tsx<=0){ roz=roy=rox=0; tsz=((z0.sz+ROIA-1)/ROIA)*ROIA; tsy=((z0.sy+ROIA-1)/ROIA)*ROIA; tsx=((z0.sx+ROIA-1)/ROIA)*ROIA; }
    if(roz||roy||rox||tsz!=z0.sz||tsy!=z0.sy||tsx!=z0.sx)
        fprintf(stderr,"ROI trim: world bbox z[%ld,%ld) y[%ld,%ld) x[%ld,%ld) -> origin(%ld,%ld,%ld) dims %ldx%ldx%ld (was %ldx%ldx%ld)\n",
                bb[0],bb[1],bb[2],bb[3],bb[4],bb[5],roz,roy,rox,tsz,tsy,tsx,z0.sz,z0.sy,z0.sx);

    /* hand the occupancy map to calibration so its sweep skips known-empty ptiles
     * (no 404 GETs) and concentrates its sample budget on the occupied volume. */
    cfg.occ=cm.present; cfg.occ_ncz=cm.ncz; cfg.occ_ncy=cm.ncy; cfg.occ_ncx=cm.ncx;
    cfg.occ_cz=z0.cz;   cfg.occ_cy=z0.cy;  cfg.occ_cx=z0.cx;

    /* ---- calibrate (reads the zarr through the fysics library) ---- */
    if(process){
        if(resume_has_progress(out,SB,BAND,&z0) && calib_load(out,&cfg,SB,BAND,&z0,profile)==0){
            fprintf(stderr,"resume: loaded calibration from %s.calib (skipping recalibration)\n",out);
        } else {
            if(fy_calibrate(in,&cfg,128,1)!=0){fprintf(stderr,"calibrate failed\n");return 1;}
            calib_save(out,&cfg,SB,BAND,&z0,profile);
        }
    }

    /* ---- --quality auto: set DC quant step to the measured white-noise floor, so
     * quantization discards noise but keeps signal (q3 ~ noise floor on BM18). Needs
     * calibration (the noise floor); refuse to guess without it. Clamp to the archival
     * step floor 0.5 and a sane cap. ---- */
    if(auto_q){
        if(!process){ fprintf(stderr,"--quality auto requires processing (drop --no-process)\n"); return 2; }
        double nf=cfg.noise_floor;
        if(!(nf>0)){ fprintf(stderr,"--quality auto: no noise floor from calibration; falling back to q3\n"); nf=3.0; }
        /* q = noise_floor: cfg.noise_floor measures the WHITE (decorrelated) noise floor,
         * which on unsharp BM18 data is ~1-3 u8 -> q~3. Do NOT scale it up by 2.8: a
         * naive flat-region/block-std reads ~18 because the unsharp content is SMOOTH
         * SIGNAL (lag-1 autocorr +0.92, a 4th-order signal-blind estimator sees ~1), and
         * scaling to q~40 would quantize away the sharpening (real signal we keep -- see
         * [[unsharp-reversal-dominated]]). tau=2q is applied at encode (a fidelity bound). */
        quality=(float)(nf<0.5?0.5:nf>16.0?16.0:nf);
        fprintf(stderr,"[auto-q] white noise_floor=%.3f u8 -> quality=%.3f tau=2q=%.0f\n",
                cfg.noise_floor,quality,2.0f*quality);
    }

    /* ---- archive + schedule ---- */
    mc_codec_init();
    mc_archive *arc=mc_archive_open_dims(out,(int)tsx,(int)tsy,(int)tsz,quality);
    if(!arc){fprintf(stderr,"cannot open %s\n",out);return 1;}
    sched sc; memset(&sc,0,sizeof sc);
    sc.prog_fd=-1;
    sc.z0=&z0; sc.cm=&cm; sc.cfg=&cfg; sc.process=process; sc.arc=arc; sc.quality=quality;
    sc.SB=SB; sc.BAND=BAND; sc.halo=cfg.halo>0?cfg.halo:8;
    sc.roz=roz; sc.roy=roy; sc.rox=rox;
    sc.px=tsx; sc.py=tsy; sc.pz=tsz;   /* trimmed extents, already 2*MCC multiples */
    sc.ntx=(sc.px+SB-1)/SB; sc.nty=(sc.py+SB-1)/SB; sc.nbz=(sc.pz+BAND-1)/BAND;
    sc.nunits=sc.ntx*sc.nty*sc.nbz;
    /* ---- RESUME journal: <out>.progress holds 8-byte completed-unit ids after a
     * geometry header; a matching journal + reopenable archive -> completed units
     * are skipped without any I/O. Geometry mismatch -> start fresh. ---- */
    {
        char pp[2200]; snprintf(pp,sizeof pp,"%s.progress",out);
        long hdr[6]={0x4d435052,SB,BAND,(long)tsz,(long)tsy,(long)tsx};   /* trimmed dims: ROI change invalidates a stale journal */
        int fd=open(pp,O_RDWR,0644);
        long nresume=0;
        if(fd>=0){
            long h[6]={0};
            if(read(fd,h,48)==48 && memcmp(h,hdr,48)==0){
                sc.done_bm=calloc((sc.nunits+7)/8,1);
                long v;
                while(read(fd,&v,8)==8) if(v>=0&&v<sc.nunits&&sc.done_bm){ if(!(sc.done_bm[v>>3]>>(v&7)&1)){sc.done_bm[v>>3]|=1<<(v&7); nresume++;} }
                lseek(fd,0,SEEK_END);
                sc.prog_fd=fd;
            } else { close(fd); fd=-1; }   /* stale geometry -> rewrite below */
        }
        if(fd<0){
            fd=open(pp,O_CREAT|O_RDWR|O_TRUNC,0644);
            if(fd>=0){ if(write(fd,hdr,48)!=48){} sc.prog_fd=fd; }
        }
        fprintf(stderr,"resume: %ld of %ld units journaled complete\n",nresume,sc.nunits);
    }
    /* AUTO-SIZING for the standard export fleet: compute instances provision
     * 2 GB RAM per hardware thread (N threads -> 2N GB). Budget split:
     *   compute = 3N/4 workers (~1.3 GB each at SB=1024)  ~ 1.0N GB
     *   in-flight fetched chunks                           ~ 0.5N GB
     *   chunk cache                                        ~ 0.2N GB
     *   slack (archive mmap, kernel, calib)                ~ 0.3N GB
     * Every term is overridable (--threads/--io-threads/--queue/--mem-gb/--cache-gb). */
    long ncpu=sysconf(_SC_NPROCESSORS_ONLN); if(ncpu<1)ncpu=4;
    /* Threading philosophy (Forrest): pools may EXCEED hardware threads in total --
     * blocked-on-network io threads cost only stack. Compute = ncpu (full width;
     * measured ~1.3 GB/worker at SB=512 incl. encoder TLS), io = 2x ncpu. */
    int nc_=nthreads>0?nthreads:(int)(ncpu<2?2:ncpu);
    /* band_max must be the halo-rounded chunk FOOTPRINT, not core voxels: a
     * SB=512 band fetches (4+2)^3 chunks for a 4^3 core (~3.4x) -- sizing on
     * core bytes let 64 in-flight bands pin 27GB against a 12GB cap. Assume
     * 128^3 input chunks (the fleet standard) for the estimate. */
    long band_max=((SB/128)+2)*((SB/128)+2)*((BAND/128)+2)*2097152L;
    int ni_=niothreads>0?niothreads:(is_s3(in)?(int)(ncpu*2):(nc_<4?nc_:4));
    /* downloaders' worst-case in-flight band footprints must fit the budget
     * (gated otherwise -> token-serialized crawl): clamp the io pool to it */
    { long capb=(long)((mem_gb==24.0?ncpu/2.0:mem_gb)+(cache_gb==12.0?ncpu/4.0:cache_gb))*1000000000L;
      int maxio=(int)(capb*7/10/band_max);
      if(niothreads<=0 && ni_>maxio && maxio>=4) ni_=maxio; }
    /* queue depth from the PINNING INVARIANT: bands held by queue+workers pin
     * their chunk bytes in the resident budget; if (qcap+nc_)*band_max >= cap the
     * downloaders all gate and fetches serialize through the progress token
     * (observed: 58/64 downloaders cond-waiting, 6 fetching). Keep worst-case
     * pinned bytes <= half the budget so fetch headroom always exists. */
    long cap_b=(long)((mem_gb==24.0?ncpu/2.0:mem_gb)*1e9)+(long)((cache_gb==12.0?ncpu/4.0:cache_gb)*1e9);
    int qc_=qcap>0?qcap:(int)(cap_b/2/band_max-nc_);
    if(qcap<=0){ if(qc_<4)qc_=4; if(qc_>nc_*2)qc_=nc_*2; }
    if(mem_gb==24.0)  mem_gb=ncpu/2.0;   /* resident budget shares RAM with ~1.3GB/worker */     /* defaults only when not user-set */
    if(cache_gb==12.0) cache_gb=ncpu/4.0;
    /* ONE resident-bytes budget covers in-flight bands AND cached reuse */
    cc_init((size_t)((mem_gb+cache_gb)*1e9));
    bq_init(&sc.q,qc_,ni_);
    fprintf(stderr,"units %ld (%ldx%ld tiles x %ld bands), SB=%ld BAND=%ld halo=%ld, "
                   "%d compute + %d io threads, queue %d\n",
            sc.nunits,sc.ntx,sc.nty,sc.nbz,SB,BAND,sc.halo,nc_,ni_,qc_);

    /* stamp provenance now that every parameter is RESOLVED (post-calibration,
     * post auto-sizing) -- the archive records what actually ran, not the CLI */
    stamp_provenance(arc,in,out,profile,quality,process,meta_path,&cfg,
                     SB,BAND,nc_,ni_,qc_,mem_gb,cache_gb,
                     roz,roy,rox,z0.sz,z0.sy,z0.sx);

    pthread_t stat_t; pthread_create(&stat_t,NULL,stats_worker,&sc); pthread_detach(stat_t);
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

    /* ---- coarse tail L1.. from the archive (each LOD built from L-1, padded to its
     * own 256 multiple -- LOD0 stays chunk-tight, no 512 coupling) ---- */
    for(int L=1;L<8;L++){
        long dz=(tsz>>L), dy=(tsy>>L), dx=(tsx>>L);   /* tail grid over the TRIMMED archive dims */
        if(dz<1&&dy<1&&dx<1) break;
        tailctx t={arc,quality,L,(dz+MCC-1)/MCC?:(long)1,(dy+MCC-1)/MCC?:(long)1,(dx+MCC-1)/MCC?:(long)1,0,0};
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
    fprintf(stderr,"mca_export: wrote %s\n",out);
    return 0;
}
