/* zarr_io.c -- minimal zarr v2 reader/writer for raw uint8 volumes (local + optional S3).
 *
 * Format produced by the export (and what we consume): level "0/" with a .zarray JSON header
 * and raw chunk files at "0/z/y/x" (dimension_separator '/'), dtype |u1, compressor null,
 * fill_value 0. A chunk file is exactly chunk_z*chunk_y*chunk_x bytes of raw u8 (C order).
 * Missing chunk -> all fill_value (air padding). No compression.
 * S3: when compiled with FYSICS_S3 (links vendored libs3 + libcurl), a root starting with
 * "s3://" is fetched per-chunk via libs3 instead of local fopen/mmap. The core library is
 * dependency-free unless FYSICS_S3 is defined. */
#include "fysics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#ifdef FYSICS_S3
#include "libs3.h"
static s3_client *g_fy_s3 = NULL;
/* per-batch connection count (env MCA_BATCH, default 16). Calibration runs
 * sweep_threads of these, so total S3 conns ~= sweep_threads * this; keep modest
 * so connections are reused (keep-alive), not churned (churn -> handshake resets). */
static int fy_batch_conc(void){ static int v=0; if(!v){const char*e=getenv("MCA_BATCH"); v=e?atoi(e):16; if(v<1)v=16;} return v; }
static void fy_s3_ensure(void){
    if(g_fy_s3) return;
    s3_config cfg; memset(&cfg,0,sizeof cfg);
    /* Vesuvius open-data buckets are PUBLIC: ANONYMOUS (unsigned) requests -- no
     * cred_provider (avoids SSO / expired-cred signing failures). Pin the region. */
    const char *reg = getenv("AWS_REGION"); if(!reg||!reg[0]) reg = getenv("AWS_DEFAULT_REGION");
    cfg.region = (reg && reg[0]) ? reg : "us-east-1";
    g_fy_s3=s3_client_new(&cfg);
}
#endif
/* fetch one S3 object into `out` (up to cn bytes; exact=1 requires >=cn, else accept short and
 * zero-fill the tail). A MISSING object (HTTP 404) is a legitimate sparse-zarr chunk -> fill.
 * Any OTHER failure (transport error, 5xx, throttling) is retried with backoff and, if it
 * persists, reported as a hard error -- it must NEVER silently become fill, or a transient
 * network blip writes undetectable air into a multi-TB output.
 * Returns: 1 = handled OK (data or legit 404 fill), 0 = not an S3 path, -1 = S3 error. */
static int fy_s3_get(const char *path, unsigned char *out, long cn, unsigned char fill, int exact){
#ifdef FYSICS_S3
    if(strncmp(path,"s3://",5)!=0) return 0;
    fy_s3_ensure();
    if(!g_fy_s3){ fprintf(stderr,"zarr: S3 client init failed for %s\n",path); return -1; }
    const int max_tries = 5;
    for(int try = 0; try < max_tries; try++){
        if(try > 0){ /* backoff: 0.2s, 0.4s, 0.8s, 1.6s */
            struct timespec ts; long ms = 200L << (try - 1);
            ts.tv_sec = ms / 1000; ts.tv_nsec = (ms % 1000) * 1000000L;
            nanosleep(&ts, NULL);
        }
        s3_response r; memset(&r,0,sizeof r);
        s3_status st=s3_get(g_fy_s3,path,&r);
        if(st==S3_OK && r.status==200 && r.body){
            long have=(long)r.body_len;
            if(exact && have<cn) memset(out,fill,cn);   /* short object = stored edge chunk */
            else { long cp=have<cn?have:cn; memcpy(out,r.body,cp); if(cp<cn) memset(out+cp,0,cn-cp); }
            s3_response_free(&r);
            return 1;
        }
        if(r.status==404){ memset(out,fill,cn); s3_response_free(&r); return 1; }  /* absent = sparse,
            regardless of the libs3 status code (it reports 404 as an error status) */
        if(try == max_tries-1)
            fprintf(stderr,"zarr: S3 GET failed after %d tries (status=%d http=%ld): %s\n",
                    max_tries, (int)st, (long)r.status, path);
        s3_response_free(&r);
    }
    return -1;
#else
    (void)path;(void)out;(void)cn;(void)fill;(void)exact; return 0;
#endif
}
/* chunk read: requires the full chunk (exact). Same return codes as fy_s3_get. */
static int fy_s3_read_chunk(const char *path, unsigned char *out, long cn, unsigned char fill){
    return fy_s3_get(path, out, cn, fill, 1);
}
/* ownership-stealing variant: hands back the S3 response body itself (libs3's
 * s3_response_free is a plain free(body), so the buffer is transferable) -- the
 * fetched chunk is then row-copied ONCE into the region, not body->scratch->region
 * (the double copy was ~17% of the calibration sweep on aarch64).
 * Returns 1 handled (*out=NULL,*len=0 => absent/404), 0 not-s3, -1 hard error. */
static int fy_s3_get_own(const char *path, unsigned char **out, size_t *len){
#ifdef FYSICS_S3
    *out=NULL; *len=0;
    if(strncmp(path,"s3://",5)!=0) return 0;
    fy_s3_ensure();
    if(!g_fy_s3){ fprintf(stderr,"zarr: S3 client init failed for %s\n",path); return -1; }
    const int max_tries = 5;
    for(int try = 0; try < max_tries; try++){
        if(try > 0){ struct timespec ts; long ms = 200L << (try - 1);
            ts.tv_sec = ms / 1000; ts.tv_nsec = (ms % 1000) * 1000000L; nanosleep(&ts, NULL); }
        s3_response r; memset(&r,0,sizeof r);
        s3_status st=s3_get(g_fy_s3,path,&r);
        if(st==S3_OK && r.status==200 && r.body){
            *out=(unsigned char*)r.body; *len=r.body_len;
            r.body=NULL; s3_response_free(&r);
            return 1;
        }
        if(r.status==404){ s3_response_free(&r); return 1; }
        if(try == max_tries-1)
            fprintf(stderr,"zarr: S3 GET failed after %d tries (status=%d http=%ld): %s\n",
                    max_tries, (int)st, (long)r.status, path);
        s3_response_free(&r);
    }
    return -1;
#else
    (void)path;(void)out;(void)len; return 0;
#endif
}

/* tiny field grab from a flat .zarray json: finds "key": <int> (first occurrence). */
static long json_int(const char *s, const char *key) {
    const char *p = strstr(s, key);
    if (!p) return -1;
    p += strlen(key);
    while (*p && (*p < '0' || *p > '9') && *p != '-') p++;
    return strtol(p, NULL, 10);
}
/* grab the i-th integer after "key" (for "shape":[a,b,c] / "chunks":[a,b,c]). */
static long json_int_at(const char *s, const char *key, int idx) {
    const char *p = strstr(s, key);
    if (!p) return -1;
    p += strlen(key);
    for (int i = 0; i <= idx; i++) {
        while (*p && (*p < '0' || *p > '9') && *p != '-') p++;
        if (i < idx) { while (*p && (*p >= '0' && *p <= '9')) p++; }
    }
    return strtol(p, NULL, 10);
}

int fy_zarr_open(fy_zarr *z, const char *root) {
    memset(z, 0, sizeof(*z));
    snprintf(z->root, sizeof(z->root), "%s", root);
    char path[2048];
    snprintf(path, sizeof(path), "%s/0/.zarray", root);
    char buf[4096]; size_t n = 0;
    if (strncmp(root, "s3://", 5) == 0) {
        /* fetch .zarray over S3 (a tiny JSON object). read_chunk's S3 path needs a fixed length,
         * so do a dedicated small GET here. */
        unsigned char tmp[4096]; memset(tmp,0,sizeof tmp);
        int s3 = fy_s3_get(path, tmp, sizeof(tmp)-1, 0, 0);
        if (s3 == 0) { fprintf(stderr,"zarr: S3 not compiled in (need FYSICS_S3) for %s\n", path); return 1; }
        if (s3 < 0)  { fprintf(stderr,"zarr: S3 error fetching %s\n", path); return 1; }
        memcpy(buf, tmp, sizeof(buf)-1); buf[sizeof(buf)-1]=0; n = strlen(buf);
        if (n == 0) { fprintf(stderr, "zarr: no %s (S3)\n", path); return 1; }
    } else {
        FILE *f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "zarr: no %s\n", path); return 1; }
        n = fread(buf, 1, sizeof(buf) - 1, f); fclose(f);
    }
    buf[n] = 0;
    z->shape[0] = json_int_at(buf, "\"shape\"", 0);
    z->shape[1] = json_int_at(buf, "\"shape\"", 1);
    z->shape[2] = json_int_at(buf, "\"shape\"", 2);
    z->chunk[0] = json_int_at(buf, "\"chunks\"", 0);
    z->chunk[1] = json_int_at(buf, "\"chunks\"", 1);
    z->chunk[2] = json_int_at(buf, "\"chunks\"", 2);
    z->fill = (unsigned char)((json_int(buf, "\"fill_value\"") > 0) ? json_int(buf, "\"fill_value\"") : 0);
    if (z->shape[0] <= 0 || z->chunk[0] <= 0) { fprintf(stderr, "zarr: bad header\n"); return 1; }
    return 0;
}

static void mkdirs(const char *path) {
    char tmp[2048]; snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++)
        if (*p == '/') { *p = 0; mkdir(tmp, 0775); *p = '/'; }
}

/* write the .zarray for an OUTPUT zarr (shape + chunk in voxels). */
int fy_zarr_create(fy_zarr *z, const char *root, const long shape[3], const long chunk[3]) {
    memset(z, 0, sizeof(*z));
    snprintf(z->root, sizeof(z->root), "%s", root);
    for (int i = 0; i < 3; i++) { z->shape[i] = shape[i]; z->chunk[i] = chunk[i]; }
    char path[2048]; snprintf(path, sizeof(path), "%s/0/.zarray", root);
    mkdirs(path);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "zarr: cannot create %s\n", path); return 1; }
    fprintf(f,
        "{\"chunks\":[%ld,%ld,%ld],\"compressor\":null,\"dtype\":\"|u1\",\"fill_value\":0,"
        "\"order\":\"C\",\"shape\":[%ld,%ld,%ld],\"zarr_format\":2,\"dimension_separator\":\"/\"}",
        chunk[0], chunk[1], chunk[2], shape[0], shape[1], shape[2]);
    fclose(f);
    return 0;
}

/* ---- chunk VIEW: local chunk files are mmap'd (no intermediate copy -- a halo
 * read that uses a 16-voxel sliver of a neighbor chunk faults only the pages it
 * touches, instead of fread'ing all 2MB into a scratch buffer and copying again).
 * S3 chunks (and the rare mmap failure) land in the caller's scratch buffer.
 * v->p == NULL => chunk absent (sparse: use fill). */
typedef struct { const unsigned char *p; size_t len; void *map; size_t maplen;
                 unsigned char *heap; /* owned S3 body (freed on close) */ } fy_chunk_view;

static int open_chunk(const fy_zarr *z, long cz, long cy, long cx, long cn,
                      unsigned char *scratch, fy_chunk_view *v) {
    v->p = NULL; v->len = 0; v->map = NULL; v->maplen = 0; v->heap = NULL;
    char path[2048];
    snprintf(path, sizeof(path), "%s/0/%ld/%ld/%ld", z->root, cz, cy, cx);
    { unsigned char *body=NULL; size_t blen=0;
      int s3 = fy_s3_get_own(path, &body, &blen);             /* S3 path (zero scratch copy) */
      if (s3 == -1) return 1;
      if (s3 == 1) {
          if (!body) return 0;                                /* 404 = sparse fill */
          v->heap = body; v->p = body;
          v->len = blen < (size_t)cn ? blen : (size_t)cn;
          return 0;
      } }
    (void)scratch;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;                                    /* absent = sparse fill */
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return 0; }
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m != MAP_FAILED) {
        v->map = m; v->maplen = (size_t)st.st_size;
        v->p = m; v->len = (size_t)st.st_size < (size_t)cn ? (size_t)st.st_size : (size_t)cn;
        return 0;
    }
    FILE *f = fopen(path, "rb");                             /* mmap-failure fallback */
    if (!f) return 0;
    size_t got = fread(scratch, 1, cn, f); fclose(f);
    v->p = scratch; v->len = got < (size_t)cn ? got : (size_t)cn;
    return 0;
}
static void close_chunk(fy_chunk_view *v) { if (v->map) munmap(v->map, v->maplen); free(v->heap); }

/* read an arbitrary region [z0..z0+dz) into `out` (dz*dy*dx, C order), assembling from chunks. */
int fy_zarr_read(const fy_zarr *z, long z0, long y0, long x0, long dz, long dy, long dx,
                 unsigned char *out) {
    long C0 = z->chunk[0], C1 = z->chunk[1], C2 = z->chunk[2];
    long cn = C0 * C1 * C2;
    unsigned char *cb = (unsigned char *)malloc(cn);   /* max chunk size */
    if (!cb) return 1;
#ifdef FYSICS_S3
    /* S3: BATCH the region's chunk GETs over pooled connections (one serial GET per
     * chunk pays TLS+TTFB each, ~13 MB/s per thread; the batch multiplexes them).
     * Responses are indexed by chunk coordinate for the assembly loop below. */
    s3_response *batch = NULL; char *bpaths = NULL;
    long bcz0=0,bcy0=0,bcx0=0, bnz=0,bny=0,bnx=0;
    if (strncmp(z->root, "s3://", 5) == 0) {
        bcz0 = z0 / C0; bcy0 = y0 / C1; bcx0 = x0 / C2;
        bnz = (z0+dz-1)/C0 - bcz0 + 1; bny = (y0+dy-1)/C1 - bcy0 + 1; bnx = (x0+dx-1)/C2 - bcx0 + 1;
        long nb = bnz*bny*bnx;
        if (nb >= 2 && nb <= 1024) {
            fy_s3_ensure();
            batch  = calloc(nb, sizeof(s3_response));
            bpaths = malloc((size_t)nb * 2048);
            s3_range_req *reqs = malloc(nb * sizeof(s3_range_req));
            if (batch && bpaths && reqs && g_fy_s3) {
                for (long i = 0; i < nb; i++) {
                    long qz = bcz0 + i/(bny*bnx), r = i%(bny*bnx), qy = bcy0 + r/bnx, qx = bcx0 + r%bnx;
                    snprintf(bpaths + i*2048, 2048, "%s/0/%ld/%ld/%ld", z->root, qz, qy, qx);
                    reqs[i].url = bpaths + i*2048; reqs[i].offset = 0; reqs[i].length = 0;
                }
                if (s3_get_batch(g_fy_s3, reqs, nb, fy_batch_conc(), batch) != S3_OK) {
                    for (long i = 0; i < nb; i++) s3_response_free(&batch[i]);
                    free(batch); batch = NULL;   /* transport failure -> serial fallback (retries) */
                }
            } else { free(batch); batch = NULL; }
            free(reqs);
        }
    }
#endif
    for (long gz = z0; gz < z0 + dz; ) {
        long cz = gz / C0, lz = gz - cz * C0, hz = (lz + (z0 + dz - gz) < C0) ? lz + (z0 + dz - gz) : C0;
        long ez = (z->shape[0] - cz * C0 < C0) ? (z->shape[0] - cz * C0) : C0;   /* TRUE z-extent */
        for (long gy = y0; gy < y0 + dy; ) {
            long cy = gy / C1, ly = gy - cy * C1, hy = (ly + (y0 + dy - gy) < C1) ? ly + (y0 + dy - gy) : C1;
            long ey = (z->shape[1] - cy * C1 < C1) ? (z->shape[1] - cy * C1) : C1;
            for (long gx = x0; gx < x0 + dx; ) {
                long cx = gx / C2, lx = gx - cx * C2, hx = (lx + (x0 + dx - gx) < C2) ? lx + (x0 + dx - gx) : C2;
                long ex = (z->shape[2] - cx * C2 < C2) ? (z->shape[2] - cx * C2) : C2;
                long cn2 = ez * ey * ex;
                fy_chunk_view cv;
#ifdef FYSICS_S3
                int from_batch = 0;
                if (batch) {
                    long bi = (cz-bcz0)*bny*bnx + (cy-bcy0)*bnx + (cx-bcx0);
                    s3_response *r = &batch[bi];
                    if (r->status == 200 && r->body) {
                        cv.p = (const unsigned char*)r->body;
                        cv.len = (size_t)r->body_len < (size_t)cn2 ? (size_t)r->body_len : (size_t)cn2;
                        cv.map = NULL; cv.maplen = 0; cv.heap = NULL; from_batch = 1;
                    } else if (r->status == 404 || r->status == 0) {
                        cv.p = NULL; cv.len = 0; cv.map = NULL; cv.maplen = 0; cv.heap = NULL; from_batch = 1;
                    }
                    /* other statuses: fall through to the serial path (retries) */
                }
                if (!from_batch)
#endif
                if (open_chunk(z, cz, cy, cx, cn2, cb, &cv) != 0) {
#ifdef FYSICS_S3
                    if (batch) { long nb=bnz*bny*bnx; for (long i=0;i<nb;i++) s3_response_free(&batch[i]); free(batch); free(bpaths); }
#endif
                    free(cb); return 1;
                }
                /* clamp the copy span to the chunk's true extent (halo reads near the volume edge
                 * must not read past the stored data). */
                long hzc = hz < ez ? hz : ez, hyc = hy < ey ? hy : ey, hxc = hx < ex ? hx : ex;
                long want = hxc - lx;
                for (long zz = lz; zz < hzc && want > 0; zz++)
                    for (long yy = ly; yy < hyc; yy++) {
                        size_t soff = ((size_t)zz * ey + yy) * ex + lx;       /* TRUE strides */
                        unsigned char *dst = out + (((cz*C0+zz - z0) * dy) + (cy*C1+yy - y0)) * dx + (cx*C2+lx - x0);
                        if (!cv.p || soff >= cv.len) { memset(dst, z->fill, want); continue; }
                        long avail = (long)(cv.len - soff);
                        long c = avail < want ? avail : want;
                        memcpy(dst, cv.p + soff, c);
                        if (c < want) memset(dst + c, z->fill, want - c);     /* short file tail */
                    }
#ifdef FYSICS_S3
                if (!from_batch)
#endif
                close_chunk(&cv);
                gx = cx * C2 + hx;
            }
            gy = cy * C1 + hy;
        }
        gz = cz * C0 + hz;
    }
#ifdef FYSICS_S3
    if (batch) { long nb=bnz*bny*bnx; for (long i=0;i<nb;i++) s3_response_free(&batch[i]); free(batch); free(bpaths); }
#endif
    free(cb);
    return 0;
}

/* write a full output chunk (cz,cy,cx in CHUNK units) from `buf` (chunk-sized or smaller edge). */
int fy_zarr_write_chunk(const fy_zarr *z, long cz, long cy, long cx,
                        const unsigned char *buf, long bz, long by, long bx) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/0/%ld/%ld/%ld", z->root, cz, cy, cx);
    mkdirs(path);   /* creates every parent dir of the chunk file (incl. 0/cz/cy) */
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "zarr: cannot write %s\n", path); return 1; }
    fwrite(buf, 1, (size_t)bz * by * bx, f);
    fclose(f);
    return 0;
}
