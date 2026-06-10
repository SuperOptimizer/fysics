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
#include <errno.h>
#include <time.h>
#ifdef FYSICS_S3
#include "vendor/libs3/libs3.h"
static s3_client *g_fy_s3 = NULL;
static s3_status fy_s3_cred(void *ud, s3_credentials *out){ (void)ud; return s3_credentials_load(NULL,out); }
static void fy_s3_ensure(void){
    if(g_fy_s3) return;
    s3_config cfg; memset(&cfg,0,sizeof cfg);
    s3_credentials probe; memset(&probe,0,sizeof probe);
    if(s3_credentials_load(NULL,&probe)==S3_OK){
        cfg.cred_provider=fy_s3_cred;
        if(probe.region&&probe.region[0]) cfg.region=strdup(probe.region);
        s3_credentials_free(&probe);
    }
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
        if(st==S3_OK && r.status==404){ memset(out,fill,cn); s3_response_free(&r); return 1; }
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

/* read one chunk (cz,cy,cx) into `out`, laid out with the chunk's TRUE stored extent
 * (ez,ey,ex) -- EDGE chunks are stored smaller than the nominal chunk size (matching how
 * fy_zarr_write_chunk wrote them). Fills with z->fill if the chunk is absent (sparse zarr).
 * Returns 0 on success (incl. fill), 1 on a real I/O error (S3 failure after retries). */
static int read_chunk(const fy_zarr *z, long cz, long cy, long cx,
                      long ez, long ey, long ex, unsigned char *out) {
    long cn = ez * ey * ex;
    char path[2048];
    snprintf(path, sizeof(path), "%s/0/%ld/%ld/%ld", z->root, cz, cy, cx);
    int s3 = fy_s3_read_chunk(path, out, cn, z->fill);   /* S3 path (if FYSICS_S3 & s3://) */
    if (s3 == 1) return 0;
    if (s3 == -1) return 1;
    FILE *f = fopen(path, "rb");
    if (!f) { memset(out, z->fill, cn); return 0; }      /* absent = sparse fill */
    size_t got = fread(out, 1, cn, f); fclose(f);
    if ((long)got < cn) memset(out + got, z->fill, cn - got);
    return 0;
}

/* read an arbitrary region [z0..z0+dz) into `out` (dz*dy*dx, C order), assembling from chunks. */
int fy_zarr_read(const fy_zarr *z, long z0, long y0, long x0, long dz, long dy, long dx,
                 unsigned char *out) {
    long C0 = z->chunk[0], C1 = z->chunk[1], C2 = z->chunk[2];
    long cn = C0 * C1 * C2;
    unsigned char *cb = (unsigned char *)malloc(cn);   /* max chunk size */
    if (!cb) return 1;
    for (long gz = z0; gz < z0 + dz; ) {
        long cz = gz / C0, lz = gz - cz * C0, hz = (lz + (z0 + dz - gz) < C0) ? lz + (z0 + dz - gz) : C0;
        long ez = (z->shape[0] - cz * C0 < C0) ? (z->shape[0] - cz * C0) : C0;   /* TRUE z-extent */
        for (long gy = y0; gy < y0 + dy; ) {
            long cy = gy / C1, ly = gy - cy * C1, hy = (ly + (y0 + dy - gy) < C1) ? ly + (y0 + dy - gy) : C1;
            long ey = (z->shape[1] - cy * C1 < C1) ? (z->shape[1] - cy * C1) : C1;
            for (long gx = x0; gx < x0 + dx; ) {
                long cx = gx / C2, lx = gx - cx * C2, hx = (lx + (x0 + dx - gx) < C2) ? lx + (x0 + dx - gx) : C2;
                long ex = (z->shape[2] - cx * C2 < C2) ? (z->shape[2] - cx * C2) : C2;
                if (read_chunk(z, cz, cy, cx, ez, ey, ex, cb) != 0) { free(cb); return 1; }
                /* clamp the copy span to the chunk's true extent (halo reads near the volume edge
                 * must not read past the stored data). */
                long hzc = hz < ez ? hz : ez, hyc = hy < ey ? hy : ey, hxc = hx < ex ? hx : ex;
                for (long zz = lz; zz < hzc; zz++)
                    for (long yy = ly; yy < hyc; yy++) {
                        unsigned char *src = cb + (zz * ey + yy) * ex + lx;   /* TRUE strides */
                        unsigned char *dst = out + (((cz*C0+zz - z0) * dy) + (cy*C1+yy - y0)) * dx + (cx*C2+lx - x0);
                        if (hxc > lx) memcpy(dst, src, hxc - lx);
                    }
                gx = cx * C2 + hx;
            }
            gy = cy * C1 + hy;
        }
        gz = cz * C0 + hz;
    }
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
