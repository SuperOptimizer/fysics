/* clahe.c -- 2D Contrast-Limited Adaptive Histogram Equalization (grayscale).
 *
 * Standard CLAHE: tile the image, build a clipped histogram CDF per tile, then
 * bilinearly interpolate the tile mappings per pixel (so there are no tile seams).
 * Used as the LOCAL stage of GLCAE (glcae.c). Input/output normalized [0,1].
 */
#include "fysics.h"
#include <math.h>
#include <stdlib.h>

static void tile_cdf(const float *img, int ny, int nx,
                     int y0, int y1, int x0, int x1,
                     int nbins, float clip_limit, float *cdf) {
    long *h = (long *)calloc(nbins, sizeof(long));
    long total = 0;
    for (int y = y0; y < y1; y++) {
        size_t row = (size_t)y * nx;
        for (int x = x0; x < x1; x++) {
            int b = (int)(img[row + x] * (nbins - 1) + 0.5f);
            if (b < 0) b = 0; if (b >= nbins) b = nbins - 1;
            h[b]++; total++;
        }
    }
    if (total == 0) { for (int i = 0; i < nbins; i++) cdf[i] = (float)i / (nbins - 1); free(h); return; }
    long clip = (long)(clip_limit * total / nbins); if (clip < 1) clip = 1;
    long excess = 0;
    for (int i = 0; i < nbins; i++) if (h[i] > clip) { excess += h[i] - clip; h[i] = clip; }
    long add = excess / nbins;
    for (int i = 0; i < nbins; i++) h[i] += add;
    long acc = 0;
    for (int i = 0; i < nbins; i++) { acc += h[i]; cdf[i] = (float)acc / total; }
    free(h);
}

int fy_clahe2d(const float *in, float *out, int ny, int nx,
               int tiles_y, int tiles_x, int nbins, float clip_limit) {
    if (tiles_y < 1) tiles_y = 8; if (tiles_x < 1) tiles_x = 8;
    if (nbins < 16) nbins = 256; if (clip_limit <= 0) clip_limit = 2.0f;
    int ntiles = tiles_y * tiles_x;
    float *cdfs = (float *)malloc((size_t)ntiles * nbins * sizeof(float));
    if (!cdfs) return 1;
    for (int ty = 0; ty < tiles_y; ty++)
        for (int tx = 0; tx < tiles_x; tx++) {
            int y0 = ty * ny / tiles_y, y1 = (ty + 1) * ny / tiles_y;
            int x0 = tx * nx / tiles_x, x1 = (tx + 1) * nx / tiles_x;
            tile_cdf(in, ny, nx, y0, y1, x0, x1, nbins, clip_limit,
                     &cdfs[(size_t)(ty * tiles_x + tx) * nbins]);
        }
    for (int y = 0; y < ny; y++) {
        float fy_ = ((float)y + 0.5f) * tiles_y / ny - 0.5f;
        int ty0 = (int)floorf(fy_); float wy = fy_ - ty0;
        int ty1 = ty0 + 1; if (ty0 < 0) { ty0 = 0; wy = 0; } if (ty1 >= tiles_y) ty1 = tiles_y - 1;
        for (int x = 0; x < nx; x++) {
            float fx = ((float)x + 0.5f) * tiles_x / nx - 0.5f;
            int tx0 = (int)floorf(fx); float wx = fx - tx0;
            int tx1 = tx0 + 1; if (tx0 < 0) { tx0 = 0; wx = 0; } if (tx1 >= tiles_x) tx1 = tiles_x - 1;
            int b = (int)(in[(size_t)y * nx + x] * (nbins - 1) + 0.5f);
            if (b < 0) b = 0; if (b >= nbins) b = nbins - 1;
            #define C(ty,tx) cdfs[(size_t)((ty)*tiles_x+(tx))*nbins + b]
            float c00 = C(ty0,tx0), c01 = C(ty0,tx1), c10 = C(ty1,tx0), c11 = C(ty1,tx1);
            #undef C
            float c0 = c00*(1-wx)+c01*wx, c1 = c10*(1-wx)+c11*wx;
            out[(size_t)y * nx + x] = c0*(1-wy)+c1*wy;
        }
    }
    free(cdfs);
    return 0;
}
