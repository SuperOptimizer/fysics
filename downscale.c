/* downscale.c -- 2x volumetric LOD kernels (ported from volume-compressor).
 *
 * STRICTLY WITHIN-CELL: each output voxel is a function of exactly its own
 * 2x2x2 parents -- a downscale never reaches across chunk/tile boundaries, so
 * pyramid cascades need NO halo and tiles stay independent. Zero-preserving
 * (an all-zero cell stays zero), which is what makes coarse pyramid levels
 * valid occupancy masks for the fine ones.
 *
 *   FY_DS_BOX  -- 2x2x2 mean. Anti-aliases but washes thin sheets toward gray.
 *   FY_DS_CBOX -- contrast-maintaining box: the mean pushed toward the cell's
 *                 max-deviation voxel (bright sheet OR dark gap) by `alpha`,
 *                 so thin papyrus structure stays visible at coarse zoom.
 */
#include "fysics.h"

static inline unsigned char ds_u8(float v){ return (unsigned char)(v<0?0:v>255?255:(v+0.5f)); }

static inline unsigned char ds_cell(const unsigned char c8[8], unsigned acc,
                                    fy_ds_method method, float alpha){
    if (acc == 0) return 0;
    float mean = (c8[0]+c8[1]+c8[2]+c8[3]+c8[4]+c8[5]+c8[6]+c8[7]) * 0.125f;
    if (method == FY_DS_BOX) return ds_u8(mean);
    float best=mean, bestdev=-1.0f;            /* FY_DS_CBOX */
    for(int i=0;i<8;++i){ float d=(float)c8[i]-mean; float ad=d<0?-d:d; if(ad>bestdev){bestdev=ad;best=c8[i];} }
    return ds_u8(mean + alpha*(best - mean));
}

void fy_downscale2x(const unsigned char *in, int nx, int ny, int nz,
                    unsigned char *out, int *oxp, int *oyp, int *ozp,
                    fy_ds_method method, float alpha){
    int ox=(nx+1)/2, oy=(ny+1)/2, oz=(nz+1)/2;
    if(ox<1)ox=1; if(oy<1)oy=1; if(oz<1)oz=1;
    if(oxp)*oxp=ox; if(oyp)*oyp=oy; if(ozp)*ozp=oz;
    for (int z=0; z<oz; ++z)
    for (int y=0; y<oy; ++y)
    for (int x=0; x<ox; ++x) {
        int ix=x*2, iy=y*2, iz=z*2;
        unsigned char c8[8]; int n=0; unsigned acc=0;
        for(int dz=0;dz<2;++dz){ int z2=iz+dz; if(z2>=nz)z2=nz-1;
          for(int dy=0;dy<2;++dy){ int y2=iy+dy; if(y2>=ny)y2=ny-1;
            for(int dx=0;dx<2;++dx){ int x2=ix+dx; if(x2>=nx)x2=nx-1;
              unsigned char v=in[((size_t)z2*ny+y2)*nx+x2]; c8[n++]=v; acc|=v; }}}
        out[((size_t)z*oy + y)*ox + x] = ds_cell(c8,acc,method,alpha);
    }
}
