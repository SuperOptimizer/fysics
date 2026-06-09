# Fused preprocess -> v2/v3 VCA export (fysics)

Vendored from volume-compressor (see VENDORED.md). The v2/v3 codec + libs3, used to build a
.v3 archive from an uncompressed zarr (local OR s3://).

## Pieces
- vca/v2codec.{c,h}    DCT-16 + dead-zone quant + range coder. v2_enc_block(vox, air, ...).
- vca/v3archive.c + v3archive_api.h + v3write.h   sparse-tree + dense-256^3-chunk archive.
    Entry: v3_build_from_zarr(root, out, dim, q). Encodes chunk-by-chunk; air mask derived
    from voxel==0 (so our pipeline's air-zero output feeds it directly). Reads local OR s3://.
- libs3/   S3 GET (s3_get, s3_credentials_load, s3_url_is_s3). Needs libcurl.

## Status
- S3 source reading: DONE (upstream v3archive.c, vendored here). Verified: local byte-identical,
  S3 fetches real chunks from the public bucket.
- FUSED single-pass (read s3 zarr -> our preprocess per chunk -> v2/v3 encode): TODO. The seam is
  v3vol_get / load_zarr_vsrc: fill the vsrc chunks with PREPROCESSED data (calibrate once on
  samples, then process each 128^3 chunk with halo, air-zero -> store) instead of raw zarr bytes.

## Build (standalone, until wired into CMake)
  cc -std=c23 -O2 -Ilibs3 -o v3enc vca/v3enc.c vca/v3archive.c vca/v2codec.c \
     libs3/libs3.c $(pkg-config --cflags --libs libcurl) -lm
