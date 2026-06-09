# Fused preprocess -> matter-compressor export (fysics)

Vendored from matter-compressor (see VENDORED.md): the callback-driven codec + archive used to
build a .v3 archive from an uncompressed zarr (local OR s3://), with our preprocessing fused in.

## Pieces (vendor/matter/)
- mc_codec.{c,h}     DCT-16 + dead-zone quant + range coder. mc_enc_block(vox, air, ...).
- mc_archive.{c,h} + mc_archive_api.h   sparse-tree + dense-256^3-chunk archive, 8 LODs, 128KB
    user-metadata carve-out. SOURCE-AGNOSTIC: mc_build(src_fn, ud, opts) pulls voxels via a
    callback. Air mask derived from voxel==0 -> our air-zero output feeds it directly.
  (matter-compressor has NO S3/zarr -- that's the exporter's job. fysics zarr_io does S3 itself.)

## The fused export (tools/vca_export.c, CMake target `vca_export`)
1. fy_calibrate ONCE on the input zarr (local/s3).
2. fy_process_chunk every 128^3 chunk (OpenMP) -> cache (denoise + air-zero, air==0).
3. mc_build_to_file(prep_voxel, ...) -- the callback serves preprocessed voxels; matter builds
   all LODs + the archive. No intermediate zarr.

## Verified (1024^3 test volume)
1.07GB -> 27.3MB (39.3x) at q=8; round-trip PSNR 39.5 dB; air-leak 0.0000%; reads local AND S3.
