#!/usr/bin/env python3
"""Real-data smoke test for LAYER 1 affine registration (fy_register_affine).

Loads two REAL PHerc0139 cubes copied read-only from the superresolution analysis
dir (1.129um and 2.399um) and runs the C registration on them via ctypes.

IMPORTANT LIMITATION: these two cubes are from DIFFERENT spatial chunk indices, so
they very likely do NOT cover the same physical region of the scroll. This harness
therefore does NOT prove we can recover a correct physical transform between them.
It only proves the API runs end-to-end on real CT data and reports the NCC actually
achieved. Finding truly-corresponding regions is a later (fusion) concern.

The voxel-size ratio (2.399/1.129 ~ 2.125) is seeded into the initial matrix so the
optimizer starts near the correct scale, demonstrating the scale-ratio init path.

Run single-threaded:
    OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 python3 tests/register_real_smoke.py
"""
import ctypes as C
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.join(ROOT, "build", "libfysics.so")
SRC = "/home/forrest/superresolution/analysis/PHerc0139"
HI = os.path.join(SRC, "1.129um", "cube.npy")   # higher resolution (finer voxels)
LO = os.path.join(SRC, "2.399um", "cube.npy")   # lower resolution (coarser voxels)

VOXEL_HI = 1.129
VOXEL_LO = 2.399


def load_lib():
    lib = C.CDLL(LIB)
    f32 = C.POINTER(C.c_float)
    f64 = C.POINTER(C.c_double)
    lib.fy_register_affine.restype = C.c_int
    lib.fy_register_affine.argtypes = [f32, f32, C.c_int, C.c_int, C.c_int, f64, C.c_int]
    lib.fy_ncc_warped.restype = C.c_double
    lib.fy_ncc_warped.argtypes = [f32, f32, C.c_int, C.c_int, C.c_int, f64]
    return lib


def as_f32(a):
    a = np.ascontiguousarray(a.astype(np.float32) / 255.0)
    return a, a.ctypes.data_as(C.POINTER(C.c_float))


def main():
    if not os.path.exists(LIB):
        sys.exit(f"build the shared lib first: cmake --build build  (missing {LIB})")
    lib = load_lib()

    hi = np.load(HI)   # fixed  (1.129um, finer grid)
    lo = np.load(LO)   # moving (2.399um, coarser grid)
    nz, ny, nx = hi.shape
    assert lo.shape == hi.shape, "cubes must share grid for this smoke test"

    fixed_np, fixed = as_f32(hi)
    moving_np, moving = as_f32(lo)

    # Seed initial M with the voxel-size ratio. Both cubes are stored on a 128^3
    # grid, but physically the 2.399um voxels are ~2.125x bigger; to overlay the
    # moving (coarse) scan onto the fixed (fine) grid we scale output->input coords
    # by ratio < 1 (one fine output voxel steps a fraction of a coarse input voxel).
    ratio = VOXEL_HI / VOXEL_LO   # ~0.4706
    M = (C.c_double * 12)(ratio, 0, 0, 0,  0, ratio, 0, 0,  0, 0, ratio, 0)

    ncc_init = lib.fy_ncc_warped(fixed, moving, nz, ny, nx, M)
    print(f"cubes: fixed(1.129um)={hi.shape} moving(2.399um)={lo.shape}")
    print(f"voxel-ratio init (scale={ratio:.4f}):  initial NCC = {ncc_init:.4f}")

    rc = lib.fy_register_affine(fixed, moving, nz, ny, nx, M, 0)  # full affine
    ncc_final = lib.fy_ncc_warped(fixed, moving, nz, ny, nx, M)
    print(f"fy_register_affine rc={rc}")
    print(f"recovered M (3x4, output->input voxel map):")
    Mv = list(M)
    for r in range(3):
        print("   [%+.4f %+.4f %+.4f | %+.3f]" % tuple(Mv[r*4:r*4+4]))
    print(f"final NCC = {ncc_final:.4f}   (improvement {ncc_final-ncc_init:+.4f})")
    print()
    print("NOTE: these cubes are from different spatial chunks and almost")
    print("certainly do NOT overlap physically. A high NCC here would be")
    print("coincidental texture alignment, not a true physical registration.")
    print("This run only demonstrates the API executes on real data.")


if __name__ == "__main__":
    main()
