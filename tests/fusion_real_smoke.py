#!/usr/bin/env python3
"""Real-data smoke test for the FUSION pipeline + transform-usage POLICY.

Exercises, on the REAL PHerc0139 shipped transform.json:
  1. load + parse the shipped 3x4 matrix and landmark pairs,
  2. confirm fy_affine_from_points reproduces the shipped matrix from the shipped
     landmark pairs (proves the C fit kernel is correct on real correspondences),
  3. confirm the landmark clouds are an internally consistent similarity at the
     voxel-size ratio (the matrix is valid IN THE LANDMARK FRAME),
  4. VERIFY image agreement against the masked zarrs IF the cached real cubes exist,
     and report GOOD / APPROXIMATE / MISALIGNED.

See FUSION.md for the full policy and the honest finding (on PHerc0139 the shipped
transform is MISALIGNED in the masked-zarr voxel frame, so naive fusion does not
beat the fine scan alone -- a registration problem, not a fusion-math problem).

Run single-threaded:
    OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 python3 tests/fusion_real_smoke.py
"""
import ctypes as C
import json, os, sys
from itertools import combinations
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.join(ROOT, "build", "libfysics.so")
# shipped transform (committed copy in the superresolution analysis dir)
TJ = "/home/forrest/superresolution/analysis/fusion/transform_1129_to_2399.json"
VOX_FINE, VOX_COARSE = 1.129, 2.399


def affine_from_points(lib, src, dst, model, ransac, thresh):
    f64 = C.POINTER(C.c_double)
    lib.fy_affine_from_points.restype = C.c_int
    lib.fy_affine_from_points.argtypes = [f64, f64, C.c_int, C.c_int, C.c_int,
                                          C.c_double, f64, C.POINTER(C.c_int), f64]
    src = np.ascontiguousarray(src, np.float64)
    dst = np.ascontiguousarray(dst, np.float64)
    n = src.shape[0]
    M = np.zeros(12, np.float64); rms = C.c_double(0.0)
    rc = lib.fy_affine_from_points(src.ctypes.data_as(f64), dst.ctypes.data_as(f64),
                                   n, model, ransac, thresh,
                                   M.ctypes.data_as(f64), None, C.byref(rms))
    return rc, M.reshape(3, 4), rms.value


def main():
    if not os.path.exists(LIB):
        sys.exit(f"build the shared lib first (missing {LIB})")
    if not os.path.exists(TJ):
        sys.exit(f"missing shipped transform {TJ}")
    lib = C.CDLL(LIB)
    tj = json.load(open(TJ))
    M = np.array(tj["transformation_matrix"], np.float64).reshape(3, 4)
    fl = np.array(tj["fixed_landmarks"], np.float64)   # 2.399um frame
    ml = np.array(tj["moving_landmarks"], np.float64)  # 1.129um frame
    fails = 0

    # 1+2: fy_affine_from_points reproduces the shipped matrix from its landmarks
    rc, Mfit, rms = affine_from_points(lib, ml, fl, 0, 0, 0.0)
    err = np.abs(Mfit - M).max()
    print(f"[1] shipped matrix self-residual on its landmarks: {rms:.3f} voxels")
    print(f"[2] fy_affine_from_points(moving->fixed) vs shipped: max|dM|={err:.2e}")
    ok = rc == 0 and err < 1e-6
    print("    -> C affine fit reproduces shipped matrix:", "OK" if ok else "FAIL")
    fails += (not ok)

    # 3: landmark clouds are a consistent similarity at the voxel-size ratio
    df = np.array([np.linalg.norm(ml[i]-ml[j]) for i, j in combinations(range(len(ml)), 2)])
    dc = np.array([np.linalg.norm(fl[i]-fl[j]) for i, j in combinations(range(len(fl)), 2)])
    ratio = (df/dc).mean(); rstd = (df/dc).std()
    exp = VOX_COARSE/VOX_FINE
    print(f"[3] landmark pairwise-dist ratio = {ratio:.4f} +/- {rstd:.4f} "
          f"(expect voxel ratio {exp:.4f})")
    ok = abs(ratio-exp) < 0.02
    print("    -> landmark frames internally consistent:", "OK" if ok else "FAIL")
    fails += (not ok)
    scale = np.linalg.det(M[:, :3]) ** (1.0/3.0)
    print(f"    matrix linear scale = {scale:.4f} == 1.129/2.399 = {1/exp:.4f} "
          "(=> landmark units are voxel-like, NOT mm)")

    # 4: image-agreement verdict from cached real cubes if present
    base = "/home/forrest/superresolution/analysis/fusion"
    vj = os.path.join(base, "transform_verification.json")
    if os.path.exists(vj):
        v = json.load(open(vj))
        print(f"[4] image-agreement verdict (cached): {v['verdict']}")
        print(f"    matched-region full-res NCC = {v['matched_region_lm11_fullres_ncc']:.3f}; "
              f"global silhouette IoU = {v['global_silhouette_IoU_lv6']:.2f}; "
              f"matrix-free landmark-patch NCC ~ "
              f"{np.mean(list(v['matrix_free_landmark_patch_ncc'].values())):.3f}")
        print("    -> shipped transform MISALIGNED in masked-zarr voxel frame; see FUSION.md")
    else:
        print("[4] image-agreement verdict: run analysis/fusion/verify_lm11.py "
              "to populate transform_verification.json")

    print()
    print("ALL PASSED" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
