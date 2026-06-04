# fysics

Fast CPU physics kernels for Vesuvius Challenge scroll volumes.

Pure modern C (no SIMD intrinsics — clean auto-vectorizable code), CPU-only,
dependency-free. Inverts the **known ESRF/nabu reconstruction operators** (Paganin
single-distance phase-retrieval low-pass + unsharp mask) to **sharpen reconstructed
volumes** — a physics deblur with no machine learning and no hallucination, since
it analytically inverts a known transfer function.

Intended for import by viewers/tools (e.g. **vc3d**) as on-the-fly post-processing,
or as batch preprocessing.

## What's physics-grounded vs. generic (read this)

Be honest about what we can defensibly claim on ESRF/BM18 data:

| operation | grounded? | where / how |
|---|---|---|
| **Paganin deconvolution** | ✅ **exact** | metadata gives `delta_beta`, energy, distance; we invert the *exact* nabu filter (verified to 1e-16). Apply to the reconstructed volume. |
| **Unsharp accounting** | ✅ **exact** | metadata gives `coeff`/`sigma`; modeled in the net transfer. |
| **NLM / bilateral denoise** | ✅ safe, generic | denoising is always valid; pairs with deconv (which amplifies noise). Strength tuned empirically (we don't have the exact noise model). Apply *after* deconv. |
| FBP ramp filter inversion | ❌ not clean | applied in sinogram domain before backprojection — cannot be cleanly inverted from the reconstructed volume. **Not implemented.** |
| ring-artifact removal | ⚠️ heuristic | residual rings have no metadata model; standard suppression is heuristic, not a physics inverse. (Could add, clearly labeled.) |

So fysics provides a **physics-exact Paganin deblur** + a **standard denoise pass**.
The deblur is grounded in the metadata; the denoise is a safe complement. We do
*not* claim to invert the FBP filter or rings as "physics."

## What it does

The BM18/ESRF reconstructions apply a Paganin phase-retrieval filter (a deliberate
low-pass that trades resolution for contrast — it makes the near-invisible
carbonized papyrus visible). That filter is analytic and *partly invertible*. fysics
applies a Wiener-regularized inverse of the exact recon transfer:

```
T_paganin(f) = 1 / (1 + delta_beta * lambda * D * pi * f^2)     (matches nabu exactly)
U_unsharp(f) = 1 + coeff * (1 - exp(-2 pi^2 sigma^2 f^2))
H(f)         = T_paganin(f) * U_unsharp(f)
deconv:  F_out = F_in * H / (H^2 + reg)
```

`reg` is the **sharpening-strength dial** (lower = sharper + noisier). The Paganin
parameters (`delta_beta`, `energy`, `distance`, `pixel_size`, unsharp) come from each
volume's `metadata.json`.

## Locality (why it's viewer-friendly)

The transfer is a convolution, so each output voxel depends only on a **local
neighborhood** (~15 voxels for BM18 Paganin), not the whole volume. So you can
sharpen just the region being viewed plus a small halo. Use `fy_kernel_halo()` to
get the margin, process `region + halo`, keep the inner region.

## API (`fysics.h`)

```c
typedef struct {
    double delta_beta, energy_kev, distance_mm, pixel_um, unsharp_sigma, unsharp_coeff;
} fy_physics;

/* sharpen a volume (row-major, x fastest) in place-or-out; reg = strength */
int  fy_deconvolve(const float *in, float *out, int nz, int ny, int nx,
                   const fy_physics *p, double reg);

int  fy_kernel_halo(const fy_physics *p);          /* tile/viewer margin (voxels) */
double fy_recon_transfer(double f_cyc_per_voxel, const fy_physics *p);
/* plus fy_fft1d/3d if you need the FFT directly */
```

### vc3d usage sketch

```c
fy_physics p = { .delta_beta=1000, .energy_kev=78, .distance_mm=220,
                 .pixel_um=2.4, .unsharp_sigma=1.2, .unsharp_coeff=4.0 };
int halo = fy_kernel_halo(&p);                 /* ~15 */
/* fetch the viewed region grown by `halo`, then: */
fy_deconvolve(region_in, region_out, dz, dy, dx, &p, reg);
/* keep the inner [halo..dz-halo] etc. for display */
```

## Whole-volume processing at 20TB+ (streaming)

Real scroll volumes are **20+ TB of dense u8** — they never fit in RAM. fysics is
built for this: it does the per-chunk **math**, and the **caller owns chunk
iteration + I/O** (use your existing fast zarr reader). Operators split two ways:

**LOCAL ops** (deconv, denoise, mask) — each output voxel depends only on a small
neighborhood, so process one chunk **+ a halo** independently (embarrassingly
parallel). Halo = `fy_kernel_halo()` (~15 vox).

**GLOBAL ops** (normalization, GLCAE global stage) — need whole-volume statistics,
so they are **two-pass streaming**:

```c
/* PASS 1: accumulate a tiny 256-bin histogram over every chunk */
fy_hist_state st; fy_hist_init(&st);
for (each chunk)            /* your I/O loop */
    fy_hist_accumulate_u8(&st, chunk_u8, chunk_n);
/* (parallel? accumulate per-thread, then fy_hist_merge) */

/* FINALIZE: compute the mapping ONCE from the global histogram */
int glcae_map[256];  fy_glcae_global_finalize(&st, glcae_map);
unsigned char lo, hi; fy_norm_finalize(&st, 0.5, 99.5, &lo, &hi);

/* PASS 2: apply the SAME mapping to every chunk -> consistent everywhere */
for (each chunk) {
    fy_norm_apply_u8(chunk_u8, tmpf, chunk_n, lo, hi);      /* global normalize */
    fy_glcae_global_apply_u8(chunk_u8, outf, chunk_n, glcae_map);  /* global GLCAE */
    /* + local ops on chunk+halo: fy_deconvolve, fy_bilateral_denoise, ... */
    fy_float_to_u8(outf, out_u8, chunk_n);                  /* write back u8 */
}
```

The global state is a 256-long histogram (a few KB) — **20TB never sits in RAM**,
and a per-slice operation never sees inconsistent global stats. Verified: chunked
two-pass GLCAE == whole-volume processing, **bit-identical**.

## Build (CMake)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build          # correctness tests
./build/bench_fysics            # timing
```

Flags: `-Ofast -ffast-math -funroll-loops -march=native` (override host arch with
`-DFYSICS_ARCH=...` for a portable build). Builds both a static (`libfysics.a`) and
shared (`libfysics.so`) library plus the public header.

## Performance (CPU, -Ofast -march=native)

| region | time | throughput |
|---|---|---|
| 8×512×512 (viewer slab) | ~150 ms | ~14 Mvox/s |
| 64×256×256 | ~420 ms | ~10 Mvox/s |
| 256×256×256 (batch tile) | ~1.6 s | ~10 Mvox/s |

Interactive for viewer-sized regions; the FFT is the cost (self-contained radix-2,
auto-vectorized). A finer FFT (radix-4/split-radix) would speed it further if needed.

## Verification

The C kernels are tested against (a) a naive DFT, (b) FFT round-trip, (c) the exact
nabu Paganin formula, and (d) the Python reference implementation in
[SuperOptimizer/superresolution](https://github.com/SuperOptimizer/superresolution)
(`superres/paganin.py`) — bit-identical on matching sizes.

## License

See `LICENSE`.
