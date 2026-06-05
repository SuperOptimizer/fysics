# Coherence-Enhancing Anisotropic Diffusion (Weickert) — `fy_coherence_diffusion`

A volume-in → volume-out filter that **cleans papyrus sheets for tracing**: it smooths
**along** sheet surfaces (denoise, make sheets continuous) but **not across** them, so the
**thin dark gaps between two touching layers are preserved** — CED never merges adjacent
sheets. A plain Gaussian / isotropic blur of matched strength *would* fill those gaps;
CED keeps them because diffusion across the sheet normal is suppressed.

Reference: J. Weickert, *"Coherence-Enhancing Diffusion Filtering"*, IJCV 31(2/3), 1999.

## API

```c
int fy_coherence_diffusion(const float *in, float *out, int nz, int ny, int nx,
                           double sigma, double rho, double tau, int n_iters,
                           double coherence_alpha);
int fy_coherence_diffusion_halo(double sigma, double rho, int n_iters);
```

Row-major, x-fastest (`idx=(z*ny+y)*nx+x`), `float` data, returns `0` on success
(`1` on alloc failure). `out` may alias a distinct buffer from `in`; `in` is not modified.
Work buffers are allocated internally (12 volumes of `nz*ny*nx` floats).

## Method

1. **Structure tensor** `J_rho = G_rho * (∇u_sigma ⊗ ∇u_sigma^T)`:
   presmooth `u` with a Gaussian `G_sigma` (noise scale), take the central-difference
   gradient, form the 6 unique components of the `3×3` outer product per voxel, then
   smooth each component with `G_rho` (integration / sheet-coherence scale). `G_rho` is a
   fast 3-pass separable box filter (≈ Gaussian by CLT).
2. **Eigen-analysis** of the symmetric `J` per voxel → eigenvalues `μ1≥μ2≥μ3` and
   orthonormal eigenvectors (analytic, same closed form as `eig3_sym` in `sheetness.c`,
   extended to return eigenvectors via null-space cross products + Gram-Schmidt). On a
   **sheet** (planar structure) the gradient points *across* the sheet, so `μ1` is large
   (its eigenvector = the **sheet normal**) and `μ2,μ3` are small (the two **in-plane**
   directions, within the sheet).
3. **Diffusion tensor** `D`: same eigenvectors, **remapped** eigenvalues so diffusion is
   **strong along the in-plane directions** and **weak along the normal**. Weickert's
   coherence-enhancing diffusivities:
   - `λ_normal = alpha`  (across the sheet — tiny, so the gap survives)
   - `λ_plane  = alpha + (1-alpha)·exp(-C / κ²)`  (within the sheet — → 1 where coherent),
     with the planarity-contrast coherence measure `κ² = (μ1-μ3)²`. `C` is set
     data-adaptively to the mean of `κ²` over the volume on the first iteration, so one
     `alpha` works across intensity ranges. In a coherent sheet `λ_plane → 1` (smooth a
     lot in-plane) while `λ_normal` stays `= alpha`; in flat noise (`κ²≈0`) all three
     → `alpha` (mild isotropic smoothing). `D = Σ_k λ_k v_k v_kᵀ`.
4. **Evolve** `u_{t+1} = u_t + tau · div(D ∇u)`, explicit Euler, `n_iters` steps. The
   divergence uses a central-difference flux form: assemble `F = D ∇u` (full `3×3` D times
   the central-difference gradient of the current `u`), then take `div F` by central
   differences. Reflecting (Neumann, zero-flux) boundaries conserve total intensity.

The structure tensor is **rebuilt every iteration** from the evolving `u` (true CED, not a
frozen tensor), so the orientation field tracks the sheets as they sharpen.

## Parameters & guidance

| param | meaning | typical | effect |
|---|---|---|---|
| `sigma` | noise-scale presmoothing for the gradient | 0.5–1.0 vox | larger = more robust orientation, slightly blurs fine sheets |
| `rho` | integration / sheet-coherence scale | 2–4 vox | larger = orientation averaged over a wider patch (more continuous sheets), too large smears across the gap |
| `tau` | explicit time step | ≤ 0.12 (3D stable) | larger = faster but `>~0.15` can go unstable in 3D |
| `n_iters` | explicit iterations | 3–30 | more = more smoothing (and slowly more gap erosion) |
| `coherence_alpha` | base diffusivity across the normal | 0.001–0.05 | **the gap knob**: smaller = harder gap preservation but less in-plane cleaning; larger = more cleaning, gaps soften |

**Conservative (max gap preservation, gentle clean):** `sigma 0.7, rho 3, tau 0.12,
n_iters 10, alpha 0.001-0.01` — ~98% gap depth retained, modest noise drop. Use when
adjacent layers are extremely close and must not touch.

**Balanced (recommended for tracing prep):** `sigma 1.0, rho 4, tau 0.12, n_iters 30,
alpha 0.05` — validated on real PHerc data: ~18% high-pass-noise drop on sheets, coherence
0.874→0.909, **92% of inter-sheet gap depth retained vs 67% for a matched Gaussian**.

CED exposes a smooth, monotone **clean-vs-gap tradeoff** through `n_iters` and `alpha`:
pushing `alpha→0.2-0.5` and 30+ iters reaches ~36-39% noise drop but erodes gaps to
~85%. Stay at `alpha≤0.05` when gap preservation is the priority (it usually is, for
tracing). The key property — verified on real data — is that **at any matched smoothing
strength CED preserves the gaps far better than a Gaussian.**

## Streaming / tiling (for vc3d)

The filter is **local**: per iteration the diffusion stencil reaches 1 voxel, and the
per-iteration structure-tensor build adds the presmooth (`~3·sigma`) and the 3-pass box
(`~3·rho`) halo. The recommended per-side halo to feed a tile so the inner result is
seam-free is:

```
halo = ceil(3*sigma) + 3*ceil(rho) + n_iters + 2   (voxels)
```

returned by `fy_coherence_diffusion_halo(sigma, rho, n_iters)`. For the balanced preset
that is **47 voxels/side**; for the conservative preset, **24 voxels/side**. Process a tile
plus this margin, keep only the inner region. Tile ≤ 256³ keeps peak RSS modest
(12 work volumes; a 128³ tile peaks at ~270 MB resident in the validation harness incl.
Python/numpy).

## Tests

`tests/test_fysics.c :: test_coherence_diffusion_gap` — synthetic two-sheet / thin-dark-gap
volume + noise. Asserts (a) sheet-interior noise drops >40% and (b) the inter-sheet gap
contrast is retained >70% of the clean value **and** ≥1.3× better than a matched Gaussian
blur. Passes: noise halves (0.084→0.042), gap contrast 100% retained while the Gaussian
collapses it to 3%.

Real-data validation (PHerc0139, public bucket): see
`analysis/diffusion_test/RESULTS.md` and the before/CED/Gaussian slice figures.
