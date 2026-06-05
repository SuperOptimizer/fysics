# In-Plane (surface-tangent) Texture Enhancer

`fy_texture_enhance` / `fy_texture_enhance_auto` — a volume-in → volume-out filter
that **lifts the fine in-plane "crackle" surface texture** on papyrus sheets (the
faint surface texture that ink sits in, per Vesuvius Grand Prize findings) while
**suppressing cross-sheet / layering variation** and **not amplifying noise**.

It is the **complement of the coherence-diffusion filter** (`diffusion.c`): where CED
*smooths along* sheets, this *high-passes along* sheets. Both share the same primitive —
the per-voxel structure-tensor orientation field.

## Honest scope (read this first)

We have **no ground-truth ink labels**, so this is **not an ink detector** and is not
validated as one. It is the *right transform if ink is a fine in-plane texture signal*:
it enhances in-plane texture over noise and over layering, auto-calibrated. The claims
that are actually **measured and proven** (synthetic test + real PHerc data) are:

1. in-plane texture-band energy goes **up**;
2. texture/noise ratio **improves** (the noise gate, not just uniform scaling);
3. cross-sheet / layering variation is **not amplified** (distinguishes it from a plain
   unsharp mask, which boosts the layering step too).

Do **not** report this as "detects ink." Report it as "enhances in-plane texture over
noise and over layering, auto-calibrated."

## Method

A **steered, noise-gated, direction-gated unsharp mask** built on the orientation field.

1. **Orientation field — the same structure tensor as CED.** Presmooth `u` at noise
   scale `sigma`, central-difference gradient, form the 6 outer-product components
   `grad u ⊗ grad u`, smooth each with `G_rho` (3-pass box ≈ Gaussian, the integration
   scale). Eigen-decompose `J` per voxel (`eig3_sym_vec`, identical to `diffusion.c`):
   the large-eigenvalue eigenvector is the **sheet normal**; the two small-eigenvalue
   eigenvectors are the **in-plane tangents** `t1,t2`.
   *Axis-order note:* the tensor is built from `(gx,gy,gz)`, so eigenvector components
   come out in **(x,y,z)** order — steered sampling maps them to `(z,y,x)` displacements.

2. **In-plane steered low-pass.** At each voxel, average `u` only *within the tangent
   plane* — sample along `±k·t1` and `±k·t2` (k=1..R, R≈`inplane_scale`) by trilinear
   interpolation, never across the normal. A cross-sheet step (layering) lies along the
   normal, so the in-plane average does **not** cross it and it does **not** appear in
   the detail. A fine in-plane ripple **does** appear in the detail.

3. **Detail + two gates.** `detail = u − inplane_lowpass`.
   - **Noise gate (Wiener-style):** `detail · det²/(det² + (c·nf)²)`. Below the noise
     scale `nf` this suppresses *quadratically* (flat-region noise cut ~5× at c=2) while
     supra-noise texture passes ~untouched — this is what makes texture/**noise** improve
     rather than uniformly scaling noise up.
   - **Directional gate:** `hp²/(hp² + hn²)` with `hp=|u−inplane_lp|`,
     `hn=|u−normal_lp|`. Genuine in-plane texture has `hn≈0` → pass; a layering edge (or
     the residual leak from a slightly mis-estimated normal) has `hn≫hp` → suppress.
     This is the second guarantee the layering step is not amplified.

4. **Output:** `out = u + gain · noise_gate(detail) · directional_gate`.

## Auto-calibration (`fy_texture_enhance_auto`)

No knobs except `strength ∈ {1,2,3}`. What it measures from the data:

| Parameter      | Source |
|----------------|--------|
| `sigma`        | `fy_estimate_noise` → presmooth scale, clamped `[0.6, 1.2]` vox |
| `noise gate`   | the **detail's own** robust scale `1.4826·MAD(detail)` (sentinel `noise_floor<0`). Measured directly on the gated quantity — more reliable than mapping an absolute intensity-domain noise model, which the bright sheets/layering corrupt |
| `rho`          | ≈ half the measured cross-sheet **layer spacing** (1-D autocorrelation, same idea as `diffusion.c`'s `estimate_sheet_spacing`), clamped `[1.5, 6]` |
| `inplane_scale`| the measured **in-plane texture scale** — lag of the first in-plane autocorrelation minimum, measured *within the dominant tangent plane*, clamped `[1, 4]` |
| `gain`         | `strength`: 1→0.8 (gentle), 2→1.5 (normal), 3→2.5 (strong) |

## Parameter guidance (explicit `fy_texture_enhance`)

- `sigma` 0.6–1.0: gradient presmooth. Larger → more stable orientation, blurrier texture.
- `rho` 2–4: orientation integration scale. Too large bleeds across thin sheets.
- `gain` 0.5–3: unsharp strength on the gated in-plane detail.
- `inplane_scale` 1.5–4: steered low-pass radius. The texture band is detail *finer*
  than this within the plane. Set it just above the crackle half-period.
- `noise_floor`: detail-signal units. `0` disables gating; `<0` = auto (recommended,
  uses `1.4826·MAD`); `>0` = explicit threshold ≈ the detail-band noise σ.

## Streaming / tiling halo

The filter is **single-pass and local** (no iteration). Per-side halo:

```
halo = ceil(3*sigma)         (presmooth)
     + 3*ceil(rho)           (structure-tensor integration box)
     + ceil(inplane_scale)   (steered low-pass + normal-pass reach)
     + 2                     (gradient/safety margin)
```

See `fy_texture_enhance_halo(sigma, rho, inplane_scale)`. Feed each tile its inner
region plus this margin, keep the inner region. Tiles ≤ 256³ recommended.

## Validation summary

- **Synthetic** (`tests/test_fysics.c::test_texture_enhance`): bright sheet + fine
  in-plane ripple + a cross-sheet layering step + noise. Asserts (a) texture-band energy
  ↑ (≈2.2×), (b) layering edge slope **not** amplified (≈1.0×) while a matched **plain**
  unsharp mask boosts it ≈2.6×, (c) texture/noise ↑ (≈1.26×) with flat-region noise
  barely moved.
- **Real data** (`analysis/texture_test/`): a textured PHerc0139 cube
  (`128³`, mean 62 / std 41), `fy_texture_enhance_auto(strength=2)`, peak RSS 177 MB.
  All three claims hold: in-plane texture power **↑1.23×**, texture/noise **↑1.05×**,
  cross-sheet layering **1.02×** (not amplified) and flat-region noise **1.06×** (not
  amplified). **Honest comparison to a plain unsharp mask:** by the raw selectivity ratio
  (texture-gain ÷ layering-gain) plain unsharp actually scores *higher* on this cube
  (its crackle is broadband), so the steered filter does **not** win that ratio and we do
  not claim it does. Its real advantage is **restraint at matched lift**: forced to the
  same ~1.43× texture gain, the steered filter adds only ~3 % excess layering and ~11 %
  noise vs ~11 % layering / ~14 % noise for plain — a *layering-safe, noise-safe*
  enhancer, modest but real on this data. Full numbers, figures and the matched-gain
  table in `analysis/texture_test/RESULTS.md`.
