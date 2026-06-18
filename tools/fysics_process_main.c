/* fysics_process_main.c -- single CLI for the whole-volume preprocessing pipeline.
 *
 * Reads metadata.json physics via the tiny python stub (read_meta.py, prints key=value),
 * fills fy_pipeline_cfg with BM18-validated defaults, and runs the pure-C 2-pass pipeline
 * (calibrate -> global stats -> parallel per-tile chain) zarr->zarr.
 *
 * Usage:
 *   fysics_process IN_ZARR OUT_ZARR [--tile N] [--profile P]
 *                  [--deconv] [--musica] [--no-air-zero] [--meta PATH]
 */
#include "fysics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <limits.h>
#include <unistd.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s IN_ZARR OUT_ZARR [options]\n"
        "  IN_ZARR        input local zarr root (raw u8, level 0/)\n"
        "  OUT_ZARR       output zarr root (created)\n"
        "options:\n"
        "  --tile N       tile edge in voxels (default 128)\n"
        "  --profile P    export profile: 'conservative' (fidelity: keep faint material, full\n"
        "                 res, gentle denoise) or 'aggressive' (readability: cut faint\n"
        "                 material below the sheets, stronger denoise)\n"
        "  --air-cut-aggr A  air-cut aggressiveness 0..1 (void-peak->valley, default 0.0)\n"
        "  --deconv       STORE matched-Wiener deconv (BM18 default: OFF, view-time only)\n"
        "  --musica       apply MUSICA viewing enhancement (default OFF)\n"
        "  --no-dering    disable residual-ring detect+subtract (default ON, gated on detection)\n"
        "  --dering-center Y X  rotation axis in voxels (default: volume center; use the\n"
        "                 metadata rotation_axis_position if it differs)\n"
        "  --no-air-zero  disable the air-zero masking stage (default ON)\n"
        "  --meta PATH    metadata.json path (default: IN_ZARR/metadata.json)\n",
        prog);
}

/* set a cfg field from a "key=value" metadata line */
static void apply_meta(fy_pipeline_cfg *c, const char *key, double v) {
    if      (!strcmp(key, "delta_beta"))            c->delta_beta = v;
    else if (!strcmp(key, "energy_kev"))            c->energy_kev = v;
    else if (!strcmp(key, "distance_mm"))           c->distance_mm = v;
    else if (!strcmp(key, "pixel_um"))              c->pixel_um = v;
    else if (!strcmp(key, "unsharp_sigma"))         c->unsharp_sigma = v;
    else if (!strcmp(key, "unsharp_coeff"))         c->unsharp_coeff = v;
    else if (!strcmp(key, "machine_current_start")) c->machine_current_start = v;
    else if (!strcmp(key, "machine_current_stop"))  c->machine_current_stop = v;
    else if (!strcmp(key, "window_lo"))             c->window_lo = v;
    else if (!strcmp(key, "window_hi"))             c->window_hi = v;
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(argv[0]); return 2; }
    const char *in_zarr = argv[1], *out_zarr = argv[2];
    int tile = 128, do_deconv = 0, do_musica = 0, do_air_zero = 1, scratch_passes = 5, do_dering = 1;
    double dering_cy = -1, dering_cx = -1;
    double air_cut_aggr = 0.0, denoise_k = 0.0;   /* 0 -> default 4.2 inside */
    const char *profile = NULL;
    char meta_path[PATH_MAX]; meta_path[0] = 0;

    for (int i = 3; i < argc; i++) {
        if      (!strcmp(argv[i], "--tile") && i+1 < argc)  tile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--air-cut-aggr") && i+1 < argc) air_cut_aggr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--profile") && i+1 < argc) profile = argv[++i];
        else if (!strcmp(argv[i], "--deconv"))              do_deconv = 1;
        else if (!strcmp(argv[i], "--musica"))              do_musica = 1;
        else if (!strcmp(argv[i], "--no-air-zero"))         do_air_zero = 0;
        else if (!strcmp(argv[i], "--no-dering"))           do_dering = 0;
        else if (!strcmp(argv[i], "--dering-center") && i+2 < argc) { dering_cy = atof(argv[++i]); dering_cx = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--scratch-passes") && i+1 < argc) scratch_passes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--meta") && i+1 < argc)  snprintf(meta_path, sizeof(meta_path), "%s", argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }
    /* PROFILES set every aggressiveness knob coherently (per-volume numbers still measured). */
    if (profile) {
        /* Profiles differ only in air-cut aggressiveness + denoise strength; both full-res.
         * (Downsampling was removed entirely -- the volume is not meaningfully oversampled and the
         * auto-factor rested on an unreliable PSF sigma.) */
        if (!strcmp(profile, "conservative")) {        /* FIDELITY: keep faint material */
            air_cut_aggr = 0.0; denoise_k = 3.0;
        } else if (!strcmp(profile, "aggressive")) {   /* READABILITY: cut faint material */
            air_cut_aggr = 1.0; denoise_k = 3.0;   /* conservative's gentler denoise */
        } else { fprintf(stderr, "unknown profile: %s (conservative|aggressive)\n", profile); return 2; }
    }
    if (!meta_path[0]) snprintf(meta_path, sizeof(meta_path), "%s/metadata.json", in_zarr);

    /* ---- BM18 defaults ---- */
    fy_pipeline_cfg cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.delta_beta = 1000.0; cfg.energy_kev = 78.0; cfg.distance_mm = 220.0; cfg.pixel_um = 2.4;
    cfg.unsharp_sigma = 1.2; cfg.unsharp_coeff = 4.0;
    cfg.window_lo = 0.0; cfg.window_hi = 0.0;
    cfg.do_deconv = do_deconv;                 /* STORED deconv off for BM18 unless --deconv */
    cfg.use_matched_deconv = 1; cfg.psf_sigma_vox = 1.0; cfg.deconv_tikhonov = 0.05;
    cfg.guided_eps = 0.0;                      /* 0 -> calibrate from flat_nf inside */
    cfg.do_air_zero = do_air_zero; cfg.air_cut_u8 = -1; cfg.air_cut_band = 8; cfg.air_thresh = 0.05;
    cfg.air_cut_aggr = air_cut_aggr; cfg.denoise_k = denoise_k;
    cfg.scratch_passes = scratch_passes;
    cfg.do_normalize = 0; cfg.norm_lo = -1; cfg.norm_hi = -1;   /* no recenter/stretch by default */
    cfg.do_zdrift = 1; cfg.zdrift_factor = NULL;
    cfg.do_dering = do_dering; cfg.dering_cy = dering_cy; cfg.dering_cx = dering_cx;
    cfg.do_musica = do_musica; cfg.musica_p = 0.7; cfg.musica_levels = 4; cfg.musica_core = 0.0;  /* 0.7: gentler gain (less brightening) */

    /* ---- locate read_meta.py: next to the binary, an installed ../share or the source
     * tools/ tree, then $FYSICS_READMETA, then PATH-relative fallback. ---- */
    char script[PATH_MAX] = {0};
    char exedir[PATH_MAX] = {0};
    char exe[PATH_MAX]; ssize_t en = readlink("/proc/self/exe", exe, sizeof(exe)-1);
    if (en > 0) { exe[en] = 0; snprintf(exedir, sizeof(exedir), "%s", dirname(exe)); }
    const char *env = getenv("FYSICS_READMETA");
    char cands[5][PATH_MAX]; int nc = 0;
    if (env) snprintf(cands[nc++], PATH_MAX, "%s", env);
    if (exedir[0]) {
        snprintf(cands[nc++], PATH_MAX, "%s/read_meta.py", exedir);          /* installed bin/ */
        snprintf(cands[nc++], PATH_MAX, "%s/../tools/read_meta.py", exedir); /* build tree */
    }
    snprintf(cands[nc++], PATH_MAX, "%s/../tools/read_meta.py", in_zarr);    /* unlikely, harmless */
    snprintf(cands[nc++], PATH_MAX, "read_meta.py");                         /* cwd / PATH */
    for (int i = 0; i < nc; i++) if (access(cands[i], R_OK) == 0) { memcpy(script, cands[i], sizeof(script)); script[sizeof(script)-1] = 0; break; }
    if (!script[0]) snprintf(script, sizeof(script), "read_meta.py");

    char cmd[PATH_MAX*2];
    snprintf(cmd, sizeof(cmd), "python3 '%s' '%s' 2>/dev/null", script, meta_path);
    FILE *pp = popen(cmd, "r");
    int meta_lines = 0;
    if (pp) {
        char line[256];
        while (fgets(line, sizeof(line), pp)) {
            char *eq = strchr(line, '='); if (!eq) continue;
            *eq = 0; char *key = line; double v = atof(eq+1);
            apply_meta(&cfg, key, v); meta_lines++;
        }
        pclose(pp);
    }
    if (meta_lines == 0)
        fprintf(stderr, "[warn] no metadata read from %s -- using BM18 defaults\n", meta_path);

    /* WINDOW -> physical air threshold. The u8 is a clipped linear window of physical attenuation
     * mu: u8 = clip((mu - window_lo)/(window_hi - window_lo), 0, 1)*255 (nabu 32-bit conversion
     * window, then the tif->u8 re-window). So the physical air/void level (mu ~ 0) maps to
     * u8 ~= (0 - window_lo)/(window_hi - window_lo)*255. Use that as air_thresh (a [0,1] frac) so
     * the air-cut has a PHYSICS-anchored floor (used when no clean histogram valley exists, and as
     * the conservative end of the void-peak->valley interpolation). */
    if (cfg.window_hi > cfg.window_lo) {
        double air_frac = (0.0 - cfg.window_lo) / (cfg.window_hi - cfg.window_lo);
        if (air_frac < 0) air_frac = 0; if (air_frac > 1) air_frac = 1;
        cfg.air_thresh = air_frac;   /* e.g. window [-0.04,0.22] -> mu=0 at u8 ~39 -> 0.154 */
    }

    printf("fysics_process\n");
    printf("  in       : %s\n", in_zarr);
    printf("  out      : %s\n", out_zarr);
    printf("  meta     : %s (%d keys)\n", meta_path, meta_lines);
    printf("  physics  : db=%.1f E=%.1fkeV D=%.0fmm px=%.3fum unsharp=(%.2f,%.2f)\n",
           cfg.delta_beta, cfg.energy_kev, cfg.distance_mm, cfg.pixel_um,
           cfg.unsharp_sigma, cfg.unsharp_coeff);
    printf("  window   : [%.4g, %.4g]\n", cfg.window_lo, cfg.window_hi);
    printf("  stages   : deconv=%d(stored) air_zero=%d normalize=%d zdrift=%d musica=%d\n",
           cfg.do_deconv, cfg.do_air_zero, cfg.do_normalize, cfg.do_zdrift, cfg.do_musica);
    printf("  tile     : %d\n", tile);
    fflush(stdout);

    int rc = fy_run_pipeline(in_zarr, out_zarr, &cfg, tile, 1);
    if (rc != 0) { fprintf(stderr, "pipeline failed (rc=%d)\n", rc); return 1; }

    printf("\nSUMMARY\n");
    printf("  guided_eps   : %.5f\n", cfg.guided_eps);
    printf("  air_cut_u8   : %d (band +/-%d)\n", cfg.air_cut_u8, cfg.air_cut_band);
    printf("  normalize    : %s (lo=%d hi=%d)\n", cfg.norm_lo >= 0 ? "applied" : "skipped", cfg.norm_lo, cfg.norm_hi);
    printf("  halo         : %d\n", cfg.halo);
    printf("  out written  : %s\n", out_zarr);
    return 0;
}
