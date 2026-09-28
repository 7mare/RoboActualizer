"""Per-channel mean/std over a latent cache -> <cache_dir>/latent_norm_stats.json.

Float64 streaming sums (population std); used as model.latent_norm_stats_path when training.

    python scripts/compute_latent_norm_stats.py ./data/int2_clip4/libero
"""
import argparse
import json
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", help="e.g. ./data/int2_clip4/libero")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache_dir = args.cache_dir.rstrip("/")
    out_path = args.out or os.path.join(cache_dir, "latent_norm_stats.json")
    if os.path.exists(out_path):
        raise SystemExit(f"Refusing to overwrite {out_path}")

    # All suites must come from the same encoder settings.
    metas = {}
    for suite in sorted(os.listdir(cache_dir)):
        mp = os.path.join(cache_dir, suite, "latents_meta.json")
        if os.path.isfile(mp):
            metas[suite] = json.load(open(mp))
    if not metas:
        raise SystemExit(f"No latents_meta.json under {cache_dir}")
    ident = {k: {f: m[f] for f in ("embed_dim", "clip_len", "frame_interval",
                                   "latent_dtype", "backbone_model_name") if f in m}
             for k, m in metas.items()}
    first = next(iter(ident.values()))
    for suite, v in ident.items():
        if v != first:
            raise SystemExit(f"Suites disagree: {suite}={v} vs {first}")
    embed_dim = int(first["embed_dim"])
    print(f"cache={cache_dir}  settings={first}  suites={list(metas)}")

    files = []
    for suite in sorted(metas):
        d = os.path.join(cache_dir, suite)
        files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.startswith("episode_")]
    print(f"{len(files)} episode files")

    s = torch.zeros(embed_dim, dtype=torch.float64)
    s2 = torch.zeros(embed_dim, dtype=torch.float64)
    n = 0
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        z = torch.load(fp, map_location="cpu", weights_only=False)["latents"]
        x = z.reshape(-1, z.shape[-1]).to(torch.float64)
        s += x.sum(0)
        s2 += (x * x).sum(0)
        n += x.shape[0]
        if i % 100 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  {i}/{len(files)}  rows={n:,}  {el:.0f}s  eta={el / i * (len(files) - i):.0f}s",
                  flush=True)

    mean = s / n
    var = (s2 / n - mean * mean).clamp_min(0.0)
    std = var.sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or bool((std <= 0).any()):
        raise SystemExit("Invalid stats (NaN/Inf or std <= 0)")

    payload = {
        "backbone_model_name": first.get("backbone_model_name"),
        "embed_dim": embed_dim,
        "cache_dir": cache_dir,
        "clip_len": int(first["clip_len"]),
        "frame_interval": int(first["frame_interval"]),
        "latent_dtype": str(first.get("latent_dtype", "float16")),
        "num_files": len(files),
        "num_elements": int(n * embed_dim),
        "method": ("full-pass float64 sum/sumsq over all cached latents (population std), "
                   f"computed {time.strftime('%Y-%m-%d')}"),
        # Summary only: mean of channel means, and the pooled std over all elements.
        "global_mean": float(mean.mean()),
        "global_std": float((s2.sum() / (n * embed_dim) - float(mean.mean()) ** 2) ** 0.5),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    import hashlib
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    print(f"\nWrote {out_path}")
    print(f"  sha256[:16]={sha[:16]}  global_mean={payload['global_mean']:.6f} "
          f"global_std={payload['global_std']:.6f}")
    print(f"  max |mean|={float(mean.abs().max()):.3f}  "
          f"std range=[{float(std.min()):.3f}, {float(std.max()):.3f}]")


if __name__ == "__main__":
    main()
