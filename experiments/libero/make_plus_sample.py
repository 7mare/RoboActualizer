"""Build the frozen LIBERO-Plus evaluation sample (2,640 tasks), run once.

Per suite 660 tasks, stratified over the 7 perturbation categories (Camera and Robot 95, others 94),
drawn without replacement from sorted candidates with random.Random(seed). The output is
deterministic given task_classification.json and the seed.

    python experiments/libero/make_plus_sample.py --out experiments/libero/libero_plus_sample_2640.txt
"""

import argparse
import json
import os

# Category -> dimension name, as in summarize_results_plus.py.
CATEGORY_TO_DIM = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
# Fixed stratification order.
CATEGORY_ORDER = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]
PER_SUITE = 660
# 94 x 7 = 658 per suite; Camera and Robot get one extra each.
BONUS_CATEGORIES = {"Camera Viewpoints", "Robot Initial States"}


def _resolve_classification_path(explicit):
    """Same lookup as summarize_results_plus._resolve_classification_path."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    cfg_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    cfg_file = os.path.join(cfg_dir, "config.yaml")
    if os.path.exists(cfg_file):
        try:
            import yaml

            with open(cfg_file) as f:
                cfg = yaml.safe_load(f)
            root = cfg.get("benchmark_root")
            if root:
                candidates.append(os.path.join(root, "benchmark", "task_classification.json"))
        except Exception:
            pass
    try:
        import libero.libero as ll

        candidates.append(
            os.path.join(os.path.dirname(ll.__file__), "benchmark", "task_classification.json")
        )
    except Exception:
        pass
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "task_classification.json not found; pass --classification. Tried: " + str(candidates)
    )


def _quota(category: str) -> int:
    return 95 if category in BONUS_CATEGORIES else 94


def build_sample(classification_path: str, seed: int):
    """{suite: sorted 0-based task_ids} (660 per suite)."""
    import random

    with open(classification_path) as f:
        data = json.load(f)

    rng = random.Random(seed)
    sample = {}
    for suite in SUITES:
        entries = data.get(suite)
        if entries is None:
            raise KeyError(f"suite missing from classification: {suite}")
        # category -> 0-based task ids (id - 1)
        by_cat = {c: [] for c in CATEGORY_ORDER}
        for e in entries:
            cat = e.get("category", "")
            if cat in by_cat:
                by_cat[cat].append(int(e["id"]) - 1)
        picked = []
        for cat in CATEGORY_ORDER:
            pool = sorted(by_cat[cat])
            q = _quota(cat)
            if len(pool) < q:
                raise ValueError(
                    f"{suite}/{cat} has only {len(pool)} tasks for a quota of {q}."
                )
            picked.extend(rng.sample(pool, q))
        if len(picked) != PER_SUITE:
            raise AssertionError(f"{suite}: picked {len(picked)}, expected {PER_SUITE}")
        sample[suite] = sorted(picked)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output file, one suite,task_id per line")
    ap.add_argument("--classification", default=None, help="task_classification.json (auto-located by default)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    class_path = _resolve_classification_path(args.classification)
    print(f"Using task_classification: {class_path}")
    print(f"seed = {args.seed}")

    sample = build_sample(class_path, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for suite in SUITES:
            for tid in sample[suite]:
                f.write(f"{suite},{tid}\n")
                total += 1
    print(f"\nWrote {total} tasks -> {args.out}")

    # Per-category summary.
    with open(class_path) as fh:
        data = json.load(fh)
    print("\n=== per-suite category distribution ===")
    for suite in SUITES:
        id2cat = {int(e["id"]) - 1: e.get("category", "") for e in data[suite]}
        counts = {c: 0 for c in CATEGORY_ORDER}
        for tid in sample[suite]:
            counts[id2cat[tid]] += 1
        line = "  ".join(f"{CATEGORY_TO_DIM[c]}={counts[c]}" for c in CATEGORY_ORDER)
        print(f"{suite:16s} total={len(sample[suite])}  {line}")


if __name__ == "__main__":
    main()
