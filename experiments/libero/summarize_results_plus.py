"""Summarize LIBERO-Plus results by the 7 perturbation dimensions (aggregated over the 4 suites).

Dimensions come from LIBERO-Plus task_classification.json (task_id = id - 1).

    python experiments/libero/summarize_results_plus.py --output_dir <eval output dir>
"""

import argparse
import json
import os
from collections import defaultdict

# LIBERO-Plus category -> dimension name (column order of the paper table).
CATEGORY_TO_DIM = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
DIM_ORDER = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def _resolve_classification_path(explicit):
    candidates = []
    if explicit:
        candidates.append(explicit)
    # 1) benchmark_root in $LIBERO_CONFIG_PATH/config.yaml
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
    # 2) the installed libero package
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


def build_taskid_to_dim(classification_path):
    """{suite: {0-based task_id: dim}}."""
    with open(classification_path) as f:
        data = json.load(f)
    mapping = {}
    unknown = set()
    for suite, entries in data.items():
        m = {}
        for e in entries:
            task_id = int(e["id"]) - 1  # id 1-based -> task_id 0-based
            cat = e.get("category", "")
            dim = CATEGORY_TO_DIM.get(cat)
            if dim is None:
                unknown.add(cat)
                continue
            m[task_id] = dim
        mapping[suite] = m
    if unknown:
        print(f"[warn] skipped unknown categories: {sorted(unknown)}")
    return mapping


def summarize(output_dir, classification_path, quiet=False):
    """With quiet=True, return the stats without printing (used when summarizing many runs)."""
    _p = (lambda *a, **k: None) if quiet else print
    id2dim = build_taskid_to_dim(classification_path)

    # Micro-average over trials per dimension.
    dim_stats = {d: {"trials": 0, "successes": 0, "tasks": 0} for d in DIM_ORDER}
    per_suite_seen = defaultdict(int)
    missing_class = 0

    for suite in SUITES:
        suite_dir = os.path.join(output_dir, suite)
        if not os.path.isdir(suite_dir):
            continue
        for fn in os.listdir(suite_dir):
            if not fn.startswith("gpu") or not fn.endswith("_results.json"):
                continue
            task_id = int(fn.split("_")[1].replace("task", ""))
            with open(os.path.join(suite_dir, fn)) as f:
                r = json.load(f)
            dim = id2dim.get(suite, {}).get(task_id)
            if dim is None:
                missing_class += 1
                continue
            s = dim_stats[dim]
            s["trials"] += int(r["total_episodes"])
            s["successes"] += int(r["successes"])
            s["tasks"] += 1
            per_suite_seen[(dim, suite)] += 1

    _p("\n=== LIBERO-Plus results by perturbation dimension ===")
    header = f"{'Dimension':<12}{'Tasks':>8}{'Trials':>9}{'Success%':>10}"
    _p(header)
    _p("-" * len(header))
    dim_rates = []
    total_trials = total_succ = 0
    for d in DIM_ORDER:
        s = dim_stats[d]
        rate = (100.0 * s["successes"] / s["trials"]) if s["trials"] else float("nan")
        dim_rates.append(rate)
        total_trials += s["trials"]
        total_succ += s["successes"]
        _p(f"{d:<12}{s['tasks']:>8}{s['trials']:>9}{rate:>9.1f}")

    valid = [r for r in dim_rates if r == r]  # drop nan
    macro = sum(valid) / len(valid) if valid else float("nan")
    micro = (100.0 * total_succ / total_trials) if total_trials else float("nan")
    _p("-" * len(header))
    _p(f"{'Avg(macro)':<12}{'':>8}{'':>9}{macro:>9.1f}   # paper Table 3 'Avg' = mean of the 7 per-dimension rates")
    _p(f"{'Avg(micro)':<12}{sum(dim_stats[d]['tasks'] for d in DIM_ORDER):>8}"
          f"{total_trials:>9}{micro:>9.1f}   # pooled over all tasks")
    if missing_class:
        _p(f"\n[warn] {missing_class} result file(s) have a task_id missing from the "
           f"classification json; ignored.")

    _p("\n=== Paper-style row (Camera Robot Language Light Background Noise Layout | Avg) ===")
    cells = " ".join(f"{r:5.1f}" if r == r else "  -  " for r in dim_rates)
    _p(f"{cells}  | {macro:5.1f}")

    return {
        "by_dimension": {d: dim_stats[d] for d in DIM_ORDER},
        "avg_macro": macro,
        "avg_micro": micro,
        # Convenience for batch summaries: success rates ordered by DIM_ORDER, plus short
        # aliases for the two averaging conventions.
        "dim_rates": dim_rates,
        "macro": macro,
        "micro": micro,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True, help="eval output dir (<suite>/gpu*_task*_results.json)")
    ap.add_argument("--classification", default=None, help="task_classification.json (auto-located by default)")
    ap.add_argument("--json_out", default=None, help="optional json output")
    args = ap.parse_args()

    class_path = _resolve_classification_path(args.classification)
    print(f"Using task_classification: {class_path}")
    result = summarize(args.output_dir, class_path)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote summary json: {args.json_out}")


if __name__ == "__main__":
    main()
