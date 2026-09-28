"""Download flex-pi/robotwin_3d (RGB + action, no depth) and write the 50-demo subset file.

`huggingface-cli download` silently skips about two thirds of the videos (the server truncates
repo_info().siblings), so files are listed with list_repo_files and fetched one by one.
Existing files are skipped, so the script can be re-run to resume.

    python scripts/download_robotwin_3d.py --subset-only   # 50 demos per task (training set)
    python scripts/download_robotwin_3d.py                 # subset, then everything else
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "flex-pi/robotwin_3d"
OUT = Path(os.environ.get("ROBOTWIN_3D_DIR", "/mnt/ssd1/brookdu/robotwin_3d"))
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CHUNK = 1000  # chunks_size in meta/info.json

_print_lock = threading.Lock()

# Anonymous HF resolves are limited to 3000 per 300 s; a 429 surfaces as LocalEntryNotFoundError.
# Raise HF_RATE_Q when HF_TOKEN is set.
RATE_Q = int(os.environ.get("HF_RATE_Q", "2700"))
RATE_W = float(os.environ.get("HF_RATE_W", "300"))
_rate_lock = threading.Lock()
_rate_hits: list[float] = []


def _rate_gate():
    """Block until a request fits in the sliding RATE_Q / RATE_W window."""
    while True:
        with _rate_lock:
            now = time.time()
            cut = now - RATE_W
            while _rate_hits and _rate_hits[0] < cut:
                _rate_hits.pop(0)
            if len(_rate_hits) < RATE_Q:
                _rate_hits.append(now)
                return
            wait = _rate_hits[0] + RATE_W - now + 0.5
        time.sleep(max(wait, 0.5))


def _paths_for_episode(ep: int):
    c = ep // CHUNK
    yield f"data/chunk-{c:03d}/episode_{ep:06d}.parquet"
    for cam in CAMS:
        yield f"videos/chunk-{c:03d}/observation.images.{cam}/episode_{ep:06d}.mp4"


def _retry_after() -> float:
    """Seconds left in the current rate-limit window (t= in the ratelimit header)."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://huggingface.co/datasets/{REPO}/resolve/main/meta/info.json",
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            hdr = r.headers.get("ratelimit", "")
    except Exception as e:  # noqa: BLE001
        hdr = getattr(getattr(e, "headers", None), "get", lambda *_: "")("ratelimit") or ""
    for part in str(hdr).split(";"):
        if part.strip().startswith("t="):
            try:
                return float(part.strip()[2:]) + 2
            except ValueError:
                pass
    return 30.0


def _fetch(path: str, retries: int = 8) -> str:
    """Download one file unless present; on 429 wait for the window to reset."""
    dst = OUT / path
    if dst.exists() and dst.stat().st_size > 0:
        return "skip"
    for a in range(retries):
        _rate_gate()
        try:
            hf_hub_download(REPO, path, repo_type="dataset", local_dir=str(OUT))
            return "ok"
        except Exception as e:  # noqa: BLE001
            if a == retries - 1:
                with _print_lock:
                    print(f"  ✗ {path}: {type(e).__name__} {str(e)[:120]}", flush=True)
                return "fail"
            wait = _retry_after() if a >= 1 else 2 ** a
            time.sleep(min(wait, 120))
    return "fail"


def _run(paths, label: str, workers: int):
    todo = [p for p in paths if not (OUT / p).exists()]
    print(f"\n=== {label}: {len(paths)} files, {len(todo)} to fetch ===", flush=True)
    if not todo:
        return 0
    done = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_fetch, p): p for p in todo}
        for f in as_completed(futs):
            r = f.result()
            done += 1
            fail += r == "fail"
            if done % 500 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (len(todo) - done) / max(rate, 1e-9)
                print(f"  {done}/{len(todo)}  failed {fail}  {rate:.1f} files/s  "
                      f"eta {eta/60:.0f} min", flush=True)
    return fail


def subset_episode_ids(task_map: dict, per_task: int, seed: int = 42):
    """Per task: shuffle its episode range with a fresh Random(seed) and keep the first N."""
    ids = []
    for t in sorted(task_map):
        rng = task_map[t]
        rr = list(range(rng["episode_start"], rng["episode_end"] + 1))
        if per_task < len(rr):
            random.Random(seed).shuffle(rr)
            rr = rr[:per_task]
        ids.extend(sorted(rr))
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subset-only", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    # 1. meta and root-level files
    all_files = api.list_repo_files(REPO, repo_type="dataset")
    print(f"{len(all_files):,} files in the repo")
    meta = [f for f in all_files
            if (f.startswith("meta/") or "/" not in f) and "depth" not in f]
    _run(meta, "meta", min(a.workers, 4))

    # 2. subset
    tm_path = OUT / "task_episode_map.json"
    if not tm_path.exists():
        print("task_episode_map.json is missing", file=sys.stderr)
        return 1
    task_map = json.loads(tm_path.read_text())["tasks"]
    sub = subset_episode_ids(task_map, a.per_task, a.seed)
    print(f"\nsubset: {len(task_map)} tasks x {a.per_task} = {len(sub):,} episodes (seed={a.seed})")
    (OUT / f"subset_perTask{a.per_task}_seed{a.seed}.json").write_text(
        json.dumps({"per_task": a.per_task, "seed": a.seed,
                    "num_episodes": len(sub), "episode_ids": sub}, indent=1)
    )

    have = set(all_files)
    sub_paths = [p for ep in sub for p in _paths_for_episode(ep) if p in have]
    fail = _run(sub_paths, f"subset ({len(sub):,} episodes)", a.workers)
    print(f"\nsubset done, {fail} failed")

    if a.subset_only:
        return 0

    # 3. remaining episodes
    rest = sorted(set(range(27500)) - set(sub))
    rest_paths = [p for ep in rest for p in _paths_for_episode(ep) if p in have]
    fail = _run(rest_paths, f"remaining {len(rest):,} episodes", a.workers)
    print(f"\nall done, {fail} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
