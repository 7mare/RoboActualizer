"""Check that the flex-pi/robotwin_3d RGB + action download is complete and decodable.

Checks: files vs the full remote list, zero-byte / .incomplete leftovers, every episode has
1 parquet + 3 videos, meta consistency, and a decoded sample (frames / rows vs meta lengths).

    python scripts/verify_robotwin_3d.py [--sample 100] [--write-missing missing.txt]
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

REPO = "flex-pi/robotwin_3d"
OUT = Path(os.environ.get("ROBOTWIN_3D_DIR", "./data/robotwin_3d"))
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
N_EPISODES = 27500

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok ' if ok else 'BAD'} {name}{(' | ' + detail) if detail else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=30, help="episodes to decode")
    ap.add_argument("--write-missing", type=str, default=None)
    a = ap.parse_args()

    print(f"data dir {OUT}\n")

    print("[1] files vs remote list")
    api = HfApi()
    remote = api.list_repo_files(REPO, repo_type="dataset")
    want = [f for f in remote if "depth" not in f]
    check("remote list not truncated", len(remote) > 200_000, f"{len(remote):,} remote files")

    missing = [f for f in want if not (OUT / f).exists()]
    empty = [f for f in want if (OUT / f).exists() and (OUT / f).stat().st_size == 0]
    check("all expected files present", not missing, f"expected {len(want):,}, missing {len(missing):,}")
    check("no zero-byte files", not empty, f"{len(empty)} empty")

    if a.write_missing and (missing or empty):
        Path(a.write_missing).write_text("\n".join(missing + empty))
        print(f"      missing list written to {a.write_missing}")

    if missing:
        by_kind = defaultdict(int)
        for f in missing:
            by_kind["parquet" if f.endswith(".parquet") else
                    ("video" if f.endswith(".mp4") else "other")] += 1
        print(f"      missing by kind {dict(by_kind)}")
        eps = sorted({int(f.split("episode_")[1][:6]) for f in missing if "episode_" in f})
        print(f"      {len(eps)} episodes affected, first 10: {eps[:10]}")

    print("\n[2] leftover temp files")
    inc = list(OUT.rglob("*.incomplete"))
    check("no .incomplete files", not inc, f"{len(inc)}" if inc else "")

    print("\n[3] per-episode files")
    have_pq, have_vid = set(), defaultdict(set)
    for p in (OUT / "data").rglob("episode_*.parquet"):
        have_pq.add(int(p.stem.split("_")[1]))
    for cam in CAMS:
        for p in (OUT / "videos").rglob(f"observation.images.{cam}/episode_*.mp4"):
            have_vid[cam].add(int(p.stem.split("_")[1]))

    check(f"parquet count = {N_EPISODES}", len(have_pq) == N_EPISODES, f"got {len(have_pq):,}")
    for cam in CAMS:
        check(f"{cam} video count = {N_EPISODES}", len(have_vid[cam]) == N_EPISODES,
              f"got {len(have_vid[cam]):,}")

    full = have_pq & have_vid["cam_high"] & have_vid["cam_left_wrist"] & have_vid["cam_right_wrist"]
    check("complete episodes", len(full) == N_EPISODES,
          f"{len(full):,}/{N_EPISODES:,}")
    holes = sorted(set(range(N_EPISODES)) - full)
    if holes:
        print(f"      {len(holes)} incomplete episodes, first 20: {holes[:20]}")

    print("\n[4] meta")
    try:
        info = json.loads((OUT / "meta" / "info.json").read_text())
        check("info.json parses", True,
              f"episodes={info['total_episodes']:,} frames={info['total_frames']:,} fps={info['fps']}")
        check("info episode count", info["total_episodes"] == N_EPISODES)
        shp = info["features"]["observation.images.cam_high"]["shape"]
        check("camera resolution 240x320", shp[:2] == [240, 320], str(shp))
    except Exception as e:  # noqa: BLE001
        check("info.json parses", False, str(e))
        return 1

    ep_len = {}
    try:
        with open(OUT / "meta" / "episodes.jsonl") as f:
            for line in f:
                d = json.loads(line)
                ep_len[d["episode_index"]] = d["length"]
        check("episodes.jsonl rows", len(ep_len) == N_EPISODES,
              f"{len(ep_len):,}")
        check("total frames match info.json", sum(ep_len.values()) == info["total_frames"],
              f"{sum(ep_len.values()):,}")
    except Exception as e:  # noqa: BLE001
        check("episodes.jsonl parses", False, str(e))

    check("task_episode_map.json exists", (OUT / "task_episode_map.json").exists())

    print(f"\n[5] decode {a.sample} sampled episodes")
    if not full:
        check("complete episodes to sample", False)
        return 1
    import pandas as pd
    try:
        import av
    except ImportError:
        av = None

    rng = random.Random(0)
    picks = rng.sample(sorted(full), min(a.sample, len(full)))
    bad_rows, bad_frames = [], []
    for ep in picks:
        c = ep // 1000
        try:
            df = pd.read_parquet(OUT / f"data/chunk-{c:03d}/episode_{ep:06d}.parquet")
            if ep in ep_len and len(df) != ep_len[ep]:
                bad_rows.append((ep, len(df), ep_len[ep]))
        except Exception as e:  # noqa: BLE001
            bad_rows.append((ep, f"read failed {type(e).__name__}", ep_len.get(ep)))
        if av is None:
            continue
        for cam in CAMS:
            vp = OUT / f"videos/chunk-{c:03d}/observation.images.{cam}/episode_{ep:06d}.mp4"
            try:
                with av.open(str(vp)) as ct:
                    n = sum(1 for _ in ct.decode(video=0))
                if ep in ep_len and n != ep_len[ep]:
                    bad_frames.append((ep, cam, n, ep_len[ep]))
            except Exception as e:  # noqa: BLE001
                bad_frames.append((ep, cam, f"decode failed {type(e).__name__}", ep_len.get(ep)))

    check("parquet rows match meta", not bad_rows,
          f"{len(bad_rows)} mismatches: {bad_rows[:3]}" if bad_rows else f"{len(picks)} ok")
    if av is None:
        print("      av not installed; skipping video decode")
    else:
        check("video frames match meta", not bad_frames,
              f"{len(bad_frames)} mismatches: {bad_frames[:3]}" if bad_frames else f"{len(picks)*3} ok")

    print("\n" + "=" * 62)
    print(f"passed {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("failed:")
        for f in FAIL:
            print("  -", f)
        print("\nRe-run scripts/download_robotwin_3d.py to fetch what is missing.")
        return 1
    print("data complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
