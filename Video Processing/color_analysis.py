"""
Color Analysis Script
---------------------
Samples VIDEOS_PER_SOURCE videos per source folder and FRAMES_PER_VIDEO frames
per video. For each frame, detects the person (same Otsu+morph pipeline as the
main processing script) and measures pixel statistics for 5 regions:

  background    – pixels outside the detected person blob (non-person area)
  face_top      – top 15 % of the person blob (usually head / face)
  skin_ycrcb    – pixels INSIDE the person blob that fall in a YCrCb skin range
                  (robust across skin tones; captures hands, face, arms)
  outfit_space  – middle 20-60 % of the person blob by height (torso / clothing)
  outfit_nonskin– person pixels that do NOT match the skin detector (clothing+hair)

Per region, reports:
  R / G / B  mean  ±  std   (8-bit, 0-255)
  H / S / V  mean  (HSV 0-179 / 0-255 / 0-255)
  pixel count

At the end, prints a diff table comparing every source vs the baseline
(sign videos_new_2_17).  Saves raw per-frame data to color_analysis.csv.

Usage:
    python39  color_analysis.py
"""

import csv
import os
import re
import shutil
import uuid
import subprocess
from pathlib import Path

import cv2
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(r"/path/to/dataset")
FFMPEG           = "ffmpeg.exe"
CSV_OUT          = BASE_DIR / "color_analysis.csv"
BASELINE_SOURCE  = "sign videos_new_2_17"

VIDEOS_PER_SOURCE = 3   # videos sampled per source (evenly spaced)
FRAMES_PER_VIDEO  = 5   # frames extracted per video (evenly spaced)
MORPH_FRAC        = 0.005  # morph kernel = frame_height * MORPH_FRAC  (match main script)

# YCrCb skin detection range  (Y, Cr, Cb)
# Works for South/South-East Asian skin tones; tweak if detections look wrong.
SKIN_LOW  = np.array([ 0, 130,  70], dtype=np.uint8)
SKIN_HIGH = np.array([255, 180, 130], dtype=np.uint8)

SOURCES = [
    # BASE_DIR / "sign videos",
    BASE_DIR / "sign videos_new_2_17",
    # BASE_DIR / "sign-videos-categories",
    # BASE_DIR / "sign-video-categories-2",
    BASE_DIR / "sign videos final",
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Regions to collect and display (in order)
REGIONS = ["background", "face_top", "skin_ycrcb", "outfit_space", "outfit_nonskin"]

REGION_LABELS = {
    "background":    "background",
    "face_top":      "face (top 15%)",
    "skin_ycrcb":    "skin (YCrCb det.)",
    "outfit_space":  "outfit (mid 20-60%)",
    "outfit_nonskin":"outfit (non-skin)",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def collect_videos(source_dir: Path) -> list:
    videos = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(Path(root) / f)
    return sorted(videos)


def sample_evenly(lst: list, n: int) -> list:
    if n is None or len(lst) <= n:
        return lst
    step = len(lst) / n
    return [lst[int(i * step)] for i in range(n)]


def get_video_info(filepath: Path):
    """Return (width, height, total_frames). Returns (None,None,None) on failure."""
    cmd = [FFMPEG, "-hide_banner", "-i", str(filepath)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=30, encoding="utf-8", errors="replace")
    text = r.stderr
    m_res = re.search(r"\b(\d{3,5})x(\d{3,5})\b", text)
    if not m_res:
        return None, None, None
    w, h = int(m_res.group(1)), int(m_res.group(2))
    m_fps  = re.search(r"([\d.]+)\s+fps", text)
    fps    = float(m_fps.group(1)) if m_fps else 30.0
    m_dur  = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", text)
    if m_dur:
        secs     = int(m_dur.group(1))*3600 + int(m_dur.group(2))*60 + float(m_dur.group(3))
        n_frames = max(1, int(secs * fps))
    else:
        n_frames = 30
    return w, h, n_frames


def extract_frames(filepath: Path, n_frames: int, total_frames: int):
    """Extract n_frames evenly-spaced PNGs to a RAM-disk temp dir."""
    tmp_dir = Path(r"R:\\") / f"cana_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    step        = max(1, total_frames // n_frames)
    select_expr = f"not(mod(n\\,{step}))"
    out_pat     = str(tmp_dir / "f%04d.png")
    cmd = [
        FFMPEG, "-hide_banner", "-y", "-i", str(filepath),
        "-vf", f"select={select_expr}",
        "-vsync", "vfr", "-frames:v", str(n_frames),
        out_pat,
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)
    return tmp_dir, sorted(tmp_dir.glob("*.png"))


# ── Per-frame analysis ─────────────────────────────────────────────────────────

def _mean_stats(frame_bgr: np.ndarray, mask: np.ndarray):
    """
    Given a BGR frame and a uint8 mask (>0 = selected), return a dict:
        R_mean, R_std, G_mean, G_std, B_mean, B_std
        H_mean, S_mean, V_mean
        count
    Returns None if fewer than 10 selected pixels.
    """
    pixels = frame_bgr[mask > 0]
    if len(pixels) < 10:
        return None

    b_ch = pixels[:, 0].astype(np.float32)
    g_ch = pixels[:, 1].astype(np.float32)
    r_ch = pixels[:, 2].astype(np.float32)

    # HSV per-pixel then average
    roi = frame_bgr.copy()
    roi[mask == 0] = 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0][mask > 0].astype(np.float32)
    s_ch = hsv[:, :, 1][mask > 0].astype(np.float32)
    v_ch = hsv[:, :, 2][mask > 0].astype(np.float32)

    return {
        "R_mean": float(r_ch.mean()), "R_std": float(r_ch.std()),
        "G_mean": float(g_ch.mean()), "G_std": float(g_ch.std()),
        "B_mean": float(b_ch.mean()), "B_std": float(b_ch.std()),
        "H_mean": float(h_ch.mean()),
        "S_mean": float(s_ch.mean()),
        "V_mean": float(v_ch.mean()),
        "count":  int(len(pixels)),
    }


def analyze_frame(img_path: Path, morph_ksize: int):
    """
    Detect person in frame, measure colour statistics for each region.
    Returns dict  region_name → stats_dict  (or None for failed regions).
    Returns None if person detection fails.
    """
    frame = cv2.imread(str(img_path))
    if frame is None:
        return None
    fh, fw = frame.shape[:2]

    # ── Person detection (identical to main script) ──────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    otsu_val, _ = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_bin = cv2.threshold(gray, otsu_val, 255, cv2.THRESH_BINARY_INV)
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (morph_ksize, morph_ksize))
    mask_clean   = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean)
    if n_labels <= 1:
        return None

    best  = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    by    = stats[best, cv2.CC_STAT_TOP]
    bh    = stats[best, cv2.CC_STAT_HEIGHT]

    person_mask = (labels == best).astype(np.uint8) * 255

    # ── Region masks ─────────────────────────────────────────────────────────

    # background: pixels NOT belonging to ANY detected foreground component
    bg_mask = (mask_clean == 0).astype(np.uint8) * 255

    # face_top: top 15 % of person blob height
    face_y2   = by + max(1, int(bh * 0.15))
    face_mask = np.zeros((fh, fw), dtype=np.uint8)
    face_mask[by:face_y2, :] = person_mask[by:face_y2, :]

    # skin_ycrcb: YCrCb skin pixels inside the person blob
    ycrcb         = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_raw      = cv2.inRange(ycrcb, SKIN_LOW, SKIN_HIGH)
    skin_mask     = cv2.bitwise_and(skin_raw, person_mask)

    # outfit_space: middle 20-60 % of person blob height
    out_y1          = by + int(bh * 0.20)
    out_y2          = by + int(bh * 0.60)
    outfit_sp_mask  = np.zeros((fh, fw), dtype=np.uint8)
    outfit_sp_mask[out_y1:out_y2, :] = person_mask[out_y1:out_y2, :]

    # outfit_nonskin: person pixels outside skin detector
    outfit_ns_mask              = person_mask.copy()
    outfit_ns_mask[skin_mask > 0] = 0

    # ── Measure each region ───────────────────────────────────────────────────
    return {
        "background":    _mean_stats(frame, bg_mask),
        "face_top":      _mean_stats(frame, face_mask),
        "skin_ycrcb":    _mean_stats(frame, skin_mask),
        "outfit_space":  _mean_stats(frame, outfit_sp_mask),
        "outfit_nonskin": _mean_stats(frame, outfit_ns_mask),
    }


# ── Aggregation ────────────────────────────────────────────────────────────────

STAT_KEYS = ["R_mean", "R_std", "G_mean", "G_std", "B_mean", "B_std",
             "H_mean", "S_mean", "V_mean", "count"]


def aggregate(records: list):
    """
    records: list of stats-dicts (from _mean_stats), or None entries (skip).
    Returns a single aggregated dict with mean of each stat key, or None if empty.
    """
    valid = [r for r in records if r is not None]
    if not valid:
        return None
    agg = {}
    for k in STAT_KEYS:
        vals = [r[k] for r in valid if k in r]
        agg[k] = float(np.mean(vals)) if vals else 0.0
    return agg


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt_stats(agg, baseline_agg=None, indent=4):
    if agg is None:
        return " " * indent + "(no data)"
    pad = " " * indent
    r, g, b = agg["R_mean"], agg["G_mean"], agg["B_mean"]
    rs, gs, bs = agg["R_std"], agg["G_std"], agg["B_std"]
    h, s, v = agg["H_mean"], agg["S_mean"], agg["V_mean"]
    cnt = int(agg["count"])

    rgb_line = (f"R={r:6.1f}±{rs:4.1f}  G={g:6.1f}±{gs:4.1f}  "
                f"B={b:6.1f}±{bs:4.1f}    "
                f"HSV=({h:5.1f},{s:5.1f},{v:5.1f})  "
                f"n={cnt:,}")

    lines = [pad + rgb_line]

    if baseline_agg is not None:
        br, bg_, bb = baseline_agg["R_mean"], baseline_agg["G_mean"], baseline_agg["B_mean"]
        dr, dg, db  = r - br, g - bg_, b - bb
        def sgn(x): return f"{x:+.1f}"
        diff_line = (f"  diff vs baseline → "
                     f"ΔR={sgn(dr)}  ΔG={sgn(dg)}  ΔB={sgn(db)}")
        lines.append(pad + diff_line)

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("COLOR ANALYSIS")
    print(f"  {VIDEOS_PER_SOURCE} videos × {FRAMES_PER_VIDEO} frames per source")
    print(f"  Baseline: {BASELINE_SOURCE}")
    print("=" * 80)

    # Dict: source_name → region → list of per-frame stats dicts
    all_source_data: dict[str, dict[str, list]] = {}
    # Dict: source_name → video_name → region → list of per-frame stats dicts
    csv_rows = []

    for source in SOURCES:
        name = source.name
        print(f"\n{'─'*80}")
        print(f"SOURCE: {name}")
        print(f"{'─'*80}")

        if not source.exists():
            print(f"  [MISSING] {source}")
            continue

        all_videos = collect_videos(source)
        samples    = sample_evenly(all_videos, VIDEOS_PER_SOURCE)
        print(f"  {len(all_videos)} videos total → sampling {len(samples)}\n")

        # region → flat list of per-frame stats dicts across all sampled videos
        source_region_data: dict[str, list] = {r: [] for r in REGIONS}

        for vid in samples:
            vid_name = vid.name
            print(f"  • {vid_name}")
            w, h, total_frames = get_video_info(vid)
            if w is None:
                print(f"    [SKIP] cannot read video info")
                continue
            print(f"    {w}×{h}, ~{total_frames} frames  (extracting {FRAMES_PER_VIDEO})")

            morph_ksize = max(3, int(h * MORPH_FRAC))
            tmp_dir, pngs = extract_frames(vid, FRAMES_PER_VIDEO, total_frames)
            try:
                for i, png in enumerate(pngs, 1):
                    result = analyze_frame(png, morph_ksize)
                    if result is None:
                        print(f"    frame {i}: detection failed — skipped")
                        continue
                    for region in REGIONS:
                        st = result.get(region)
                        if st is not None:
                            source_region_data[region].append(st)
                            csv_rows.append({
                                "source":  name,
                                "video":   vid_name,
                                "frame":   i,
                                "region":  region,
                                "R_mean":  f"{st['R_mean']:.2f}",
                                "R_std":   f"{st['R_std']:.2f}",
                                "G_mean":  f"{st['G_mean']:.2f}",
                                "G_std":   f"{st['G_std']:.2f}",
                                "B_mean":  f"{st['B_mean']:.2f}",
                                "B_std":   f"{st['B_std']:.2f}",
                                "H_mean":  f"{st['H_mean']:.1f}",
                                "S_mean":  f"{st['S_mean']:.1f}",
                                "V_mean":  f"{st['V_mean']:.1f}",
                                "count":   st["count"],
                            })
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        all_source_data[name] = source_region_data

        # Print aggregated stats for this source
        print()
        for region in REGIONS:
            agg = aggregate(source_region_data[region])
            label = REGION_LABELS[region]
            print(f"  [{label}]")
            print(fmt_stats(agg, indent=4))

    # ── Comparison vs baseline ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"COMPARISON vs BASELINE  ({BASELINE_SOURCE})")
    print(f"{'='*80}")

    baseline_data = all_source_data.get(BASELINE_SOURCE, {})
    baseline_agg  = {r: aggregate(baseline_data.get(r, [])) for r in REGIONS}

    for source in SOURCES:
        name = source.name
        if name not in all_source_data:
            continue
        print(f"\n  SOURCE: {name}")
        if name == BASELINE_SOURCE:
            print(f"    (this IS the baseline)")
            for region in REGIONS:
                agg = aggregate(all_source_data[name].get(region, []))
                print(f"  [{REGION_LABELS[region]}]")
                print(fmt_stats(agg, indent=4))
            continue
        for region in REGIONS:
            agg      = aggregate(all_source_data[name].get(region, []))
            base_agg = baseline_agg.get(region)
            print(f"  [{REGION_LABELS[region]}]")
            print(fmt_stats(agg, base_agg, indent=4))

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if csv_rows:
        fieldnames = ["source", "video", "frame", "region",
                      "R_mean", "R_std", "G_mean", "G_std",
                      "B_mean", "B_std", "H_mean", "S_mean", "V_mean", "count"]
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nRaw per-frame data saved → {CSV_OUT}")
    else:
        print("\n[WARN] No data collected — CSV not written.")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
