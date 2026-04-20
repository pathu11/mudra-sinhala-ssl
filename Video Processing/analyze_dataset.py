"""
Dataset Video Analysis Script
Analyzes all video files across all source folders and reports
resolution, FPS, duration, codec, and file size per source.
Uses ffmpeg -i to extract video metadata.

Outputs:
  - Console log (also saved to dataset_analysis.log)
  - dataset_analysis.csv  (per-file detail)
"""

import os
import re
import sys
import subprocess
import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(r"/path/to/dataset")
FFMPEG   = "ffmpeg.exe"
LOG_FILE = BASE_DIR / "dataset_analysis.log"

SOURCES = [
    BASE_DIR / "sign videos",
    BASE_DIR / "sign videos_new_2_17",
    BASE_DIR / "sign-videos-categories",
    BASE_DIR / "sign-video-categories-2",
    BASE_DIR / "sign videos final",
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# --- Regex patterns for ffmpeg -i stderr ---
RE_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):([\d.]+)", re.IGNORECASE)
RE_BITRATE  = re.compile(r"bitrate:\s*(\d+)\s*kb/s", re.IGNORECASE)
# Matches the video stream line; captures codec name
RE_VIDEO_LINE = re.compile(r"Stream\s+#\S+.*?Video:\s+(\w+)", re.IGNORECASE)
# From the video stream line, extract resolution (WxH) and fps
RE_RESOLUTION = re.compile(r"\b(\d{3,5})x(\d{3,5})\b")
RE_FPS        = re.compile(r"([\d.]+)\s+fps")


# ── Tee: write to both stdout and log file ──────────────────────────────────
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def get_video_info(filepath: Path) -> dict | None:
    """Run ffmpeg -i on filepath and parse stream metadata."""
    cmd = [FFMPEG, "-hide_banner", "-i", str(filepath)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace"
        )
        # ffmpeg prints info to stderr; exit code is always 1 (no output given)
        text = result.stderr
    except Exception as e:
        print(f"  [ERROR] {filepath.name}: {e}")
        return None
    m_dur = RE_DURATION.search(text)
    m_brt = RE_BITRATE.search(text)
    # Find the video stream line
    video_line = None
    for line in text.splitlines():
        if RE_VIDEO_LINE.search(line):
            video_line = line
            break
    if not video_line:
        print(f"  [SKIP]  {filepath.name}: no video stream detected")
        return None
    m_codec = RE_VIDEO_LINE.search(video_line)
    m_res   = RE_RESOLUTION.search(video_line)
    m_fps   = RE_FPS.search(video_line)

    if not m_res or not m_fps:
        print(f"  [SKIP]  {filepath.name}: could not parse resolution/fps")
        return None
    h = int(m_dur.group(1)) if m_dur else 0
    mi = int(m_dur.group(2)) if m_dur else 0
    s = float(m_dur.group(3)) if m_dur else 0.0
    duration = h * 3600 + mi * 60 + s
    codec  = m_codec.group(1) if m_codec else "?"
    width  = int(m_res.group(1))
    height = int(m_res.group(2))
    fps    = float(m_fps.group(1))
    bitrate_kbps = int(m_brt.group(1)) if m_brt else 0
    size_mb = round(filepath.stat().st_size / 1_048_576, 2)

    return {
        "width":        width,
        "height":       height,
        "fps":          round(fps, 3),
        "duration_s":   round(duration, 3),
        "codec":        codec,
        "size_mb":      size_mb,
        "bitrate_kbps": bitrate_kbps,
    }


def collect_videos(source_dir: Path) -> list[Path]:
    videos = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(Path(root) / f)
    return sorted(videos)


def fmt_fps(fps: float) -> str:
    common = {29.97: "29.97", 30.0: "30", 59.94: "59.94", 60.0: "60", 25.0: "25", 24.0: "24"}
    for k, v in common.items():
        if abs(fps - k) < 0.1:
            return v
    return str(fps)


def analyze_source(source_dir: Path) -> list[dict]:
    videos = collect_videos(source_dir)
    if not videos:
        print(f"  No videos found.\n")
        return []

    print(f"  Found {len(videos)} video(s). Probing...")
    rows = []
    for i, v in enumerate(videos, 1):
        info = get_video_info(v)
        if info is None:
            continue
        rel = v.relative_to(BASE_DIR)
        category = v.parent.name if v.parent != source_dir else "(root)"
        rows.append({
            "source": source_dir.name,
            "category": category,
            "filename": v.name,
            "relative_path": str(rel),
            **info,
        })
        if i % 20 == 0:
            print(f"    ... {i}/{len(videos)} done")

    return rows


def print_source_summary(source_name: str, rows: list[dict]):
    if not rows:
        return

    resolutions = set()
    fps_set = set()
    durations = []
    codecs = set()
    sizes = []

    for r in rows:
        resolutions.add(f"{r['width']}x{r['height']}")
        fps_set.add(fmt_fps(r["fps"]))
        durations.append(r["duration_s"])
        codecs.add(r["codec"])
        sizes.append(r["size_mb"])

    print(f"  Files       : {len(rows)}")
    print(f"  Resolutions : {', '.join(sorted(resolutions))}")
    print(f"  FPS values  : {', '.join(sorted(fps_set))}")
    print(f"  Duration    : min={min(durations):.1f}s  max={max(durations):.1f}s  avg={sum(durations)/len(durations):.1f}s")
    print(f"  File size   : min={min(sizes):.1f}MB  max={max(sizes):.1f}MB  avg={sum(sizes)/len(sizes):.1f}MB")
    print(f"  Codecs      : {', '.join(sorted(codecs))}")


def main():
    log_fh = open(LOG_FILE, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_fh)

    all_rows = []

    for source in SOURCES:
        print(f"\n{'='*60}")
        print(f"SOURCE: {source.name}")
        print(f"{'='*60}")
        if not source.exists():
            print(f"  [MISSING] Directory not found: {source}")
            continue

        rows = analyze_source(source)
        all_rows.extend(rows)
        print_source_summary(source.name, rows)

    # Overall summary
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*60}")
    print_source_summary("ALL SOURCES", all_rows)

    # Cross-source resolution comparison
    print(f"\n--- Resolution breakdown by source ---")
    by_source = defaultdict(list)
    for r in all_rows:
        by_source[r["source"]].append(f"{r['width']}x{r['height']}")
    for src, ress in by_source.items():
        unique = sorted(set(ress))
        print(f"  {src[:40]:<40}: {', '.join(unique)}")

    print(f"\n--- FPS breakdown by source ---")
    by_source_fps = defaultdict(list)
    for r in all_rows:
        by_source_fps[r["source"]].append(fmt_fps(r["fps"]))
    for src, fps_list in by_source_fps.items():
        unique = sorted(set(fps_list))
        print(f"  {src[:40]:<40}: {', '.join(unique)}")

    # Save CSV
    if all_rows:
        csv_path = BASE_DIR / "dataset_analysis.csv"
        fieldnames = ["source", "category", "filename", "relative_path",
                      "width", "height", "fps", "duration_s", "codec",
                      "size_mb", "bitrate_kbps"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nDetailed CSV saved to : {csv_path}")

    print(f"Log saved to          : {LOG_FILE}")
    print(f"Done. Total videos analyzed: {len(all_rows)}")

    sys.stdout = sys.__stdout__
    log_fh.close()


if __name__ == "__main__":
    main()
