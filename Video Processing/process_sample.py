"""
Dataset Sample Processing Script
---------------------------------
Takes SAMPLES_PER_SOURCE evenly-spread samples from each of the 4 source folders.
For each video:
  1. Extracts DETECT_FRAMES evenly spaced frames via ffmpeg → R:\\ (RAM disk).
  2. For each frame: Otsu threshold → morphological opening → largest connected
     component → per-frame person bounding box.
  3. Unions all per-frame bounding boxes (SIZE), takes MEDIAN of per-frame bbox
     centers (POSITION) — robust: hand extensions inflate the crop area but
     don't shift the person center.
  4. Adds PADDING_PCT, makes the region square, clamps to frame bounds.
  5. Encodes: crop → per-source color correction (see SOURCE_COLOR_FILTER below)
     → scale 960×720 (pad white if needed) → 30fps → h264.

Color correction is PER-SOURCE with manually tunable raw ffmpeg filter strings.
Each source gets its own filter in SOURCE_COLOR_FILTER (None = no correction).

Filter options and when to use them:
  curves=r='x0/y0 x1/y1 ...':g='...':b='...'
      Per-channel tone curve with arbitrary control points (normalized 0–1).
      Ideal for NON-LINEAR corrections where the cast is different across tones.
      Example: sign videos blue cast hits skin midtones (ΔB=+35) far harder
      than the near-white background (ΔB=+5). One point fixes skin B, another
      fixes background B — cannot be done with a single linear multiplier.

  colorchannelmixer=rr=...:gg=...:bb=...
      Linear per-channel scale: out_R = in_R * rr, out_G = in_G * gg, etc.
      Works on the FULL tonal range including near-white highlights.
      Good for: uniformly underexposed sources (4K sign-video* folders) where
      ALL tones need the same proportional boost.

  eq=gamma_r=...:gamma_g=...:gamma_b=...
      Applies  out = (in/255)^(1/gamma) * 255  per channel.
      ⚠ Gamma maps 255→255 always — no effect on near-white backgrounds.
      Prefer colorchannelmixer for uniform exposure corrections.

HOW TO TUNE:
  Edit SOURCE_COLOR_FILTER values, run with SAMPLES_PER_SOURCE=2 to check visuals,
  adjust until all sources look consistent, then set SAMPLES_PER_SOURCE to None
  and OUT_DIR to unified_dataset for full-dataset processing.

Color correction values were derived from color_analysis.py (5 measured regions
per source: background, face_top, skin_ycrcb, outfit_space, outfit_nonskin).
Baseline = sign videos_new_2_17 (bg: R=247.7 G=249.1 B=248.1).

Uses python39 / cv2 4.8 for image analysis; ffmpeg for frame extraction
and final encoding. No OpenCV video capture — avoids HEVC codec issues.

Output folder : ./unified_output/
Filename scheme: <sanitized_relative_path>.mp4
"""

import os
import re
import sys
import uuid
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"/path/to/dataset")
FFMPEG     = "ffmpeg.exe"
OUT_DIR    = BASE_DIR / "unified_output"
LOG_FILE   = BASE_DIR / "process_all.log"

SAMPLES_PER_SOURCE  = None  # None = process all videos; set to int for sampling
DETECT_FRAMES       = 10   # frames extracted per video for person detection
TARGET_W            = 960  # output width  (pixels)  — 4:3 standard
TARGET_H            = 720  # output height (pixels)
TARGET_FPS          = 30
PADDING_PCT         = 0.20 # fractional padding added around detected person bbox
MORPH_FRAC          = 0.005 # morph kernel size as fraction of frame height

SOURCES = [
    BASE_DIR / "sign videos",
    BASE_DIR / "sign videos_new_2_17",
    BASE_DIR / "sign-videos-categories",
    BASE_DIR / "sign-video-categories-2",
    BASE_DIR / "sign videos final",
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# ── Per-source color correction (raw ffmpeg filter string, or None) ──────────
# Measured per color_analysis.py (background / face_top regions):
#
#  Source                   bg_R   bg_G   bg_B   face_R  face_G  face_B
#  sign videos_new_2_17     247.7  249.1  248.1  129.9   94.8    78.6   ← BASELINE
#  sign videos              240.4  247.4  252.9  123.3   96.4   113.8
#    ΔBG                     -7.3   -1.7   +4.8   -6.5   +1.6   +35.2
#  sign-videos-categories   215.5  208.9  200.3   96.2   71.8    59.1
#    ΔBG                    -32.2  -40.2  -47.8  -33.6  -23.0   -19.5
#  sign-video-categories-2  204.0  196.0  186.8   86.1   59.9    48.3
#    ΔBG                    -43.7  -53.1  -61.3  -43.7  -34.9   -30.3
#  sign videos final        243.3  244.4  244.2  115.3   85.0    71.7
#    ΔBG                     -4.4   -4.7   -3.9  -14.5   -9.9    -6.9
#
# sign videos — non-linear blue cast: skin ΔB=+35 vs bg ΔB=+5.
#   colorchannelmixer alone cannot fix both tones simultaneously.
#   → curves filter with two control points per channel (skin + background).
#   R control pts: face (0.484→0.510), bg (0.943→0.971)
#   B control pts: face (0.446→0.308), bg (0.992→0.973)   ← large B reduction in midtones
#
# 4K sources — uniform underexposure across all channels.
#   colorchannelmixer linear scale maps measured bg value to baseline bg value.
#   Scale = target_bg / measured_bg (e.g. 247.7/215.5 = 1.149 for R channel).
#   Near-white bg pixels at or above the scale clip point round to 255 (desired).


SOURCE_COLOR_FILTER = {
    # curves: per-channel tone correction derived from measured data.
    # Normalized control pts: x=input/255, y=output/255.

    "sign videos": (
        "curves="
        "r='0/0 0.484/0.510 0.943/0.971':"
        "b='0/0 0.446/0.308 0.992/0.973',"
        "eq=brightness=-0.10"
    ),

    "sign videos_new_2_17": None,  # BASELINE — no correction

    # colorchannelmixer: linear scale to bring measured bg to baseline bg.
    # Factors: rr=247.7/215.5, gg=249.1/208.9, bb=248.1/200.3

    "sign-videos-categories":  "colorchannelmixer=rr=1.149:gg=1.192:bb=1.239",

    # Factors: rr=247.7/204.0, gg=249.1/196.0, bb=248.1/186.8

    "sign-video-categories-2": "colorchannelmixer=rr=1.214:gg=1.271:bb=1.329",

    # curves: non-linear correction — background barely darker (-4) but skin
    # significantly underexposed (-10 to -14 across channels). Two control
    # points per channel: face midtones + background near-white.
    # R pts: face (0.452→0.509), bg (0.954→0.971)
    # G pts: face (0.333→0.372), bg (0.958→0.977)
    # B pts: face (0.281→0.308), bg (0.958→0.973)

    "sign videos final": (
        "curves="
        "r='0/0 0.452/0.509 0.954/0.971':"
        "g='0/0 0.333/0.372 0.958/0.977':"
        "b='0/0 0.281/0.308 0.958/0.973'"
    ),
}


# ── Tee stdout → console + log ───────────────────────────────────────────────
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_videos(source_dir: Path) -> list:
    """Recursively collect all video files under source_dir, sorted."""
    videos = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(Path(root) / f)
    return sorted(videos)


def sample_evenly(lst: list, n) -> list:
    """Return n evenly spaced items from lst. If n is None, return all."""
    if n is None or len(lst) <= n:
        return lst
    step = len(lst) / n
    return [lst[int(i * step)] for i in range(n)]


def sanitize_path_to_name(filepath: Path) -> str:
    """
    Convert relative path (from BASE_DIR) to a flat output filename.
    Directory separators → __, spaces → _, extension replaced with .mp4
    """
    rel = filepath.relative_to(BASE_DIR)
    parts = [p.replace(" ", "_") for p in rel.parts]
    stem = "__".join(parts)
    stem = Path(stem).stem   # strip the last extension
    return stem + ".mp4"


def get_video_info(filepath: Path) -> tuple:
    """
    Return (width, height, total_frames) by parsing ffmpeg -i stderr.
    total_frames is estimated from duration × fps.
    Returns (None, None, None) on failure.
    """
    cmd = [FFMPEG, "-hide_banner", "-i", str(filepath)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=30, encoding="utf-8", errors="replace")
    text = r.stderr

    m_res = re.search(r"\b(\d{3,5})x(\d{3,5})\b", text)
    if not m_res:
        return None, None, None
    w, h = int(m_res.group(1)), int(m_res.group(2))

    m_fps = re.search(r"([\d.]+)\s+fps", text)
    fps = float(m_fps.group(1)) if m_fps else 30.0

    m_dur = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", text)
    if m_dur:
        total_secs = (int(m_dur.group(1)) * 3600 +
                      int(m_dur.group(2)) * 60 +
                      float(m_dur.group(3)))
        n_frames = max(1, int(total_secs * fps))
    else:
        n_frames = 30  # fallback

    return w, h, n_frames


def extract_frames_to_tmpdir(filepath: Path, n_frames: int,
                              total_frames: int) -> tuple:
    """
    Extract n_frames evenly spaced PNGs from filepath into a unique temp dir
    on R:\\ (RAM disk). Returns (tmp_dir_path, list_of_png_paths).
    Caller must delete tmp_dir_path via shutil.rmtree after use.
    """
    tmp_dir = Path(r"R:\\") / f"ssl_tmp_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    step = max(1, total_frames // n_frames)
    select_expr = f"not(mod(n\\,{step}))"
    out_pattern = str(tmp_dir / "frame%04d.png")

    cmd = [
        FFMPEG, "-hide_banner", "-y",
        "-i", str(filepath),
        "-vf", f"select={select_expr}",
        "-vsync", "vfr",
        "-frames:v", str(n_frames),
        out_pattern,
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)
    return tmp_dir, sorted(tmp_dir.glob("*.png"))


def person_bbox_from_frame(img_path: Path, morph_ksize: int):
    """
    Detect the person bounding box in a single frame.
      1. Otsu threshold (THRESH_BINARY_INV): person pixels → white mask.
      2. Morphological opening: removes small noise blobs (e.g. corner artifacts).
      3. Largest connected component = person body.
    Returns (x, y, w, h, stable_cx) or None if nothing detected.
      stable_cx: x-centroid (centre of mass) of the person component mask.
                 Computed via cv2.moments — weights all person pixels equally.
                 Naturally robust to arm extensions (a thin arm contributes far
                 fewer pixels than the torso) without any body-part assumptions.
                 Median across multiple frames further smooths transient outliers.
    """
    frame = cv2.imread(str(img_path))
    if frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Corner-sampling adaptive threshold:
    # Sample background brightness from 4 corners → set threshold at 88% of bg.
    # More reliable than Otsu for sources where the bg isn't uniformly bright
    # (e.g. sign-videos-categories has bg_gray≈209, confusing Otsu's bimodal
    # assumption and causing a false dark blob to be picked up instead of the person).
    img_h, img_w = gray.shape
    cs = max(50, min(100, img_h // 20, img_w // 40))
    corners = np.concatenate([
        gray[:cs, :cs].flatten(),
        gray[:cs, img_w - cs:].flatten(),
        gray[img_h - cs:, :cs].flatten(),
        gray[img_h - cs:, img_w - cs:].flatten(),
    ])
    bg_gray = float(corners.mean())
    threshold = int(bg_gray * 0.88)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (morph_ksize, morph_ksize))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean)
    if n_labels <= 1:
        return None

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(component_areas)) + 1
    x  = stats[best, cv2.CC_STAT_LEFT]
    y  = stats[best, cv2.CC_STAT_TOP]
    bw = stats[best, cv2.CC_STAT_WIDTH]
    bh = stats[best, cv2.CC_STAT_HEIGHT]

    # x-centroid of the person component (centre of mass, pixel-area weighted).
    # Thin arms extending sideways contribute far fewer pixels than the torso,
    # so the centroid naturally stays near the body centre without any explicit
    # body-part assumptions.
    person_mask = (labels == best).astype(np.uint8) * 255
    m = cv2.moments(person_mask)
    if m["m00"] > 0:
        stable_cx = int(m["m10"] / m["m00"])
    else:
        stable_cx = x + bw // 2  # fallback: full-blob bbox centre

    return (x, y, bw, bh, stable_cx)


def detect_crop_box(filepath: Path, vid_w: int, vid_h: int,
                    total_frames: int) -> tuple:
    """
    Extract DETECT_FRAMES frames, detect person in each, compute crop box.

    Crop SIZE  = padded union of all per-frame bboxes (captures full motion range).
    Crop CENTER X = median of per-frame x-centroid values.
        x-centroid per frame = pixel-area-weighted x mean of the person component.
        The torso (large pixel area) dominates; thin extended arms barely shift it.
        Median across frames further suppresses transient outliers.
    Crop CENTER Y = median of per-frame bbox vertical centres.

    Returns (crop_x, crop_y, side, cx).
    Falls back to largest centred square on failure.
    """
    morph_ksize = max(3, int(vid_h * MORPH_FRAC))

    tmp_dir, pngs = extract_frames_to_tmpdir(filepath, DETECT_FRAMES, total_frames)
    try:
        if not pngs:
            print(f"    [WARN] Frame extraction failed — using centred fallback crop")
            return _fallback_square(vid_w, vid_h)

        x1_list, y1_list, x2_list, y2_list = [], [], [], []
        stable_cx_list, cy_list = [], []

        for p in pngs:
            box = person_bbox_from_frame(p, morph_ksize)
            if box is None:
                continue
            bx, by, bw, bh, stable_cx = box
            x1_list.append(bx);              y1_list.append(by)
            x2_list.append(bx + bw);         y2_list.append(by + bh)
            stable_cx_list.append(stable_cx)
            cy_list.append(by + bh // 2)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not x1_list:
        print(f"    [WARN] No person detected — using centred fallback crop")
        return _fallback_square(vid_w, vid_h)

    # Union bbox → crop SIZE
    ux1, uy1 = min(x1_list), min(y1_list)
    ux2, uy2 = max(x2_list), max(y2_list)
    person_w = ux2 - ux1
    person_h = uy2 - uy1

    pad_x = int(person_w * PADDING_PCT)
    pad_y = int(person_h * PADDING_PCT)
    px1 = max(0, ux1 - pad_x);   py1 = max(0, uy1 - pad_y)
    px2 = min(vid_w, ux2 + pad_x); py2 = min(vid_h, uy2 + pad_y)

    side = max(px2 - px1, py2 - py1)

    # Crop POSITION:
    #   X: median of per-frame x-centroids (pixel-area-weighted body centre).
    #   Y: median of per-frame bbox vertical centres.
    cx = int(np.median(stable_cx_list))
    cy = int(np.median(cy_list))
    crop_x = max(0, cx - side // 2)
    crop_y = max(0, cy - side // 2)

    if crop_x + side > vid_w:
        crop_x = max(0, vid_w - side)
        side   = min(side, vid_w)
    if crop_y + side > vid_h:
        crop_y = max(0, vid_h - side)
        side   = min(side, vid_h)

    side   = side   - (side   % 2)
    crop_x = crop_x - (crop_x % 2)
    crop_y = crop_y - (crop_y % 2)

    return crop_x, crop_y, side, cx


def _fallback_square(vid_w: int, vid_h: int) -> tuple:
    """Centred square crop using the shorter dimension."""
    side   = min(vid_w, vid_h) - (min(vid_w, vid_h) % 2)
    crop_x = (vid_w - side) // 2
    crop_y = (vid_h - side) // 2
    cx     = vid_w // 2
    return crop_x, crop_y, side, cx


def process_video(src: Path, out_path: Path, source_name: str) -> bool:
    """
    Full pipeline for one video:
      get video info → detect crop box → apply per-source color correction
      → encode h264 720×720 30fps.
    Returns True on success.
    """
    vid_w, vid_h, total_frames = get_video_info(src)
    if vid_w is None:
        print(f"    [ERROR] Could not read video info.")
        return False
    print(f"    Source: {vid_w}x{vid_h}, ~{total_frames} frames")

    crop_x, crop_y, side, cx = detect_crop_box(src, vid_w, vid_h, total_frames)

    # Build 4:3 horizontal crop centred on person centroid (cx).
    # crop_h = side (from person height detection).
    # crop_w = side * TARGET_W/TARGET_H  →  rectangle that scales to
    # TARGET_W×TARGET_H exactly, filling with real video background instead of white.
    # e.g. for 4K source: side=2160, crop_w=2880 (fits in 3840-wide frame).
    crop_w = side * TARGET_W // TARGET_H   # maintain 4:3 ratio
    crop_w += crop_w % 2                   # keep even
    crop_x_wide = max(0, cx - crop_w // 2)
    if crop_x_wide + crop_w > vid_w:
        crop_x_wide = max(0, vid_w - crop_w)
        crop_w = min(crop_w, vid_w)
    crop_x_wide -= crop_x_wide % 2        # keep even

    print(f"    Crop: x={crop_x_wide} y={crop_y} w={crop_w} h={side}  "
          f"(person cx={cx}, {side/vid_h*100:.1f}% of height)")

    if side <= 0:
        print(f"    [ERROR] Invalid crop dimensions, skipping.")
        return False

    # Look up per-source color correction filter
    color_filter = SOURCE_COLOR_FILTER.get(source_name)
    print(f"    Color filter: {color_filter or '(none)'}")

    color_seg = f"{color_filter}," if color_filter else ""

    # Filter chain:
    # 1. 4:3 rectangle crop centred on person (fills with real video background)
    # 2. Per-source color correction (removes tint, adjusts brightness)
    # 3. Scale to TARGET_W × TARGET_H (exact fit; pad only if crop_w was clamped)
    # 4. Pad safety for edge-clamped cases
    # 5. Normalise FPS to 30
    vf = (
        f"crop={crop_w}:{side}:{crop_x_wide}:{crop_y},"
        f"{color_seg}"
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:color=white,"
        f"fps={TARGET_FPS}"
    )

    cmd = [
        FFMPEG, "-hide_banner", "-y",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "17",        # visually lossless
        "-preset", "medium",
        "-an",
        str(out_path),
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=900, encoding="utf-8", errors="replace")

        if r.returncode != 0 or not out_path.exists():
            print(f"    [ERROR] ffmpeg encode failed:\n{r.stderr[-800:]}")
            return False
    except subprocess.TimeoutExpired:
        print("    [ERROR] ffmpeg encode timed out after 900 seconds.")
        # If it timed out, it might have left a partial file
        if out_path.exists():
            try:
                out_path.unlink()
            except:
                pass
        return False
    except Exception as e:
        print(f"    [ERROR] ffmpeg encode failed with exception: {e}")
        return False

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"    Output: {out_path.name}  ({size_mb:.2f} MB)")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_FILE, "a", encoding="utf-8")  # Changed to append mode for resuming
    sys.stdout = Tee(sys.__stdout__, log_fh)

    total_ok    = 0
    total_fail  = 0

    # ── Pre-collect all tasks so tqdm knows the total count upfront ──────────
    all_tasks = []   # list of (source_path, all_videos, samples)
    for source in SOURCES:
        if not source.exists():
            print(f"\n[MISSING] {source}")
            continue
        all_videos = collect_videos(source)
        samples    = sample_evenly(all_videos, SAMPLES_PER_SOURCE)
        all_tasks.append((source, all_videos, samples))

    total_videos = sum(len(s) for _, _, s in all_tasks)

    # tqdm writes directly to the real stdout so the bar is not written to the log file.
    pbar = tqdm(
        total       = total_videos,
        unit        = "vid",
        desc        = "Overall",
        dynamic_ncols = True,
        file        = sys.__stdout__,
        smoothing   = 0.1,   # ETA averaged over last ~10 videos
    )

    for source, all_videos, samples in all_tasks:
        print(f"\n{'='*60}")
        print(f"SOURCE: {source.name}")
        print(f"{'='*60}")
        print(f"  Total videos in source : {len(all_videos)}")
        n_label = SAMPLES_PER_SOURCE if SAMPLES_PER_SOURCE is not None else len(all_videos)
        print(f"  Processing {len(samples)} of {n_label} requested …\n")

        for i, src in enumerate(samples, 1):
            out_name = sanitize_path_to_name(src)
            out_path = OUT_DIR / out_name
            rel      = str(src.relative_to(BASE_DIR))

            pbar.set_description(f"{source.name[:20]} | {src.name[:25]}")
            print(f"  [{i}/{len(samples)}] {rel}")
            
            # --- Checkpoint / Resume check ---
            if out_path.exists():
                file_size = out_path.stat().st_size
                if file_size > 0:
                    print(f"    [SKIPPED] Output exists ({file_size / 1024 / 1024:.2f} MB)")
                    total_ok += 1
                    print()
                    pbar.update(1)
                    continue
                else:
                    print(f"    [CLEANUP] Removing zero-byte output file")
                    try:
                        out_path.unlink()
                    except Exception as e:
                        print(f"    [WARN] Could not remove {out_path.name}: {e}")
            
            ok = process_video(src, out_path, source.name)

            if ok:
                total_ok += 1
            else:
                total_fail += 1

            print()
            pbar.update(1)

    pbar.close()

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Succeeded : {total_ok}")
    print(f"  Failed    : {total_fail}")
    print(f"  Output dir: {OUT_DIR}")
    print(f"  Log       : {LOG_FILE}")

    sys.stdout = sys.__stdout__
    log_fh.close()


if __name__ == "__main__":
    main()
