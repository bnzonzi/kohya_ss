#!/usr/bin/env python3
"""Extract tallest OpenPose render person from a multi-pose PNG → JSON @ target canvas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter

# OpenPose body-18 limb colors (BGR) — ControlNet / lllyasviel annotator convention
BODY_18_COLORS_BGR = np.array(
    [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ],
    dtype=np.uint8,
)

# COCO-18 order in OpenPose JSON (same as ControlNet body)
KEYPOINT_NAMES_18 = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
]

# Skeleton pairs for redraw (1-based OpenPose index in drawing code → 0-based here)
POSE_PAIRS_18 = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]

TARGET_W = 704
TARGET_H = 1288


def skeleton_mask(img: np.ndarray) -> np.ndarray:
    gray = img.max(axis=2)
    return (gray > 45).astype(np.uint8)


def select_tallest_person_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return binary mask for tallest figure (right-hand cluster after optimal vertical cut)."""
    h, w = mask.shape
    best: tuple[int, int, int, int] | None = None  # height, area, cut, label_idx

    for cut in range(max(80, int(w * 0.28)), min(int(w * 0.62), w - 20), 5):
        sub = mask.copy()
        sub[:, :cut] = 0
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sub, connectivity=8)
        for idx in range(1, n_labels):
            x, _y, _ww, hh, area = stats[idx]
            if area < 400:
                continue
            cx = centroids[idx][0]
            if x < cut:
                continue
            if cx < cut + 25:
                continue
            if best is None or hh > best[0] + 2:
                best = (hh, area, cut, idx)
            elif abs(hh - best[0]) <= 2 and area < best[1]:
                # same stature → prefer tighter isolate (right leader, drop neighbors)
                best = (hh, area, cut, idx)

    if best is None:
        raise RuntimeError("aucun personnage détecté sur le mask OpenPose")

    _hh, _area, cut, tall_idx = best
    sub = mask.copy()
    sub[:, :cut] = 0
    n_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(sub, connectivity=8)
    return (labels == tall_idx).astype(np.uint8), cut


def tight_bbox(person: np.ndarray, pad: int = 3) -> tuple[int, int, int, int]:
    ys, xs = np.where(person)
    if len(xs) == 0:
        raise RuntimeError("bbox vide")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    h, w = person.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    return x0, y0, x1, y1


def peak_keypoints_18(crop_bgr: np.ndarray, person_mask: np.ndarray) -> list[tuple[float, float, float]]:
    """Infer COCO-18 joints from local maxima on an OpenPose render."""
    gray = crop_bgr.max(axis=2)
    peaks = (gray == maximum_filter(gray, size=7)) & (gray > 65) & person_mask.astype(bool)
    ys, xs = np.where(peaks)
    if len(xs) < 6:
        raise RuntimeError("trop peu de joints détectés dans le crop")

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    h, w = person_mask.shape
    cx = float(np.mean(xs))

    def pick_nearest(tx: float, ty: float, pool: np.ndarray) -> tuple[float, float, float]:
        d = (pool[:, 0] - tx) ** 2 + (pool[:, 1] - ty) ** 2
        i = int(np.argmin(d))
        return (float(pool[i, 0]), float(pool[i, 1]), 1.0)

    empty = (0.0, 0.0, 0.0)
    kps: list[tuple[float, float, float]] = [empty] * 18

    top_band = pts[pts[:, 1] <= np.percentile(pts[:, 1], 18)]
    mid_band = pts[(pts[:, 1] > np.percentile(pts[:, 1], 15)) & (pts[:, 1] < np.percentile(pts[:, 1], 55))]
    low_band = pts[pts[:, 1] >= np.percentile(pts[:, 1], 55)]

    # head / face
    head_i = int(np.argmin(top_band[:, 1]))
    kps[0] = tuple(top_band[head_i].tolist()) + (1.0,)
    face_pool = top_band[top_band[:, 1] <= top_band[head_i, 1] + h * 0.06]
    if len(face_pool) >= 2:
        face_pool = face_pool[np.argsort(face_pool[:, 0])]
        kps[15] = tuple(face_pool[0].tolist()) + (1.0,)
        kps[14] = tuple(face_pool[-1].tolist()) + (1.0,)
        kps[17] = pick_nearest(kps[15][0], kps[15][1] + 4, face_pool)
        kps[16] = pick_nearest(kps[14][0], kps[14][1] + 4, face_pool)

    neck_y = float(np.percentile(pts[:, 1], 22))
    kps[1] = pick_nearest(cx, neck_y, mid_band if len(mid_band) else pts)

    # shoulders
    shoulder_y = float(np.percentile(pts[:, 1], 28))
    left_mid = mid_band[mid_band[:, 0] < cx] if len(mid_band) else pts[pts[:, 0] < cx]
    right_mid = mid_band[mid_band[:, 0] >= cx] if len(mid_band) else pts[pts[:, 0] >= cx]
    if len(right_mid):
        kps[2] = pick_nearest(float(right_mid[:, 0].mean()), shoulder_y, right_mid)
    if len(left_mid):
        kps[5] = pick_nearest(float(left_mid[:, 0].mean()), shoulder_y, left_mid)

    elbow_y = float(np.percentile(pts[:, 1], 38))
    if len(right_mid):
        kps[3] = pick_nearest(kps[2][0], elbow_y, right_mid)
        wrist_y = float(np.percentile(pts[:, 1], 48))
        kps[4] = pick_nearest(kps[3][0], wrist_y, pts)
    if len(left_mid):
        kps[6] = pick_nearest(kps[5][0], elbow_y, left_mid)
        wrist_y = float(np.percentile(pts[:, 1], 48))
        kps[7] = pick_nearest(kps[6][0], wrist_y, pts)

    hip_y = float(np.percentile(pts[:, 1], 52))
    kps[8] = pick_nearest(cx + w * 0.08, hip_y, low_band if len(low_band) else pts)
    kps[11] = pick_nearest(cx - w * 0.08, hip_y, low_band if len(low_band) else pts)

    knee_y = float(np.percentile(pts[:, 1], 72))
    kps[9] = pick_nearest(kps[8][0], knee_y, low_band)
    kps[12] = pick_nearest(kps[11][0], knee_y, low_band)

    ankle_pts = pts[pts[:, 1] >= np.percentile(pts[:, 1], 88)]
    if len(ankle_pts) >= 2:
        ankle_pts = ankle_pts[np.argsort(ankle_pts[:, 0])]
        kps[10] = tuple(ankle_pts[0].tolist()) + (1.0,)
        kps[13] = tuple(ankle_pts[-1].tolist()) + (1.0,)
    else:
        kps[10] = pick_nearest(kps[9][0], float(np.max(pts[:, 1])), pts)
        kps[13] = pick_nearest(kps[12][0], float(np.max(pts[:, 1])), pts)

    return kps


def transform_keypoints(
    kps: list[tuple[float, float, float]],
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    target_w: int = TARGET_W,
    target_h: int = TARGET_H,
    pad: float = 24.0,
) -> list[tuple[float, float, float]]:
    scale = min((target_w - 2 * pad) / crop_w, (target_h - 2 * pad) / crop_h)
    new_w = crop_w * scale
    new_h = crop_h * scale
    off_x = (target_w - new_w) / 2.0
    off_y = (target_h - new_h) / 2.0
    out: list[tuple[float, float, float]] = []
    for x, y, c in kps:
        if c <= 0:
            out.append((0.0, 0.0, 0.0))
            continue
        nx = (x - x0) * scale + off_x
        ny = (y - y0) * scale + off_y
        out.append((nx, ny, c))
    return out


def render_pose(
    kps: list[tuple[float, float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for a, b in POSE_PAIRS_18:
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        if ca <= 0 or cb <= 0:
            continue
        color = tuple(int(c) for c in BODY_18_COLORS_BGR[a % len(BODY_18_COLORS_BGR)])
        cv2.line(
            canvas,
            (int(round(xa)), int(round(ya))),
            (int(round(xb)), int(round(yb))),
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    for i, (x, y, c) in enumerate(kps):
        if c <= 0:
            continue
        color = tuple(int(v) for v in BODY_18_COLORS_BGR[i % len(BODY_18_COLORS_BGR)])
        cv2.circle(canvas, (int(round(x)), int(round(y))), 4, color, -1, lineType=cv2.LINE_AA)
    return canvas


def build_openpose_json(
    kps: list[tuple[float, float, float]],
    width: int,
    height: int,
    meta: dict,
) -> dict:
    flat: list[float] = []
    for x, y, c in kps:
        flat.extend([round(x, 2), round(y, 2), float(c)])
    return {
        "version": 1.3,
        "canvas_width": width,
        "canvas_height": height,
        "meta": meta,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": flat,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_names": KEYPOINT_NAMES_18,
            }
        ],
    }


def process(
    input_png: Path,
    out_json: Path,
    out_png: Path | None,
    target_w: int = TARGET_W,
    target_h: int = TARGET_H,
) -> dict:
    img = cv2.imread(str(input_png))
    if img is None:
        raise FileNotFoundError(input_png)

    src_h, src_w = img.shape[:2]
    mask = skeleton_mask(img)
    person, cut = select_tallest_person_mask(mask)
    x0, y0, x1, y1 = tight_bbox(person)
    crop_mask = person[y0 : y1 + 1, x0 : x1 + 1]
    crop_bgr = img[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_bgr[crop_mask == 0] = 0

    kps_local = peak_keypoints_18(crop_bgr, crop_mask)
    crop_w = x1 - x0 + 1
    crop_h = y1 - y0 + 1
    kps_canvas = transform_keypoints(
        kps_local, 0, 0, crop_w, crop_h, target_w, target_h
    )

    meta = {
        "source": str(input_png),
        "source_size": [src_w, src_h],
        "extraction": "tallest_person_vertical_cut",
        "vertical_cut_x": cut,
        "crop_bbox_xyxy": [x0, y0, x1, y1],
        "crop_size": [crop_w, crop_h],
        "target_canvas": [target_w, target_h],
    }
    payload = build_openpose_json(kps_canvas, target_w, target_h, meta)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if out_png is not None:
        rendered = render_pose(kps_canvas, target_w, target_h)
        cv2.imwrite(str(out_png), rendered)

    crop_scaled = out_json.with_name(out_json.stem + "_crop_scaled.png")
    pad = 24.0
    scale = min((target_w - 2 * pad) / crop_w, (target_h - 2 * pad) / crop_h)
    new_w = int(round(crop_w * scale))
    new_h = int(round(crop_h * scale))
    resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    off_x = int(round((target_w - new_w) / 2))
    off_y = int(round((target_h - new_h) / 2))
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
    cv2.imwrite(str(crop_scaled), canvas)
    meta["crop_scaled_png"] = str(crop_scaled)

    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_png", type=Path)
    parser.add_argument(
        "--out-json",
        type=Path,
        help="defaults to <stem>_tallest_704x1288.json beside input",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="optional re-rendered pose PNG at target canvas",
    )
    parser.add_argument("--width", type=int, default=TARGET_W)
    parser.add_argument("--height", type=int, default=TARGET_H)
    args = parser.parse_args()

    stem = args.input_png.stem
    out_json = args.out_json or args.input_png.with_name(f"{stem}_tallest_{args.width}x{args.height}.json")
    out_png = args.out_png
    if out_png is False:
        out_png = None
    elif out_png is None:
        out_png = args.input_png.with_name(f"{stem}_tallest_{args.width}x{args.height}.png")

    meta = process(args.input_png, out_json, out_png, args.width, args.height)
    print(json.dumps({"written_json": str(out_json), "written_png": str(out_png) if out_png else None, **meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())