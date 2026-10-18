"""Kisi bhi drone/aerial photo se test video banao (drone upar se udta hua jaisa pan + zoom).

Usage:
    python make_test_video.py photo.jpg test_video.mp4 [seconds] [fps]
Phir app ke "Video analysis" tab me test_video.mp4 upload karo.
"""
import sys

import cv2
import numpy as np


def main(src, dst, seconds=8, fps=15):
    img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit("Image read nahi hui: " + src)
    h, w = img.shape[:2]
    W, H = 1280, 720
    # image ko itna bada karo ke pan ke liye jagah ho
    s = max(W * 1.6 / w, H * 1.6 / h)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    n = int(seconds * fps)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(n):
        t = i / max(n - 1, 1)
        zoom = 1.0 + 0.25 * t                      # halka zoom-in
        cw, ch = int(W * 1.5 / zoom), int(H * 1.5 / zoom)
        x = int((w - cw) * t)                       # left -> right pan
        y = int((h - ch) * (0.5 + 0.4 * np.sin(t * np.pi)))
        crop = img[y:y + ch, x:x + cw]
        out.write(cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA))
    out.release()
    print(f"Saved {dst}: {n} frames, {seconds}s @ {fps} fps")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else 8,
         int(sys.argv[4]) if len(sys.argv) > 4 else 15)
