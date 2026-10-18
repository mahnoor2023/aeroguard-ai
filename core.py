"""AeroGuard AI - detection core (no Streamlit code in here).

Handles: YOLO inference (normal + tiled for small objects + tracking), drawing,
restricted-zone logic, heatmap, statistics and export helpers.
"""
from __future__ import annotations

import base64
import csv
import datetime
import html
import io
import json
import threading
import time
import zipfile
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
MODEL_LOCK = threading.Lock()   # the model is shared between sessions -> one prediction at a time

_RGB = [
    (255, 99, 71), (255, 165, 0), (255, 215, 0), (50, 205, 50), (0, 206, 209),
    (65, 105, 225), (138, 43, 226), (255, 20, 147), (160, 82, 45), (0, 191, 255),
    (154, 205, 50), (220, 20, 60),
]


def color_bgr(i):
    r, g, b = _RGB[int(i) % len(_RGB)]
    return (b, g, r)


def color_hex(i):
    r, g, b = _RGB[int(i) % len(_RGB)]
    return f"#{r:02x}{g:02x}{b:02x}"


# ------------------------------------------------------------------ data model
@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    name: str
    track_id: int | None = None

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def size(self):
        """sqrt(area) in pixels - handy to judge 'small object'."""
        return float(np.sqrt(max(self.x2 - self.x1, 0) * max(self.y2 - self.y1, 0)))

    def as_dict(self):
        return {"class": self.name, "class_id": self.cls, "confidence": round(self.conf, 4),
                "x1": round(self.x1, 1), "y1": round(self.y1, 1),
                "x2": round(self.x2, 1), "y2": round(self.y2, 1),
                "size_px": round(self.size, 1), "track_id": self.track_id}


# -------------------------------------------------------------------- inference
def _starts(n, tile, step):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, step))
    if s[-1] + tile < n:
        s.append(n - tile)
    return s


def _nms(dets, iou_thr):
    """Class-aware NMS in pure numpy (keeps the app free of a torchvision import)."""
    if len(dets) < 2:
        return dets
    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], np.float32)
    scores = np.array([d.conf for d in dets], np.float32)
    cls = np.array([d.cls for d in dets], np.float32)
    boxes = boxes + (cls * (boxes.max() + 1))[:, None]       # separate classes
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return [dets[k] for k in keep]


def _predict(model, img, conf, iou, imgsz, track=False, ox=0, oy=0):
    kw = dict(conf=conf, iou=iou, imgsz=imgsz, max_det=1000, verbose=False)
    with MODEL_LOCK:
        if track:
            res = model.track(img, persist=True, tracker="bytetrack.yaml", **kw)[0]
        else:
            res = model.predict(img, **kw)[0]
    b = res.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy()
    cf = b.conf.cpu().numpy()
    cl = b.cls.cpu().numpy().astype(int)
    ids = b.id.cpu().numpy().astype(int) if getattr(b, "id", None) is not None else [None] * len(cl)
    names = res.names
    return [Detection(float(x1) + ox, float(y1) + oy, float(x2) + ox, float(y2) + oy,
                      float(c), int(k), names[int(k)], None if t is None else int(t))
            for (x1, y1, x2, y2), c, k, t in zip(xyxy, cf, cl, ids)]


def detect(model, bgr, conf=0.25, iou=0.45, imgsz=640, tiled=False, tile=640, overlap=0.2, track=False):
    """Returns (detections, milliseconds). `bgr` is an OpenCV (BGR) array."""
    t0 = time.perf_counter()
    if not tiled:
        dets = _predict(model, bgr, conf, iou, imgsz, track=track)
    else:
        h, w = bgr.shape[:2]
        step = max(32, int(tile * (1 - overlap)))
        dets = _predict(model, bgr, conf, iou, imgsz)           # full pass -> large objects
        for y in _starts(h, tile, step):
            for x in _starts(w, tile, step):
                crop = np.ascontiguousarray(bgr[y:y + tile, x:x + tile])
                dets += _predict(model, crop, conf, iou, tile, ox=x, oy=y)
        dets = _nms(dets, iou)
    return dets, (time.perf_counter() - t0) * 1000.0


def reset_tracker(model):
    with MODEL_LOCK:
        try:
            model.predictor = None
        except Exception:
            pass


# ------------------------------------------------------------------------ zone
def rect_zone(w, h, x_pct, y_pct):
    x0, x1 = x_pct[0] / 100 * w, x_pct[1] / 100 * w
    y0, y1 = y_pct[0] / 100 * h, y_pct[1] / 100 * h
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def parse_polygon(text, w, h):
    """'10,10; 80,15; 70,90' (percent of width/height) -> pixel polygon or None."""
    try:
        pts = []
        for part in text.replace("\n", ";").split(";"):
            if part.strip():
                a, b = part.split(",")
                pts.append((float(a) / 100 * w, float(b) / 100 * h))
        return pts if len(pts) >= 3 else None
    except Exception:
        return None


def in_zone(d, poly):
    cx, cy = d.center
    return cv2.pointPolygonTest(np.array(poly, np.float32).reshape(-1, 1, 2), (float(cx), float(cy)), False) >= 0


# --------------------------------------------------------------------- drawing
def annotate(bgr, dets, show_labels=False, show_conf=False, hud=None):
    img = bgr.copy()
    h, w = img.shape[:2]
    th = max(1, round(min(h, w) / 450))
    fs = max(0.4, min(h, w) / 1400)
    for d in dets:
        col = color_bgr(d.cls)
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(img, p1, p2, col, th, cv2.LINE_AA)
        if show_labels or show_conf:
            label = (d.name if show_labels else "") + (f" {d.conf:.2f}" if show_conf else "")
            if d.track_id is not None:
                label += f" #{d.track_id}"
            label = label.strip()
            (tw, tht), bl = cv2.getTextSize(label, FONT, fs, max(1, th - 1))
            y0 = max(p1[1] - tht - bl - 2, 0)
            cv2.rectangle(img, (p1[0], y0), (p1[0] + tw + 4, y0 + tht + bl + 2), col, -1)
            cv2.putText(img, label, (p1[0] + 2, y0 + tht + 1), FONT, fs, (20, 20, 20), max(1, th - 1), cv2.LINE_AA)
    if hud:
        fs2 = max(0.5, min(h, w) / 900)
        (tw, tht), bl = cv2.getTextSize(hud, FONT, fs2, max(1, th))
        cv2.rectangle(img, (0, 0), (tw + 20, tht + bl + 16), (24, 24, 24), -1)
        cv2.putText(img, hud, (10, tht + 8), FONT, fs2, (255, 255, 255), max(1, th), cv2.LINE_AA)
    return img


def draw_zone(bgr, poly, alert=False):
    h, w = bgr.shape[:2]
    th = max(2, round(min(h, w) / 300))
    col = (60, 60, 240) if alert else (80, 200, 60)
    pts = np.array(poly, np.int32).reshape(-1, 1, 2)
    over = bgr.copy()
    cv2.fillPoly(over, [pts], col)
    out = cv2.addWeighted(over, 0.22, bgr, 0.78, 0)
    cv2.polylines(out, [pts], True, col, th, cv2.LINE_AA)
    return out


class Heat:
    """Accumulates detection centres; renders a density overlay."""

    def __init__(self, shape):
        h, w = shape[:2]
        self.s = min(1.0, 512 / max(h, w))
        self.acc = np.zeros((max(1, int(h * self.s)), max(1, int(w * self.s))), np.float32)

    def add(self, dets):
        ah, aw = self.acc.shape
        for d in dets:
            cx, cy = d.center
            x, y = int(cx * self.s), int(cy * self.s)
            if 0 <= x < aw and 0 <= y < ah:
                self.acc[y, x] += 1

    def render(self, bgr, alpha=0.55):
        if self.acc.max() <= 0:
            return bgr
        blur = cv2.GaussianBlur(self.acc, (0, 0), max(3.0, min(self.acc.shape) / 40))
        m = float(blur.max())
        if m <= 0:
            return bgr
        norm = (blur / m * 255).astype(np.uint8)
        norm = cv2.resize(norm, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(bgr, 1 - alpha, color, alpha, 0)
        return np.where((norm > 12)[..., None], blended, bgr)


# ------------------------------------------------------------------ statistics
SIZE_BINS = [("Tiny (<16 px)", 0, 16), ("Small (16-32 px)", 16, 32),
             ("Medium (32-64 px)", 32, 64), ("Large (>64 px)", 64, 1e9)]


def size_buckets(dets):
    return [(label, sum(1 for d in dets if lo <= d.size < hi)) for label, lo, hi in SIZE_BINS]


def small_ratio(dets, limit=32):
    return (sum(1 for d in dets if d.size < limit) / len(dets)) if dets else 0.0


def class_counts(dets):
    return Counter(d.name for d in dets).most_common()


def conf_histogram(dets, bins=9):
    edges = np.linspace(0.1, 1.0, bins + 1)
    vals = np.histogram([d.conf for d in dets], bins=edges)[0] if dets else np.zeros(bins, int)
    labels = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(bins)]
    return labels, vals.tolist()


# --------------------------------------------------------------------- exports
def png_bytes(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def jpg_bytes(bgr, q=88):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


def csv_bytes(rows):
    if not rows:
        return b""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return out.getvalue().encode("utf-8")


def json_bytes(payload):
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def html_report(title, annotated_bgr, dets, meta):
    h, w = annotated_bgr.shape[:2]
    img = annotated_bgr
    if w > 1600:
        img = cv2.resize(img, (1600, int(h * 1600 / w)), interpolation=cv2.INTER_AREA)
    b64 = base64.b64encode(jpg_bytes(img)).decode()
    summary = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in class_counts(dets))
    meta_rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
                        for k, v in meta.items())
    det_rows = "".join(
        f"<tr><td>{i + 1}</td><td>{html.escape(d.name)}</td><td>{d.conf:.2f}</td>"
        f"<td>{int(d.x1)},{int(d.y1)},{int(d.x2)},{int(d.y2)}</td></tr>" for i, d in enumerate(dets[:500]))
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;background:#f4f6fa;color:#1b2430}}
.sub{{color:#667085;margin-bottom:24px}}.card{{background:#fff;border-radius:10px;padding:18px;margin-bottom:18px;box-shadow:0 1px 4px #0001}}
img{{max-width:100%;border-radius:8px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:6px 10px;border-bottom:1px solid #e5e9f0;text-align:left}}
th{{background:#f0f3f8}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}</style></head><body>
<h1 style="margin-bottom:0">AeroGuard AI - Detection Report</h1>
<div class="sub">{html.escape(title)} &middot; generated {datetime.datetime.now():%Y-%m-%d %H:%M}</div>
<div class="card"><img src="data:image/jpeg;base64,{b64}"></div>
<div class="grid"><div class="card"><h3>Summary ({len(dets)} objects)</h3><table><tr><th>Class</th><th>Count</th></tr>{summary}</table></div>
<div class="card"><h3>Run settings</h3><table>{meta_rows}</table></div></div>
<div class="card"><h3>Detections{' (first 500)' if len(dets) > 500 else ''}</h3>
<table><tr><th>#</th><th>Class</th><th>Conf</th><th>Box x1,y1,x2,y2</th></tr>{det_rows}</table></div></body></html>"""
    return doc.encode("utf-8")


def zip_bytes(named_files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in named_files:
            z.writestr(name, data)
    return buf.getvalue()
