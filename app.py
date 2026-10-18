"""AeroGuard AI - advanced aerial object detection dashboard (Streamlit).

Run:  streamlit run app.py
Needs core.py and best.pt (or best_final.pt) next to this file.
"""
import io
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import core

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="AeroGuard AI", page_icon="🛰️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem;}
    .hero {padding: 18px 24px; border-radius: 14px; margin-bottom: 14px;
           background: linear-gradient(120deg, #0f2a4a 0%, #1668dc 100%); color: #fff;}
    .hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
    .hero p {margin: 4px 0 0 0; opacity: .85;}
    div[data-testid="stMetric"] {background: rgba(128,128,128,.10); border-radius: 10px;
                                 padding: 10px 14px;}
    </style>
    <div class="hero"><h1>🛰️ AeroGuard AI</h1>
    <p>Aerial object detection dashboard - YOLOv8n trained on VisDrone (mAP50 = 0.307)</p></div>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).parent
MODEL_NAMES = ["best_final.pt", "best.pt"]  # first one found is used
IMG_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
MAX_SIDE = 3000  # very large uploads are down-scaled to keep the free cloud tier alive

# st.* "full width" argument changed name in newer Streamlit versions
_V = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}

CLASS_MAP50 = {"car": 0.725, "bus": 0.434, "van": 0.355, "motor": 0.333, "pedestrian": 0.315,
               "truck": 0.275, "people": 0.249, "tricycle": 0.208, "awning-tricycle": 0.109,
               "bicycle": 0.064}


# ------------------------------------------------------------------ model
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    for name in MODEL_NAMES:
        path = BASE_DIR / name
        if path.exists():
            return YOLO(str(path))
    return None


model = load_model()
if model is None:
    st.error("Model file not found. Upload best.pt (or best_final.pt) next to app.py.")
    st.stop()

NAMES = {int(k): v for k, v in model.names.items()}
NAME2ID = {v: k for k, v in NAMES.items()}


# ------------------------------------------------------------------ helpers
def decode(data: bytes) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.ascontiguousarray(np.array(img)[:, :, ::-1])  # RGB -> BGR


@st.cache_data(show_spinner=False, max_entries=48)
def detect_cached(_model, data, conf, iou, imgsz, tiled, tile, overlap):
    bgr = decode(data)
    dets, ms = core.detect(_model, bgr, conf, iou, imgsz, tiled, tile, overlap)
    return dets, ms, bgr.shape[:2]


def zone_poly(shape, P):
    if not P["zone_on"]:
        return None
    h, w = shape[:2]
    if P["zone_mode"] == "Rectangle":
        return core.rect_zone(w, h, P["zx"], P["zy"])
    return core.parse_polygon(P["zpoly"], w, h)


def visible(dets, P):
    return [d for d in dets if d.cls in P["cls_ids"]]


def zone_stats(vis, poly, P):
    if not poly:
        return 0, False
    n = sum(1 for d in vis if core.in_zone(d, poly))
    return n, (P["alert"] > 0 and n >= P["alert"])


def compose(bgr, vis, poly, P, mode):
    """Returns an RGB image for the requested view mode."""
    n_in, alert = zone_stats(vis, poly, P)
    if mode == "Original":
        return bgr[:, :, ::-1]
    base = bgr
    if mode == "Heatmap":
        heat = core.Heat(bgr.shape)
        heat.add(vis)
        base = heat.render(bgr)
        if poly:
            base = core.draw_zone(base, poly, alert)
        return base[:, :, ::-1]
    if poly:
        base = core.draw_zone(base, poly, alert)
    hud = f"AeroGuard AI | Objects: {len(vis)}" + (f" | Zone: {n_in}" if poly else "")
    return core.annotate(base, vis, P["labels"], P["show_conf"], hud)[:, :, ::-1]


def det_rows(vis, poly, image_name=None):
    rows = []
    for d in vis:
        r = {"image": image_name} if image_name else {}
        r.update(d.as_dict())
        if poly:
            r["in_zone"] = core.in_zone(d, poly)
        rows.append(r)
    return rows


def class_chart(counts):
    df = pd.DataFrame(counts, columns=["Class", "Count"])
    try:
        import altair as alt
        chart = (alt.Chart(df).mark_bar().encode(
            x=alt.X("Count:Q", title=None),
            y=alt.Y("Class:N", sort="-x", title=None),
            color=alt.Color("Class:N", legend=None, scale=alt.Scale(
                domain=[c for c, _ in counts],
                range=[core.color_hex(NAME2ID.get(c, 0)) for c, _ in counts])),
            tooltip=["Class", "Count"]).properties(height=max(120, 28 * len(df))))
        st.altair_chart(chart, **STRETCH)
    except Exception:
        st.bar_chart(df.set_index("Class"), **STRETCH)


def insights(vis, P, ms):
    msgs = []
    if not vis:
        msgs.append("No objects found. Try a lower confidence threshold, a larger image size, "
                    "or switch on **Small-object mode** in the sidebar.")
    else:
        sr = core.small_ratio(vis)
        if sr > 0.5 and not P["tiled"]:
            msgs.append(f"**{sr:.0%}** of detections are smaller than 32 px. Enable **Small-object mode** "
                        "(tiled inference) to recover more of the tiny objects.")
        if P["conf"] >= 0.5:
            msgs.append("Confidence threshold is high - low-confidence objects are hidden.")
    if P["tiled"] and ms > 15000:
        msgs.append("Tiled inference is slow on large images - use a larger tile size to speed it up.")
    for m in msgs:
        st.info(m, icon="💡")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    with st.expander("Detection", expanded=True):
        conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
        iou = st.slider("IoU (NMS) threshold", 0.10, 0.90, 0.45, 0.05)
        imgsz = st.selectbox("Image size", [640, 960, 1280, 1536], index=0)
    with st.expander("Small-object mode"):
        tiled = st.checkbox("Tiled inference", value=False,
                            help="Cuts the image into overlapping tiles so tiny objects (bicycle, "
                                 "pedestrian) are not lost when the image is resized. Slower.")
        tile = st.selectbox("Tile size", [416, 512, 640, 800], index=2, disabled=not tiled)
        overlap = st.slider("Tile overlap", 0.10, 0.50, 0.20, 0.05, disabled=not tiled)
    with st.expander("Display"):
        show_labels = st.checkbox("Show class labels", value=False)
        show_conf = st.checkbox("Show confidence", value=False)
        chosen = st.multiselect("Classes to show", list(NAMES.values()), default=list(NAMES.values()))
    with st.expander("Restricted zone / counting area"):
        zone_on = st.checkbox("Enable zone", value=False)
        zone_mode = st.radio("Zone type", ["Rectangle", "Polygon"], horizontal=True, disabled=not zone_on)
        zx = st.slider("Horizontal range (% of width)", 0, 100, (25, 75),
                       disabled=not zone_on or zone_mode != "Rectangle")
        zy = st.slider("Vertical range (% of height)", 0, 100, (25, 75),
                       disabled=not zone_on or zone_mode != "Rectangle")
        zpoly = st.text_area("Polygon points (x,y in %, separated by ;)", "10,10; 90,10; 80,90; 20,90",
                             disabled=not zone_on or zone_mode != "Polygon")
        alert = st.number_input("Alert when objects in zone >=", 0, 9999, 0,
                                help="0 = alert off", disabled=not zone_on)

P = dict(conf=conf, iou=iou, imgsz=imgsz, tiled=tiled, tile=tile, overlap=overlap,
         labels=show_labels, show_conf=show_conf, cls_ids={NAME2ID[n] for n in chosen},
         zone_on=zone_on, zone_mode=zone_mode, zx=zx, zy=zy, zpoly=zpoly, alert=int(alert))
DET_ARGS = (P["conf"], P["iou"], P["imgsz"], P["tiled"], P["tile"], P["overlap"])

tab_img, tab_vid, tab_model = st.tabs(["🖼️ Image analysis", "🎞️ Video analysis", "📊 Model info"])


# ================================================================== IMAGE TAB
with tab_img:
    files = st.file_uploader("Upload one or more drone images", type=IMG_TYPES, accept_multiple_files=True)
    if not files:
        st.info("Upload a drone image to start. You can upload several images for batch analysis.")
    else:
        idx = 0
        if len(files) > 1:
            idx = st.selectbox("Image to inspect", range(len(files)), format_func=lambda i: files[i].name)
        f = files[idx]
        data = f.getvalue()

        with st.spinner("Running detection..." + (" (tiled mode - this can take a while)" if tiled else "")):
            dets, ms, shape = detect_cached(model, data, *DET_ARGS)
        bgr = decode(data)
        vis = visible(dets, P)
        poly = zone_poly(shape, P)
        n_in, is_alert = zone_stats(vis, poly, P)
        if P["zone_on"] and poly is None:
            st.warning("Polygon zone needs at least 3 valid points, e.g. `10,10; 90,10; 50,90`.")

        # ---- metrics
        cols = st.columns(5 if poly else 4)
        cols[0].metric("Total objects", len(vis))
        cols[1].metric("Classes found", len({d.name for d in vis}))
        cols[2].metric("Avg confidence", f"{np.mean([d.conf for d in vis]):.2f}" if vis else "-")
        cols[3].metric("Inference time", f"{ms / 1000:.2f} s")
        if poly:
            cols[4].metric("Objects in zone", n_in)
        if is_alert:
            st.error(f"⚠️ ALERT: {n_in} objects inside the monitored zone (threshold {P['alert']}).")

        view = st.radio("View", ["Annotated", "Original", "Side by side", "Heatmap"], horizontal=True)
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Detection result")
            if view == "Side by side":
                a, b = st.columns(2)
                a.image(compose(bgr, vis, poly, P, "Original"), caption="Original", **STRETCH)
                b.image(compose(bgr, vis, poly, P, "Annotated"), caption="Annotated", **STRETCH)
            else:
                st.image(compose(bgr, vis, poly, P, view), **STRETCH)
            annotated_bgr = np.ascontiguousarray(compose(bgr, vis, poly, P, "Annotated")[:, :, ::-1])
        with right:
            st.subheader("Summary")
            counts = core.class_counts(vis)
            if counts:
                df = pd.DataFrame(counts, columns=["Class", "Count"])
                st.dataframe(df, hide_index=True, **STRETCH)
                class_chart(counts)
            else:
                st.info("No objects found. Try lowering the confidence threshold.")
        insights(vis, P, ms)

        # ---- detail panels
        with st.expander("📋 Detections table"):
            rows = det_rows(vis, poly)
            if rows:
                min_c = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05, key="tbl_conf")
                st.dataframe(pd.DataFrame([r for r in rows if r["confidence"] >= min_c]),
                             hide_index=True, **STRETCH)
            else:
                st.write("Nothing to list.")

        with st.expander("📐 Object size & confidence analysis"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Object size** (square-root of box area)")
                sb = pd.DataFrame(core.size_buckets(vis), columns=["Size", "Objects"]).set_index("Size")
                st.bar_chart(sb, **STRETCH)
                st.caption(f"Small objects (<32 px): {core.small_ratio(vis):.0%}")
            with c2:
                st.markdown("**Confidence distribution**")
                labels, vals = core.conf_histogram(vis)
                st.bar_chart(pd.DataFrame({"Detections": vals}, index=labels), **STRETCH)

        with st.expander("⚖️ Compare standard vs tiled inference"):
            st.caption("Runs the model both ways with your current thresholds and shows how many extra objects "
                       "tiling finds. Tiled mode can take 10-60 s on CPU.")
            key = f"cmp_{f.name}_{len(data)}"
            if st.button("Run comparison", key="btn_cmp"):
                st.session_state[key] = True
            if st.session_state.get(key):
                with st.spinner("Comparing..."):
                    d1, t1, _ = detect_cached(model, data, P["conf"], P["iou"], P["imgsz"], False, P["tile"], P["overlap"])
                    d2, t2, _ = detect_cached(model, data, P["conf"], P["iou"], P["imgsz"], True, P["tile"], P["overlap"])
                v1, v2 = visible(d1, P), visible(d2, P)
                m1, m2, m3 = st.columns(3)
                m1.metric("Standard", len(v1), help=f"{t1 / 1000:.1f} s")
                m2.metric("Tiled", len(v2), delta=len(v2) - len(v1), help=f"{t2 / 1000:.1f} s")
                m3.metric("Small objects (<32 px)", f"{core.small_ratio(v2):.0%}",
                          delta=f"{(core.small_ratio(v2) - core.small_ratio(v1)) * 100:+.0f} pts")
                c1, c2 = dict(core.class_counts(v1)), dict(core.class_counts(v2))
                comp = pd.DataFrame({"Class": sorted(set(c1) | set(c2))})
                comp["Standard"] = comp["Class"].map(lambda k: c1.get(k, 0))
                comp["Tiled"] = comp["Class"].map(lambda k: c2.get(k, 0))
                comp["Difference"] = comp["Tiled"] - comp["Standard"]
                st.dataframe(comp.sort_values("Tiled", ascending=False), hide_index=True, **STRETCH)

        # ---- downloads
        st.subheader("Export")
        stem = Path(f.name).stem
        meta = {"Image": f.name, "Size": f"{shape[1]} x {shape[0]}", "Model": MODEL_NAMES[0] if (BASE_DIR / MODEL_NAMES[0]).exists() else MODEL_NAMES[1],
                "Confidence": P["conf"], "IoU": P["iou"], "Image size": P["imgsz"],
                "Tiled": P["tiled"], "Inference": f"{ms / 1000:.2f} s"}
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Result image (PNG)", core.png_bytes(annotated_bgr), f"{stem}_result.png", "image/png", **STRETCH)
        d2.download_button("Detections (CSV)", core.csv_bytes(det_rows(vis, poly, f.name)), f"{stem}_detections.csv", "text/csv", **STRETCH)
        d3.download_button("Detections (JSON)", core.json_bytes({
            "image": f.name, "width": shape[1], "height": shape[0], "params": {k: meta[k] for k in ("Confidence", "IoU", "Image size", "Tiled")},
            "summary": dict(core.class_counts(vis)), "detections": [d.as_dict() for d in vis]}),
            f"{stem}_detections.json", "application/json", **STRETCH)
        d4.download_button("HTML report", core.html_report(f.name, annotated_bgr, vis, meta),
                           f"{stem}_report.html", "text/html", **STRETCH)

        # ---- batch
        if len(files) > 1:
            st.divider()
            st.subheader(f"Batch summary ({len(files)} images)")
            sig = (tuple((x.name, x.size) for x in files), DET_ARGS, tuple(sorted(P["cls_ids"])), P["zone_on"])
            if st.button("Run on all images", type="primary"):
                st.session_state["batch_sig"] = sig
            if st.session_state.get("batch_sig") == sig:
                rows, per_class, prog = [], defaultdict(dict), st.progress(0.0, text="Processing...")
                results = []
                for i, x in enumerate(files):
                    xd = x.getvalue()
                    ds, t, shp = detect_cached(model, xd, *DET_ARGS)
                    v = visible(ds, P)
                    pz = zone_poly(shp, P)
                    nz, al = zone_stats(v, pz, P)
                    row = {"Image": x.name, "Objects": len(v), "In zone": nz if pz else None, "Alert": al if pz else None,
                           "Time (s)": round(t / 1000, 2)}
                    for k, c in core.class_counts(v):
                        row[k] = c
                    rows.append(row)
                    results.append((x.name, xd, v, pz))
                    prog.progress((i + 1) / len(files), text=f"Processed {i + 1}/{len(files)}")
                prog.empty()
                bdf = pd.DataFrame(rows).fillna(0)
                for c in bdf.columns:
                    if c not in ("Image", "In zone", "Alert", "Time (s)"):
                        bdf[c] = bdf[c].astype(int)
                st.dataframe(bdf, hide_index=True, **STRETCH)
                tot = bdf.drop(columns=["Image", "In zone", "Alert", "Time (s)"], errors="ignore").sum().sort_values(ascending=False)
                m1, m2 = st.columns(2)
                m1.metric("Total objects (all images)", int(bdf["Objects"].sum()))
                m2.metric("Busiest image", bdf.loc[bdf["Objects"].idxmax(), "Image"])
                class_chart([(k, int(v)) for k, v in tot.items() if k != "Objects" and v > 0])
                all_rows = [r for n, _, v, pz in results for r in det_rows(v, pz, n)]
                b1, b2 = st.columns(2)
                b1.download_button("Batch table (CSV)", bdf.to_csv(index=False).encode(), "batch_summary.csv", "text/csv", **STRETCH)
                b2.download_button("All detections (CSV)", core.csv_bytes(all_rows), "batch_detections.csv", "text/csv", **STRETCH)
                if st.button("Build annotated images ZIP"):
                    with st.spinner("Building ZIP..."):
                        named = []
                        for n, xd, v, pz in results:
                            b = decode(xd)
                            named.append((f"{Path(n).stem}_result.png",
                                          core.png_bytes(np.ascontiguousarray(compose(b, v, pz, P, "Annotated")[:, :, ::-1]))))
                        st.session_state["zip"] = (sig, core.zip_bytes(named))
                z = st.session_state.get("zip")
                if z and z[0] == sig:
                    st.download_button("Download ZIP", z[1], "aeroguard_results.zip", "application/zip", **STRETCH)


# ================================================================== VIDEO TAB
def process_video(src_path, stride, max_frames, track, record, ph_img, ph_prog):
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    h0, w0 = frame.shape[:2]
    sc = min(1.0, 960 / w0)
    size = (int(w0 * sc), int(h0 * sc))
    poly = zone_poly((size[1], size[0]), P)
    heat = core.Heat((size[1], size[0]))
    use_track = track and not P["tiled"]
    if use_track:
        core.reset_tracker(model)

    writer, out_path = None, None
    if record:
        out_path = os.path.join(tempfile.gettempdir(), f"aeroguard_{os.getpid()}_{id(frame)}.mp4")
        for codec in ("avc1", "mp4v"):
            w = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), max(fps / stride, 1.0), size)
            if w.isOpened():
                writer = w
                break

    expected = min(max_frames, max(1, -(-total // stride))) if total else max_frames
    rows, unique, done, idx, pending = [], defaultdict(set), 0, 0, frame
    last_raw = last_ann = None
    while True:
        if pending is not None:
            frame, pending = pending, None
        else:
            ok, frame = cap.read()
            if not ok:
                break
        idx += 1
        if (idx - 1) % stride:
            continue
        if done >= max_frames:
            break
        if sc < 1.0:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        dets, ms = core.detect(model, frame, P["conf"], P["iou"], P["imgsz"], P["tiled"], P["tile"],
                               P["overlap"], track=use_track)
        vis = visible(dets, P)
        heat.add(vis)
        for d in vis:
            if d.track_id is not None:
                unique[d.name].add(d.track_id)
        n_in, al = zone_stats(vis, poly, P)
        base = core.draw_zone(frame, poly, al) if poly else frame
        ann = core.annotate(base, vis, P["labels"], P["show_conf"],
                            f"Frame {idx} | Objects: {len(vis)}" + (f" | Zone: {n_in}" if poly else ""))
        if writer is not None:
            writer.write(ann)
        rows.append({"frame": idx, "time_s": round((idx - 1) / fps, 2), "objects": len(vis),
                     "in_zone": n_in, "alert": al, "ms": round(ms)})
        done += 1
        last_raw, last_ann = frame, ann
        if done % 2 == 1 or done == expected:
            ph_img.image(ann[:, :, ::-1], **STRETCH)
        ph_prog.progress(min(done / expected, 1.0), text=f"Processed {done}/{expected} frames")
    cap.release()
    video_bytes = None
    if writer is not None:
        writer.release()
        if out_path and os.path.exists(out_path):
            with open(out_path, "rb") as fh:
                video_bytes = fh.read()
            os.remove(out_path)
    if not rows:
        return None
    heat_img = heat.render(last_raw)
    if poly:
        heat_img = core.draw_zone(heat_img, poly, False)
    return dict(df=pd.DataFrame(rows), unique={k: len(v) for k, v in unique.items()}, tracked=use_track,
                heat=heat_img[:, :, ::-1], last=last_ann[:, :, ::-1], video=video_bytes, fps=fps,
                stride=stride, has_zone=bool(poly))


with tab_vid:
    st.caption("Analyses a video frame by frame. On free cloud hardware keep clips short - use a larger frame "
               "step or a lower frame limit for speed.")
    vfile = st.file_uploader("Upload a drone video", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    c1, c2, c3 = st.columns(3)
    stride = c1.slider("Analyse every N-th frame", 1, 15, 5)
    max_frames = c2.slider("Max frames to analyse", 10, 400, 80, 10)
    track = c3.checkbox("Track objects (unique IDs)", value=True, disabled=tiled,
                        help="Unavailable together with tiled inference.")
    record = st.checkbox("Create annotated output video", value=True)
    if vfile is not None and st.button("▶ Analyse video", type="primary"):
        tmp = os.path.join(tempfile.gettempdir(), f"aeroguard_in_{os.getpid()}_{vfile.size}.mp4")
        with open(tmp, "wb") as fh:
            fh.write(vfile.getvalue())
        ph_prog = st.progress(0.0, text="Starting...")
        ph_img = st.empty()
        try:
            res = process_video(tmp, stride, max_frames, track, record, ph_img, ph_prog)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        ph_prog.empty()
        ph_img.empty()
        if res is None:
            st.error("Could not read any frames from this video.")
        else:
            st.session_state["vid"] = res

    res = st.session_state.get("vid")
    if res is not None:
        df = res["df"]
        st.success(f"Analysed {len(df)} frames.")
        m = st.columns(5 if res["has_zone"] else 4)
        m[0].metric("Frames analysed", len(df))
        m[1].metric("Average objects / frame", f"{df['objects'].mean():.1f}")
        m[2].metric("Peak objects", int(df["objects"].max()))
        m[3].metric("Avg time / frame", f"{df['ms'].mean() / 1000:.2f} s")
        if res["has_zone"]:
            m[4].metric("Alert frames", int(df["alert"].sum()))
        a, b = st.columns(2)
        with a:
            st.subheader("Last analysed frame")
            st.image(res["last"], **STRETCH)
        with b:
            st.subheader("Heatmap (where objects concentrate)")
            st.image(res["heat"], **STRETCH)
        st.subheader("Objects over time")
        st.line_chart(df.set_index("time_s")[["objects"] + (["in_zone"] if res["has_zone"] else [])], **STRETCH)
        if res["tracked"] and res["unique"]:
            st.subheader("Unique objects (tracked IDs)")
            udf = pd.DataFrame(sorted(res["unique"].items(), key=lambda kv: -kv[1]), columns=["Class", "Count"])
            u1, u2 = st.columns([1, 2])
            u1.dataframe(udf, hide_index=True, **STRETCH)
            st.caption("IDs are assigned by ByteTrack on the analysed frames; a larger frame step can split one "
                       "object into several IDs.")
        d1, d2 = st.columns(2)
        d1.download_button("Per-frame stats (CSV)", df.to_csv(index=False).encode(), "video_stats.csv", "text/csv", **STRETCH)
        if res["video"]:
            d2.download_button("Annotated video (MP4)", res["video"], "aeroguard_annotated.mp4", "video/mp4", **STRETCH)
            st.caption("If the file does not play in your browser, open it with VLC.")


# ================================================================== MODEL TAB
with tab_model:
    st.subheader("Model")
    a, b = st.columns(2)
    with a:
        st.table(pd.DataFrame({"Item": ["Architecture", "Dataset", "Classes", "Training", "Precision / Recall",
                                        "mAP50", "mAP50-95"],
                               "Value": ["YOLOv8n (COCO pretrained, fine-tuned)",
                                         "VisDrone2019-DET (6,471 train / 548 val)",
                                         ", ".join(NAMES.values()),
                                         "50 epochs, imgsz 640, batch 16 (Colab T4)",
                                         "0.434 / 0.320", "0.307", "0.171"]}).set_index("Item"))
    with b:
        st.markdown("**Class-wise mAP50 (validation)**")
        st.bar_chart(pd.Series(CLASS_MAP50, name="mAP50").sort_values(), horizontal=True, **STRETCH)
    st.subheader("Known limitations")
    st.markdown(
        "- Small objects (bicycle, pedestrian, people) are weak because they shrink to a few pixels at 640 px "
        "- use **Small-object mode** (tiled inference) to improve them.\n"
        "- Car, van and truck are often confused with each other.\n"
        "- Accuracy drops when the camera angle differs from VisDrone (domain shift).\n"
        "- Only the 10 VisDrone classes are detected - no military vehicles, aircraft or ships.")
    res_dir = BASE_DIR / "results"
    imgs = sorted(p for p in res_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")) if res_dir.exists() else []
    if imgs:
        with st.expander("📈 Training results (from /results)"):
            cols = st.columns(2)
            for i, p in enumerate(imgs[:12]):
                cols[i % 2].image(str(p), caption=p.name, **STRETCH)
