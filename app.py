import io

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------- Page setup ----------
st.set_page_config(page_title="AeroGuard AI", layout="wide")
st.title("AeroGuard AI: Aerial Object Detection Dashboard")
st.caption("YOLOv8n trained on VisDrone (mAP50 = 0.307)")


# ---------- Load the model once ----------
@st.cache_resource
def load_model():
    return YOLO("best_final.pt")  # keep this file next to app.py


model = load_model()

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
imgsz = st.sidebar.selectbox("Image size", [640, 960, 1280], index=2)
show_labels = st.sidebar.checkbox("Show labels on image", value=False)

# ---------- Upload ----------
uploaded = st.file_uploader("Upload a drone image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    # Run detection
    result = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]

    # Draw boxes (plot() returns BGR, so we flip to RGB)
    annotated = result.plot(labels=show_labels, conf=show_labels)[:, :, ::-1]

    # Count objects per class
    counts = {}
    for cls_id in result.boxes.cls.tolist():
        name = result.names[int(cls_id)]
        counts[name] = counts.get(name, 0) + 1

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Detection result")
        st.image(annotated, use_container_width=True)

        # Download button
        buffer = io.BytesIO()
        Image.fromarray(annotated).save(buffer, format="PNG")
        st.download_button("Download result image", buffer.getvalue(),
                           file_name="detection_result.png", mime="image/png")

    with right:
        st.subheader("Summary")
        st.metric("Total objects", len(result.boxes))

        if counts:
            df = pd.DataFrame(
                {"Class": list(counts.keys()), "Count": list(counts.values())}
            ).sort_values("Count", ascending=False)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.bar_chart(df.set_index("Class"))
        else:
            st.info("No objects found. Try lowering the confidence threshold.")
else:
    st.info("Upload a drone image to start.")
