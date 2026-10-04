# AeroGuard AI: Aerial Object Detection Dashboard

A Streamlit dashboard that detects vehicles and people in drone images.
The detector is YOLOv8n trained on the VisDrone dataset.

## Features
- Upload a drone image and see detected objects drawn on it
- Class-wise count table and bar chart
- Adjustable confidence threshold and image size
- Download the annotated image

## Model
| Item | Value |
|---|---|
| Architecture | YOLOv8n (pretrained on COCO, fine-tuned) |
| Dataset | VisDrone2019-DET (6,471 train / 548 val images) |
| Classes | pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor |
| Training | 50 epochs, image size 640, batch 16 (Google Colab, T4 GPU) |
| Precision / Recall | 0.434 / 0.320 |
| mAP50 | 0.307 |
| mAP50-95 | 0.171 |

### Class-wise mAP50 (validation set)
| Class | mAP50 |
|---|---|
| car | 0.725 |
| bus | 0.434 |
| van | 0.355 |
| motor | 0.333 |
| pedestrian | 0.315 |
| truck | 0.275 |
| people | 0.249 |
| tricycle | 0.208 |
| awning-tricycle | 0.109 |
| bicycle | 0.064 |

## Limitations
- Small objects (bicycle, pedestrian, people) are detected poorly because
  they become only a few pixels wide after resizing to 640.
- Car, van and truck are often confused with each other.
- Performance drops on images with a different camera angle than VisDrone
  (domain shift).
- The model does not detect military vehicles, aircraft or ships.
  VisDrone contains only the 10 classes listed above.

## Project structure
```
app.py               Streamlit dashboard
best_final.pt        Trained model weights
requirements.txt     Python dependencies
results/             Training curves, confusion matrices, sample predictions
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
If `streamlit` is not recognized, use `python -m streamlit run app.py`.

## Training
Training was done in Google Colab with Ultralytics. The VisDrone dataset is
not included in this repository; Ultralytics downloads it automatically
when training with `data="VisDrone.yaml"`.

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="VisDrone.yaml", epochs=50, imgsz=640, batch=16)
```
