"""Optional web demo: upload an image, get it back with detection boxes.

    python app.py   # then open http://localhost:7860
"""
import os

import gradio as gr

from detect import load_model
from yolo_helpers import detect_image

model = load_model()


def run(image, obj_thresh, nms_thresh):
    if image is None:
        return None
    return detect_image(image, model, obj_thresh=obj_thresh, nms_thresh=nms_thresh)


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Slider(0.05, 0.95, value=0.4, step=0.05, label="Objectness threshold"),
        gr.Slider(0.05, 0.95, value=0.45, step=0.05, label="NMS threshold"),
    ],
    outputs=gr.Image(type="pil", label="Detections"),
    title="YOLOv3 Object Detection",
    description="Detects the 80 COCO classes (cars, people, traffic lights, ...).",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
