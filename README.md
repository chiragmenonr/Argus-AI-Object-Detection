# YOLOv3 Object Detection for Autonomous Driving

Detects cars, people, traffic lights and the other 77 COCO classes in images and
videos using a pre-trained YOLOv3 (DarkNet) model. Converted from a Google Colab
notebook into a runnable project.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The pre-trained weights (~250 MB) are downloaded automatically to
`models/yolo_weights.h5` the first time you run the code. They are not stored in
the repo because GitHub rejects files over 100 MB.

## Run it

**On an image:**

```bash
python detect.py my_photo.jpg            # saves my_photo_detected.jpg
```

**On a video:**

```bash
python detect.py dashcam.mp4             # saves dashcam_detected.mp4
```

Options: `-o OUTPUT`, `--obj-thresh 0.4`, `--nms-thresh 0.45`, `--weights PATH`.

**Web demo (upload an image in your browser):**

```bash
python app.py                            # open http://localhost:7860
```

## How it works

1. `preprocess_input` letterboxes the image to 416×416.
2. The DarkNet model outputs three feature maps (13×13, 26×26, 52×52), one per detection scale.
3. `decode_netout` turns those into boxes above the objectness threshold.
4. `do_nms` (non-maximal suppression) removes overlapping duplicates.
5. `draw_boxes` draws the labeled boxes.

`detect_image()` in `yolo_helpers.py` runs the whole pipeline.

## Files

| File | Purpose |
|---|---|
| `detect.py` | Command-line entry point (image or video) |
| `app.py` | Optional Gradio web demo |
| `yolo_helpers.py` | Preprocessing, decoding, NMS, drawing, `detect_image`, `detect_video` |
| `requirements.txt` | Python dependencies |

A GPU makes video processing much faster but is not required.
