"""Run YOLOv3 object detection on an image or a video.

Usage:
    python detect.py path/to/image.jpg                # writes image_detected.jpg
    python detect.py path/to/video.mp4                # writes video_detected.mp4
    python detect.py input.jpg -o out.jpg --obj-thresh 0.5 --nms-thresh 0.45
"""
import argparse
import os
import urllib.request
import warnings

# Hide harmless TensorFlow/Keras startup messages (e.g. the LeakyReLU "alpha" warning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from PIL import Image

from yolo_helpers import ANCHORS, LABELS, detect_image, detect_video

WEIGHTS_URL = ("https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/"
               "Sessions%206%20-%2010%20(Projects)/Project%20-%20%20Object%20Detection%20"
               "(Autonomous%20Vehicles)/yolo.h5")
DEFAULT_WEIGHTS = os.path.join("models", "yolo_weights.h5")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_model(weights_path=DEFAULT_WEIGHTS):
    """Load the pre-trained DarkNet, downloading the weights on first run."""
    if not os.path.exists(weights_path):
        os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
        print("Downloading YOLO weights (~250 MB) to {} ...".format(weights_path))
        urllib.request.urlretrieve(WEIGHTS_URL, weights_path)
    return tf.keras.models.load_model(weights_path, compile=False)


def main():
    p = argparse.ArgumentParser(description="YOLOv3 object detection")
    p.add_argument("input", help="image or video file")
    p.add_argument("-o", "--output", help="output path (default: <input>_detected.<ext>)")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to yolo .h5/.keras weights")
    p.add_argument("--obj-thresh", type=float, default=0.4)
    p.add_argument("--nms-thresh", type=float, default=0.45)
    args = p.parse_args()

    base, ext = os.path.splitext(args.input)
    output = args.output or "{}_detected{}".format(base, ".mp4" if ext.lower() in VIDEO_EXTS else ext)

    model = load_model(args.weights)
    opts = dict(anchors=ANCHORS, labels=LABELS,
                obj_thresh=args.obj_thresh, nms_thresh=args.nms_thresh)

    if ext.lower() in VIDEO_EXTS:
        detect_video(args.input, output, model, **opts)
    else:
        result = detect_image(Image.open(args.input), model, verbose=True, **opts)
        result.save(output)
        print("Saved", output)


if __name__ == "__main__":
    main()

