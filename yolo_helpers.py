"""YOLOv3 helper functions: preprocessing, output decoding, NMS, and drawing.

Taken from the Colab notebook's setup cell / export cell.
"""
import colorsys
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABELS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

ANCHORS = [[[116, 90], [156, 198], [373, 326]],
           [[30, 61], [62, 45], [59, 119]],
           [[10, 13], [16, 30], [33, 23]]]


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        return 0 if x4 < x1 else min(x2, x4) - x1
    else:
        return 0 if x2 < x3 else min(x2, x4) - x3


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


def preprocess_input(image_pil, net_h, net_w):
    """Letterbox the image to (net_h, net_w), scale to [0,1], add batch dim."""
    image = np.asarray(image_pil.convert("RGB"))
    new_h, new_w, _ = image.shape
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) / new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) / new_h
        new_h = net_h
    new_w, new_h = int(new_w), int(new_h)
    resized = cv2.resize(image / 255., (new_w, new_h))
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2),
              int((net_w - new_w) // 2):int((net_w + new_w) // 2), :] = resized
    return np.expand_dims(new_image, 0)


def decode_netout(netout_, obj_thresh, anchors_, image_h, image_w, net_h, net_w):
    netout_all = deepcopy(netout_)
    boxes_all = []
    for i in range(len(netout_all)):
        netout = netout_all[i][0]
        anchors = anchors_[i]
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        boxes = []
        netout[..., :2] = _sigmoid(netout[..., :2])
        netout[..., 4:] = _sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh
        for j in range(grid_h * grid_w):
            row = j // grid_w
            col = j % grid_w
            for b in range(nb_box):
                objectness = netout[row][col][b][4]
                classes = netout[row][col][b][5:]
                if (classes <= obj_thresh).all():
                    continue
                x, y, w, h = netout[row][col][b][:4]
                x = (col + x) / grid_w
                y = (row + y) / grid_h
                w = anchors[b][0] * np.exp(w) / net_w
                h = anchors[b][1] * np.exp(h) / net_h
                boxes.append(BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes))
        boxes_all += boxes
    return correct_yolo_boxes(boxes_all, image_h, image_w, net_h, net_w)


def correct_yolo_boxes(boxes_, image_h, image_w, net_h, net_w):
    boxes = deepcopy(boxes_)
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h
    for i in range(len(boxes)):
        x_offset = (net_w - new_w) / 2. / net_w
        x_scale = float(new_w) / net_w
        y_offset = (net_h - new_h) / 2. / net_h
        y_scale = float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
    return boxes


def do_nms(boxes_, nms_thresh, obj_thresh):
    boxes = deepcopy(boxes_)
    if len(boxes) == 0:
        return []
    num_class = len(boxes[0].classes)
    for c in range(num_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0:
                continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
    new_boxes = []
    for box in boxes:
        for i in range(num_class):
            if box.classes[i] > obj_thresh:
                box.label = i
                box.score = box.classes[i]
                new_boxes.append(box)
                break
    return new_boxes


def draw_boxes(image_, boxes, labels, verbose=False):
    image = image_.copy()
    image_w, image_h = image.size
    try:
        font = ImageFont.truetype(
            font="/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            size=int(np.floor(3e-2 * image_h + 0.5)))
    except Exception:
        try:
            font = ImageFont.load_default(size=int(np.floor(3e-2 * image_h + 0.5)))
        except TypeError:  # older Pillow
            font = ImageFont.load_default()
    thickness = max(1, (image_w + image_h) // 300)
    hsv_tuples = [(x / len(labels), 1., 1.) for x in range(len(labels))]
    colors = [tuple(int(v * 255) for v in colorsys.hsv_to_rgb(*x)) for x in hsv_tuples]
    rng = np.random.RandomState(10101)  # fixed colors across runs
    rng.shuffle(colors)

    draw = ImageDraw.Draw(image)
    for box in reversed(boxes):
        c = box.get_label()
        label = "{} {:.2f}".format(labels[c], box.get_score())
        bbox = draw.textbbox((0, 0), label, font)
        label_h = bbox[3]
        top = max(0, int(np.floor(box.ymin + 0.5)))
        left = max(0, int(np.floor(box.xmin + 0.5)))
        bottom = min(image_h, int(np.floor(box.ymax + 0.5)))
        right = min(image_w, int(np.floor(box.xmax + 0.5)))
        if verbose:
            print(label, (left, top), (right, bottom))
        text_origin = (left, top - label_h) if top - label_h >= 0 else (left, top + 1)
        if right > left and bottom > top:
            draw.rectangle([left, top, right, bottom], outline=colors[c], width=thickness)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    return image


def detect_image(image_pil, model, anchors=ANCHORS, labels=LABELS,
                 obj_thresh=0.4, nms_thresh=0.45, net_h=416, net_w=416, verbose=False):
    """Full pipeline: preprocess -> DarkNet forward pass -> decode -> NMS -> draw."""
    image_pil = image_pil.convert("RGB")
    image_w, image_h = image_pil.size
    new_image = preprocess_input(image_pil, net_h, net_w)
    yolo_outputs = model.predict(new_image, verbose=0)
    boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_h, image_w, net_h, net_w)
    boxes = do_nms(boxes, nms_thresh, obj_thresh)
    return draw_boxes(image_pil, boxes, labels, verbose=verbose)


def detect_video(video_path, output_path, model, **kwargs):
    """Run detect_image on every frame of a video and write an annotated .mp4."""
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open video: {}".format(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = vid.get(cv2.CAP_PROP_FPS) or 30
    size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    n = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        n += 1
        # One progress line that updates in place instead of a new line per frame
        if total:
            msg = "Processing frame {}/{} ({:.0f}%)".format(n, total, 100 * n / total)
        else:
            msg = "Processing frame {}".format(n)
        print("\r" + msg, end="", flush=True)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil = detect_image(pil, model, **kwargs)
        out.write(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))

    vid.release()
    out.release()
    print("\nDone! Saved {} frames to {}".format(n, output_path))
