"""Shared constants and helpers for the BCCD blood-cell segmentation project."""
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BCCD_DIR = os.path.join(ROOT, "BCCD")
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_DIR = os.path.join(ROOT, "outputs")
CKPT_DIR = os.path.join(ROOT, "checkpoints")

# Class 0 is background; foreground classes follow the BCCD label names.
CLASSES = ["Background", "RBC", "WBC", "Platelets"]
NAME2ID = {"RBC": 1, "WBC": 2, "Platelets": 3}
# Display colours (RGB) used in all figures.
COLORS = np.array([[0, 0, 0], [220, 50, 50], [50, 90, 230], [40, 190, 90]], dtype=np.uint8)

# Network input size (original images are 640x480). Both sides divisible by 16.
IMG_W, IMG_H = 320, 240

# Minimum blob area (pixels at 320x240) for a prediction to be counted as a cell.
MIN_AREA = {1: 60, 2: 150, 3: 10}


def parse_xml(path):
    """Return (list of (class_id, xmin, ymin, xmax, ymax), original_w, original_h)."""
    root = ET.parse(path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    objs = []
    for o in root.findall("object"):
        name = o.find("name").text
        if name not in NAME2ID:
            continue
        b = o.find("bndbox")
        objs.append((NAME2ID[name], *[float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]))
    return objs, w, h


def build_mask(objs, ow, oh, W=IMG_W, H=IMG_H):
    """Pseudo pixel mask: one filled ellipse inscribed in every bounding box.

    BCCD only has bounding boxes, so this is the standard way to obtain dense labels.
    Draw order RBC -> WBC -> Platelets so small/rare classes stay visible on top.
    """
    sx, sy = W / ow, H / oh
    mask = np.zeros((H, W), np.uint8)
    for cls in (1, 2, 3):
        for c, x0, y0, x1, y1 in objs:
            if c != cls:
                continue
            cx, cy = (x0 + x1) / 2 * sx, (y0 + y1) / 2 * sy
            ax, ay = max((x1 - x0) / 2 * sx, 1), max((y1 - y0) / 2 * sy, 1)
            cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (int(round(ax)), int(round(ay))), 0, 0, 360, int(cls), -1)
    return mask


def colorize(mask):
    return COLORS[mask]


def confusion(pred, gt, n=4):
    k = (gt >= 0) & (gt < n)
    return np.bincount(n * gt[k].astype(int) + pred[k].astype(int), minlength=n * n).reshape(n, n)


def iou_dice(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return iou, dice
