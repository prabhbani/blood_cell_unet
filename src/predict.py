"""Run the trained UNet on any blood-smear image: saves a mask overlay and prints cell counts.

Usage:  python predict.py path/to/smear.jpg [out.png]
"""
import os
import sys

import cv2
import numpy as np
import torch

from common import CKPT_DIR, COLORS, IMG_H, IMG_W
from evaluate import count_cells
from model import UNet
from train import norm


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "prediction.png"
    img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
    small = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    model = UNet()
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "unet_best.pt"), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        x = norm(torch.from_numpy(small).permute(2, 0, 1)[None])
        pred = model(x).argmax(1)[0].numpy().astype(np.uint8)
    counts = {n: count_cells(pred, c) for n, c in (("RBC", 1), ("WBC", 2), ("Platelets", 3))}
    total = max(sum(counts.values()), 1)
    overlay = (0.6 * small + 0.4 * COLORS[pred]).astype(np.uint8)
    cv2.imwrite(out, cv2.cvtColor(cv2.resize(overlay, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
    print("Counts:", counts)
    print("Differential (%):", {k: round(100 * v / total, 1) for k, v in counts.items()})
    print("Saved", out)


if __name__ == "__main__":
    main()
