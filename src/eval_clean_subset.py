"""Supplementary: test metrics excluding images whose ground truth has zero RBC annotations
(visibly incomplete labels in BCCD). Reported alongside, never instead of, the full-test results."""
import json, os
import numpy as np, torch
from common import CKPT_DIR, CACHE_DIR, OUT_DIR, CLASSES, confusion, iou_dice
from model import UNet
from train import load
from evaluate import predict, count_cells

X, Y = load("test")
d = np.load(os.path.join(CACHE_DIR, "test.npz")); true = d["counts"]
model = UNet(); model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "unet_best.pt"), map_location="cpu"))
pred = predict(model, X); gt = Y.numpy()
keep = true[:, 0] > 0
print("images with 0 annotated RBC:", int((~keep).sum()), "of", len(keep))
cm = confusion(pred[keep], gt[keep]); iou, dice = iou_dice(cm)
pc = np.array([[count_cells(p, c) for c in (1, 2, 3)] for p in pred[keep]])
err = pc - true[keep]
out = dict(n_images=int(keep.sum()), n_excluded=int((~keep).sum()),
           iou={CLASSES[i]: float(iou[i]) for i in range(4)}, mean_iou_cells=float(iou[1:].mean()),
           mean_dice_cells=float(dice[1:].mean()),
           count_mae=dict(zip(CLASSES[1:], np.abs(err).mean(0).round(3).tolist())),
           rbc_within_1=float((np.abs(err[:, 0]) <= 1).mean()), rbc_within_2=float((np.abs(err[:, 0]) <= 2).mean()),
           rbc_bias=float(err[:, 0].mean()))
json.dump(out, open(os.path.join(OUT_DIR, "test_metrics_clean_subset.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
