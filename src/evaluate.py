"""Evaluate the trained UNet on the BCCD test split: IoU/Dice, cell counts, figures."""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from common import CKPT_DIR, CLASSES, COLORS, MIN_AREA, OUT_DIR, colorize, confusion, iou_dice
from model import UNet
from train import load, norm


RBC_MIN_DIST, RBC_PEAK_THR, RBC_MIN_AREA, RBC_SMOOTH = 12, 8, 100, 2.5


def count_cells(pred, cls):
    """Count instances of class `cls` in a predicted label map.

    RBCs touch each other a lot, so they are split with a distance-transform watershed.
    WBCs / platelets are counted as connected components above a minimum area.
    """
    m = (pred == cls).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if cls == 1:
        # Parameters chosen by grid search on the VALIDATION split (see tune_counting.py).
        dist = ndi.gaussian_filter(ndi.distance_transform_edt(m), RBC_SMOOTH)
        cc, n = ndi.label(m)
        peaks = peak_local_max(dist, min_distance=RBC_MIN_DIST, threshold_abs=RBC_PEAK_THR, labels=cc)
        markers = np.zeros_like(cc)
        for i, (r, c) in enumerate(peaks, 1):
            markers[r, c] = i
        lab = watershed(-dist, markers, mask=m.astype(bool))
        n_cells = sum(1 for i in range(1, lab.max() + 1) if (lab == i).sum() >= RBC_MIN_AREA)
        # blobs with no peak (small / flat) still count if big enough
        orphan = (m.astype(bool)) & (lab == 0)
        ol, on = ndi.label(orphan)
        return n_cells + sum(1 for i in range(1, on + 1) if (ol == i).sum() >= RBC_MIN_AREA)
    lab, n = ndi.label(m)
    return int(sum(1 for i in range(1, n + 1) if (lab == i).sum() >= MIN_AREA[cls]))


@torch.no_grad()
def predict(model, X, bs=16):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(norm(X[i:i + bs])).argmax(1).numpy().astype(np.uint8))
    return np.concatenate(out)


def main():
    X, Y = load("test")
    d = np.load(os.path.join(os.path.dirname(OUT_DIR), "cache", "test.npz"))
    true_counts, names = d["counts"], d["names"]
    model = UNet()
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "unet_best.pt"), map_location="cpu"))
    pred = predict(model, X)
    gt = Y.numpy()

    cm = confusion(pred, gt)
    iou, dice = iou_dice(cm)
    pred_counts = np.array([[count_cells(p, c) for c in (1, 2, 3)] for p in pred])
    err = pred_counts - true_counts
    mae = np.abs(err).mean(0)
    exact = (err == 0).mean(0)
    within1 = (np.abs(err) <= 1).mean(0)
    tot_true, tot_pred = true_counts.sum(0), pred_counts.sum(0)
    pct_true = 100 * tot_true / tot_true.sum()
    pct_pred = 100 * tot_pred / tot_pred.sum()

    res = dict(
        n_test=len(X),
        iou={CLASSES[i]: float(iou[i]) for i in range(4)},
        dice={CLASSES[i]: float(dice[i]) for i in range(4)},
        mean_iou_cells=float(iou[1:].mean()), mean_iou_all=float(iou.mean()),
        mean_dice_cells=float(dice[1:].mean()),
        pixel_accuracy=float(np.trace(cm) / cm.sum()),
        count_mae={c: float(mae[i]) for i, c in enumerate(CLASSES[1:])},
        count_exact_match={c: float(exact[i]) for i, c in enumerate(CLASSES[1:])},
        count_within_1={c: float(within1[i]) for i, c in enumerate(CLASSES[1:])},
        total_true={c: int(tot_true[i]) for i, c in enumerate(CLASSES[1:])},
        total_pred={c: int(tot_pred[i]) for i, c in enumerate(CLASSES[1:])},
        differential_true_pct={c: float(pct_true[i]) for i, c in enumerate(CLASSES[1:])},
        differential_pred_pct={c: float(pct_pred[i]) for i, c in enumerate(CLASSES[1:])},
        confusion_matrix=cm.tolist(),
    )
    json.dump(res, open(os.path.join(OUT_DIR, "test_metrics.json"), "w"), indent=2)
    np.savetxt(os.path.join(OUT_DIR, "test_counts.csv"),
               np.column_stack([np.arange(len(names)), true_counts, pred_counts]), fmt="%d", delimiter=",",
               header="idx,true_RBC,true_WBC,true_Plt,pred_RBC,pred_WBC,pred_Plt", comments="")
    with open(os.path.join(OUT_DIR, "test_image_names.txt"), "w") as f:
        f.write("\n".join(map(str, names)))

    # ---- figures ----
    h = json.load(open(os.path.join(OUT_DIR, "history.json")))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([e["train_loss"] for e in h], label="train (CE+Dice)")
    ax[0].plot([e["val_ce"] for e in h], label="val CE")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot([e["val_miou_cells"] for e in h], color="C2")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mIoU (RBC/WBC/Plt)"); ax[1].set_title("Validation mean IoU")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150); plt.close()

    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30); ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Pixel confusion (row-normalised)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for i, c in enumerate(CLASSES[1:]):
        jitter = (np.random.RandomState(i).rand(len(X)) - .5) * 0.25
        ax[i].scatter(true_counts[:, i] + jitter, pred_counts[:, i] + jitter, s=14, alpha=.6, color=COLORS[i + 1] / 255)
        lim = max(true_counts[:, i].max(), pred_counts[:, i].max()) + 1
        ax[i].plot([0, lim], [0, lim], "k--", lw=1)
        ax[i].set_title(f"{c}: MAE {mae[i]:.2f}"); ax[i].set_xlabel("ground-truth count"); ax[i].set_ylabel("predicted count")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "count_scatter.png"), dpi=150); plt.close()

    # qualitative grid: images with the best / median / worst RBC+WBC+Plt count error plus first ones
    order = np.argsort(np.abs(err).sum(1))
    pick = list(order[:2]) + list(order[len(order) // 2:len(order) // 2 + 2]) + list(order[-2:])
    fig, ax = plt.subplots(len(pick), 3, figsize=(9, 2.3 * len(pick)))
    for r, i in enumerate(pick):
        ax[r, 0].imshow(X[i].permute(1, 2, 0).numpy())
        ax[r, 1].imshow(colorize(gt[i])); ax[r, 2].imshow(colorize(pred[i]))
        ax[r, 0].set_ylabel(f"#{i}", rotation=0, labelpad=18)
        ax[r, 1].set_xlabel(f"GT  R{true_counts[i,0]} W{true_counts[i,1]} P{true_counts[i,2]}", fontsize=8)
        ax[r, 2].set_xlabel(f"Pred R{pred_counts[i,0]} W{pred_counts[i,1]} P{pred_counts[i,2]}", fontsize=8)
        for a_ in ax[r]:
            a_.set_xticks([]); a_.set_yticks([])
    for a_, t in zip(ax[0], ["Input", "Ground truth (ellipse masks)", "UNet prediction"]):
        a_.set_title(t, fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "qualitative_results.png"), dpi=130); plt.close()

    print(json.dumps({k: v for k, v in res.items() if k != "confusion_matrix"}, indent=1))


if __name__ == "__main__":
    main()
