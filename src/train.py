"""Train the UNet on cached BCCD data. Saves best checkpoint (by val mean IoU) and history."""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CACHE_DIR, CKPT_DIR, OUT_DIR, confusion, iou_dice
from model import UNet

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(split):
    d = np.load(os.path.join(CACHE_DIR, f"{split}.npz"))
    return torch.from_numpy(d["X"]).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(d["Y"]).long()


def norm(xb):
    return (xb.float() / 255 - MEAN) / STD


def augment(x, y):
    """Random flips / 180-degree rotation (cells have no canonical orientation) + colour jitter."""
    if torch.rand(1) < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if torch.rand(1) < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    x = x.float() / 255
    b = x.shape[0]
    x = x * (1 + 0.25 * (torch.rand(b, 1, 1, 1) - 0.5) * 2) + 0.08 * (torch.rand(b, 3, 1, 1) - 0.5) * 2
    return (x.clamp(0, 1) - MEAN) / STD, y


def dice_loss(logits, y, n=4):
    p = F.softmax(logits, 1)
    oh = F.one_hot(y, n).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1)).mean()


@torch.no_grad()
def evaluate(model, X, Y, bs=16):
    model.eval()
    cm = np.zeros((4, 4), np.int64)
    loss = 0.0
    for i in range(0, len(X), bs):
        lg = model(norm(X[i:i + bs]))
        loss += F.cross_entropy(lg, Y[i:i + bs], reduction="sum").item() / Y[i:i + bs][0].numel()
        cm += confusion(lg.argmax(1).numpy(), Y[i:i + bs].numpy())
    iou, dice = iou_dice(cm)
    return loss / len(X), iou, dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(os.cpu_count() or 2)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    Xtr, Ytr = load("train")
    Xva, Yva = load("val")
    model = UNet(base=a.base)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(Xtr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    # Platelets/WBC are rare -> up-weight them in the cross-entropy term.
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 2.0, 4.0]))

    hist, best = [], -1
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        run = 0.0
        for i in range(0, len(Xtr), a.bs):
            idx = perm[i:i + a.bs]
            x, y = augment(Xtr[idx], Ytr[idx])
            lg = model(x)
            loss = ce(lg, y) + dice_loss(lg, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(idx)
        vl, iou, dice = evaluate(model, Xva, Yva)
        miou = float(iou[1:].mean())  # mean over the 3 cell classes
        hist.append(dict(epoch=ep, train_loss=run / len(Xtr), val_ce=vl, val_miou_cells=miou,
                         val_iou=iou.tolist(), val_dice=dice.tolist()))
        if miou > best:
            best = miou
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "unet_best.pt"))
        print(f"ep {ep:02d}/{a.epochs} loss {run/len(Xtr):.3f} val_ce {vl:.3f} "
              f"mIoU(cells) {miou:.3f} IoU RBC/WBC/Plt {iou[1]:.2f}/{iou[2]:.2f}/{iou[3]:.2f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(hist, open(os.path.join(OUT_DIR, "history.json"), "w"), indent=1)
    print(f"best val mIoU (cells): {best:.4f}")


if __name__ == "__main__":
    main()
