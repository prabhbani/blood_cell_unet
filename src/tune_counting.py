"""Tune RBC watershed parameters on the VALIDATION split only (test split is never used here)."""
import itertools, json, os
import numpy as np, torch, cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from common import CKPT_DIR, CACHE_DIR
from model import UNet
from train import load
from evaluate import predict

X, Y = load("val")
model = UNet(); model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "unet_best.pt"), map_location="cpu"))
pred = predict(model, X)
true = np.load(os.path.join(CACHE_DIR, "val.npz"))["counts"][:, 0]

def count(p, md, th, minarea, smooth):
    m = (p == 1).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dist = ndi.distance_transform_edt(m)
    if smooth: dist = ndi.gaussian_filter(dist, smooth)
    cc, _ = ndi.label(m)
    pk = peak_local_max(dist, min_distance=md, threshold_abs=th, labels=cc)
    mk = np.zeros_like(cc)
    for i, (r, c) in enumerate(pk, 1): mk[r, c] = i
    lab = watershed(-dist, mk, mask=m.astype(bool))
    n = sum(1 for i in range(1, lab.max() + 1) if (lab == i).sum() >= minarea)
    orphan = m.astype(bool) & (lab == 0)
    ol, on = ndi.label(orphan)
    return n + sum(1 for i in range(1, on + 1) if (ol == i).sum() >= minarea)

res = []
for md, th, ma, sm in itertools.product([8, 10, 12, 15], [4, 6, 8], [30, 60, 100], [0, 1.5, 2.5]):
    pc = np.array([count(p, md, th, ma, sm) for p in pred])
    res.append((float(np.abs(pc - true).mean()), float((pc - true).mean()), md, th, ma, sm))
res.sort()
for r in res[:8]: print("MAE %.2f bias %+.2f | min_distance %d thr %d minarea %d smooth %.1f" % r)
json.dump(dict(mae=res[0][0], min_distance=res[0][2], threshold_abs=res[0][3], min_area=res[0][4], smooth=res[0][5]),
          open(os.path.join(os.path.dirname(CKPT_DIR), "outputs", "rbc_counting_params.json"), "w"))
