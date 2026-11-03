"""Parse BCCD annotations, build pseudo masks and cache train/val/test arrays."""
import os

import cv2
import numpy as np

from common import BCCD_DIR, CACHE_DIR, IMG_H, IMG_W, build_mask, parse_xml


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for split in ("train", "val", "test"):
        with open(os.path.join(BCCD_DIR, "ImageSets", "Main", f"{split}.txt")) as f:
            names = [l.strip() for l in f if l.strip()]
        X, Y, counts = [], [], []
        for n in names:
            img = cv2.cvtColor(cv2.imread(os.path.join(BCCD_DIR, "JPEGImages", n + ".jpg")), cv2.COLOR_BGR2RGB)
            objs, ow, oh = parse_xml(os.path.join(BCCD_DIR, "Annotations", n + ".xml"))
            X.append(cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA))
            Y.append(build_mask(objs, ow, oh))
            counts.append([sum(o[0] == c for o in objs) for c in (1, 2, 3)])
        X, Y, counts = np.stack(X), np.stack(Y), np.array(counts)
        np.savez_compressed(os.path.join(CACHE_DIR, f"{split}.npz"), X=X, Y=Y, counts=counts, names=np.array(names))
        print(f"{split}: {len(names)} images | mean cells/image RBC {counts[:,0].mean():.1f} "
              f"WBC {counts[:,1].mean():.2f} Platelets {counts[:,2].mean():.2f}")


if __name__ == "__main__":
    main()
