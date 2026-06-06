from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import numpy as np
import os
import torch

def cumpute_classes_weights(mask_dir):
    mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')][:100]

    all_pixels = []

    for path in mask_files:
        mask = np.array(Image.open(path).convert('L')) # загружаем как индексы (0, 1, 2...)
        all_pixels.append(mask.flatten())

    all_pixels = np.concatenate(all_pixels)
    unique_classes = np.unique(all_pixels)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=all_pixels
    )

    final_weights = [round(w, 4) for w in weights]

    return final_weights

def compute_iou(preds, targets, n_classes, ignore_index=11):
        ious = []
        for cls in range(n_classes):
            if cls == ignore_index:
                continue
            pred_cls = (preds == cls)
            target_cls = (targets == cls)
            intersection = (pred_cls & target_cls).float().sum((1, 2))
            union = (pred_cls | target_cls).float().sum((1, 2))
            iou = (intersection + 1e-6) / (union + 1e-6)
            ious.append(iou.mean().item())
        return torch.tensor(ious).mean(), torch.tensor(ious)
