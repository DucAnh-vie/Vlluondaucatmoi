from ultralytics import YOLO
import os
import time
import numpy as np

# Load model custom
model = YOLO('runs/detect/train/weights/best.pt')

# File YAML chứa đường dẫn tập val/test
data_yaml = 'path/to/data.yaml'

# Dải threshold để thử
conf_thresholds = np.arange(0.1, 0.91, 0.1)
map_results = []

print("=== Tìm threshold tốt nhất theo mAP50-95 ===")
for conf in conf_thresholds:
    metrics = model.val(data=data_yaml, conf=conf, iou=0.5, split='val', verbose=False)
    map_score = metrics.box.map
    map_results.append((conf, map_score))
    print(f"conf={conf:.2f} -> mAP50-95={map_score:.4f}")

# Chọn threshold tốt nhất
best_conf, best_map = max(map_results, key=lambda x: x[1])
print(f"\nThreshold tốt nhất: {best_conf:.2f} (mAP50-95={best_map:.4f})")
