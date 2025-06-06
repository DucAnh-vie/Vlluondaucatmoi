import os
import time
import gc
import torch
import random
from pathlib import Path
from ultralytics_custom.engine.predictor import BasePredictor
from ultralytics_custom.engine.results import Results
from ultralytics_custom.utils import ops


class DetectionPredictor(BasePredictor):
    """Custom Detection Predictor that properly handles postprocessing."""

    def postprocess(self, preds, img, orig_imgs):
        """Apply Non-maximum suppression to prediction outputs."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            getattr(self.args, 'iou', 0.45),  # Default IOU threshold
            agnostic=getattr(self.args, 'agnostic_nms', False),
            max_det=getattr(self.args, 'max_det', 300),
        )

        # Move predictions to the same device as model
        if isinstance(preds, (list, tuple)):
            preds = [p.to(self.model.device) if isinstance(p, torch.Tensor) else p for p in preds]
        elif isinstance(preds, torch.Tensor):
            preds = preds.to(self.model.device)

        results = []
        for i, pred in enumerate(preds):
            shape = orig_imgs[i].shape if isinstance(orig_imgs, list) else orig_imgs.shape
            path = self.batch[i]

            if not pred.shape[0]:
                results.append(Results(orig_img=orig_imgs[i], path=path, names=self.model.names))
                continue

            pred[:, :4] = ops.scale_coords(img.shape[2:], pred[:, :4], shape).round()

            results.append(Results(
                orig_img=orig_imgs[i],
                path=path,
                names=self.model.names,
                boxes=pred
            ))

        return results

    def preprocess(self, img):
        img = super().preprocess(img)
        return img.to(self.model.device).float() / 255.0  # Normalize and move to GPU

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warm up model with empty input to initialize GPU memory
        if not isinstance(imgsz, torch.Size):
            imgsz = torch.Size(imgsz)
        dummy_input = torch.zeros(*imgsz, device=self.model.device)
        with torch.no_grad():
            self.model(dummy_input)

def get_limited_images(source_dir, max_images=1000):
    """Lấy tối đa max_images ảnh từ thư mục."""
    # Các định dạng ảnh được hỗ trợ
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Lấy tất cả file ảnh
    all_images = []
    for file in os.listdir(source_dir):
        if Path(file).suffix.lower() in image_extensions:
            all_images.append(os.path.join(source_dir, file))
    
    # Nếu có ít hơn max_images thì lấy tất cả
    if len(all_images) <= max_images:
        return all_images
    
    # Ngẫu nhiên chọn max_images ảnh
    selected_images = random.sample(all_images, max_images)
    return selected_images

model_path = '/content/best.pt'
source_dir = '/content/test_images'
max_images = 4000  # Số ảnh tối đa muốn test

# Giải phóng RAM trước khi chạy
gc.collect()
torch.cuda.empty_cache()

# Lấy danh sách ảnh giới hạn
selected_images = get_limited_images(source_dir, max_images)
print(f"Tổng số ảnh trong thư mục: {len(os.listdir(source_dir))}")
print(f"Số ảnh được chọn để test: {len(selected_images)}")

start = time.time()

overrides = {
    'model': model_path,
    'task': 'detect',
    'mode': 'predict',
    'source': selected_images,  # Sử dụng danh sách ảnh đã chọn
    'imgsz': 640,
    'conf': 0.4,
    'max_det': 50,
    'save': False,
    'verbose': False
}

predictor = DetectionPredictor(overrides=overrides)
# Use predict_cli() instead of predict()
predictor.predict_cli()

end = time.time()
elapsed = end - start

print(f"\nTotal time: {elapsed:.2f} seconds")
print(f"Avg time/image: {elapsed / len(selected_images):.4f} sec")
