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
            classes=getattr(self.args, 'classes', None),
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            if len(pred):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

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
