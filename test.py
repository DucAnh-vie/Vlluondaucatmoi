import argparse
import os
import time
import torch
from ultralytics_custom import YOLO
import cv2

def main(args):
    # Convert device string for PyTorch warm-up
    torch_device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"

    # Load YOLO model
    model = YOLO(args.model_path)

    # Warm-up with dummy tensor
    dummy = torch.zeros((1, 3, args.imgsz, args.imgsz)).to(torch_device)
    model(dummy)  # forward pass

    print(f"\n=== Batch detection with threshold {args.conf:.2f} on device {args.device} ===")

    # Create output folder if saving images
    if args.save_images:
        os.makedirs(args.output_folder, exist_ok=True)
        saved_count = 0
        print(f"Sẽ lưu ảnh vào: {args.output_folder}")
        if args.min_detections > 0:
            print(f"Chỉ lưu ảnh có ít nhất {args.min_detections} object được detect")
        if args.max_save > 0:
            print(f"Giới hạn lưu tối đa {args.max_save} ảnh")

    # Start overall timing
    start = time.time()

    # Variables to accumulate speed breakdown
    total_preprocess = 0
    total_inference = 0
    total_postprocess = 0
    count = 0

    # Predict using stream=True to avoid RAM accumulation
    results = model.predict(
        source=args.image_folder,
        imgsz=args.imgsz,
        conf=args.conf,
        save=False,  # Tắt save mặc định của YOLO, tự xử lý
        stream=True,
        device=args.device,
        verbose=False,
        batch=args.batch,
        max_det=100
    )

    # Iterate through results to accumulate timing
    for r in results:
        count += 1
        total_preprocess += r.speed['preprocess']
        total_inference += r.speed['inference']
        total_postprocess += r.speed['postprocess']

        # Lưu ảnh nếu được yêu cầu
        if args.save_images and (args.max_save == 0 or saved_count < args.max_save):
            num_detections = len(r.boxes)
            
            # Kiểm tra điều kiện số lượng detection
            if num_detections >= args.min_detections:
                # Vẽ bounding boxes lên ảnh
                img_with_boxes = r.plot()  # Trả về numpy array với boxes đã vẽ
                
                # Tạo tên file
                img_name = os.path.basename(r.path)
                name_without_ext = os.path.splitext(img_name)[0]
                output_path = os.path.join(
                    args.output_folder, 
                    f"{name_without_ext}_detected_{num_detections}obj.jpg"
                )
                
                # Lưu ảnh
                cv2.imwrite(output_path, img_with_boxes)
                saved_count += 1
                
                if args.verbose_save:
                    print(f"Đã lưu: {output_path} ({num_detections} objects)")

    end = time.time()

    # Calculate statistics
    total_time = end - start
    avg_time = total_time / count if count > 0 else 0
    fps = count / total_time if total_time > 0 else 0

    # Average speed breakdown per image
    avg_pre = total_preprocess / count
    avg_inf = total_inference / count
    avg_post = total_postprocess / count

    # Print results
    print(f"\nProcessed {count} images.")
    if args.save_images:
        print(f"Đã lưu {saved_count} ảnh vào {args.output_folder}")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average inference time per image: {avg_time:.3f} seconds")
    print(f"FPS: {fps:.2f} frames/second")
    print(f"Speed: {avg_pre:.1f}ms preprocess, {avg_inf:.1f}ms inference, {avg_post:.1f}ms postprocess "
          f"per image at shape (1, 3, {args.imgsz}, {args.imgsz})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized YOLO Batch Detection Script with Speed Breakdown and FPS")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model .pt file")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing test images")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="0", help="Device: 0 = GPU, 'cpu' = CPU")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for inference")
    
    # Arguments cho việc lưu ảnh
    parser.add_argument("--save_images", action="store_true", help="Lưu ảnh đã detect")
    parser.add_argument("--output_folder", type=str, default="output_images", help="Thư mục lưu ảnh output")
    parser.add_argument("--min_detections", type=int, default=0, help="Số object tối thiểu để lưu ảnh (0 = lưu tất cả)")
    parser.add_argument("--max_save", type=int, default=0, help="Số lượng ảnh tối đa cần lưu (0 = không giới hạn)")
    parser.add_argument("--verbose_save", action="store_true", help="In thông tin chi tiết khi lưu ảnh")

    args = parser.parse_args()
    main(args)
