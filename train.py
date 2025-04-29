import argparse
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

def parse_args():
    parser = argparse.ArgumentParser(description="Train và Validate YOLO model")
    parser.add_argument("-m", "--model", type=str, default="yolov8n.pt",
                        help="Weights/config của model")
    parser.add_argument("-d", "--data",  type=str, required=True,
                        help="Đường dẫn tới file dataset.yaml")
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="Số epoch khi train")
    parser.add_argument("-i", "--imgsz", type=int, default=640,
                        help="Kích thước ảnh")
    return parser.parse_args()

def main():
    args = parse_args()

    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz
    }

    # Train (cfg dùng mặc định, overrides truyền config)
    trainer = DetectionTrainer(overrides=overrides)
    trainer.train()

    # Validate (cũng qua overrides)
    validator = DetectionValidator(overrides=overrides)
    validator.valid()

if __name__ == "__main__":
    main()
