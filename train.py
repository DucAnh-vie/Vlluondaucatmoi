import argparse
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

def parse_args():
    parser = argparse.ArgumentParser(description="Train và Validate YOLO model")
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Đường dẫn tới file dataset.yaml"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Số epoch khi train (mặc định: 100)"
    )
    parser.add_argument(
        "--imgsz", "-i",
        type=int,
        default=640,
        help="Kích thước ảnh (mặc định: 640)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Train
    trainer = DetectionTrainer(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz
    )
    trainer.train()

    # Validate
    validator = DetectionValidator(
        data=args.data,
        imgsz=args.imgsz
    )
    validator.valid()

if __name__ == "__main__":
    main()
