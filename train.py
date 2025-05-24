import argparse
from ultralytics_custom.models.yolo.detect import DetectionTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train và Validate YOLO model")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Đường dẫn tới file model.yaml TÙY CHỈNH")
    parser.add_argument("-d", "--data",  type=str, required=True,
                        help="Đường dẫn tới file dataset.yaml")
    parser.add_argument("-e", "--epochs", type=int, default=50,
                        help="Số epoch khi train")
    parser.add_argument("-i", "--imgsz", type=int, default=640,
                        help="Kích thước ảnh")
    return parser.parse_args()

def main():
    args = parse_args()

    overrides = {
        "model": args.model,        # ← file .yaml tùy chỉnh
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz
    }

    trainer = DetectionTrainer(overrides=overrides)
    trainer.train()

if __name__ == "__main__":
    main()
