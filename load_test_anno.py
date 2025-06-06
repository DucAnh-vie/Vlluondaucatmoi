import os
from PIL import Image

def load_widerface_test(image_root, annot_file):
    """
    Đọc ảnh và tên file từ WIDER FACE test set.
    Params:
        image_root: thư mục chứa ảnh (ví dụ: /content/Face-detection/WIDER_test/WIDER_test/images)
        annot_file: file wider_face_test_filelist.txt (ví dụ: /content/Face-detection/wider_face_split/wider_face_test_filelist.txt)
    Returns:
        Danh sách dict với đường dẫn ảnh và object PIL image.
    """
    samples = []
    with open(annot_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.endswith(".jpg"):
                continue
            img_path = os.path.join(image_root, line)
            try:
                img = Image.open(img_path).convert("RGB")
                samples.append({"file_name": img_path, "image": img})
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return samples


if __name__ == "__main__":
    image_dir = "/content/Face-detection/WIDER_test/images"
    annot_path = "/content/Face-detection/wider_face_split/wider_face_test_filelist.txt"
    
    test_samples = load_widerface_test(image_dir, annot_path)
    print(f"Loaded {len(test_samples)} test images.")
    
    # Hiển thị một ảnh nếu muốn kiểm tra
    test_samples[0]["image"].show()
