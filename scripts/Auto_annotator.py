import os
import cv2

# BASE DIRECTORY SETUP
base_path = r"C:\Users\hp\throat_cancer_dataset"
image_folders = ["benign", "malignant"]
annot_base = os.path.join(base_path, "annotations")

# Create annotations folder if not exists
for label in image_folders:
    os.makedirs(os.path.join(annot_base, label), exist_ok=True)

# FUNCTION TO GENERATE DUMMY BBOX (whole image center area)
def generate_dummy_bbox(image):
    h, w = image.shape[:2]
    x1 = int(w * 0.25)
    y1 = int(h * 0.25)
    x2 = int(w * 0.75)
    y2 = int(h * 0.75)
    return x1, y1, x2, y2

# MAIN PIPELINE
for label in image_folders:
    img_folder = os.path.join(base_path, label)
    annot_folder = os.path.join(annot_base, label)

    for file in os.listdir(img_folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(img_folder, file)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Failed to load {img_path}")
                continue

            # Create dummy bbox
            x1, y1, x2, y2 = generate_dummy_bbox(image)

            # Save annotation (YOLO-style: class x_center y_center width height)
            h, w = image.shape[:2]
            class_id = 0 if label == "benign" else 1
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            annot_file = os.path.splitext(file)[0] + ".txt"
            annot_path = os.path.join(annot_folder, annot_file)

            with open(annot_path, "w") as f:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            print(f"Saved annotation: {annot_path}")
