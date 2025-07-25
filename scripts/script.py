import os
import shutil
from sklearn.model_selection import train_test_split

# Update paths
base_dir = r"C:\Users\hp\throat_cancer_dataset"
benign_img_dir = os.path.join(base_dir, "benign")
maligant_img_dir = os.path.join(base_dir, "malignant")
annotations_benign_dir = os.path.join(base_dir, "annotations", "benign")
annotations_maligant_dir = os.path.join(base_dir, "annotations", "malignant")

# Output folders
output_dir = os.path.join(base_dir, "processed")
os.makedirs(output_dir, exist_ok=True)

images_output_dir = os.path.join(output_dir, "images")
labels_output_dir = os.path.join(output_dir, "labels")

for folder in [images_output_dir, labels_output_dir]:
    os.makedirs(os.path.join(folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(folder, "val"), exist_ok=True)

def collect_files(img_dir, ann_dir):
    data = []
    for file in os.listdir(img_dir):
        file_base, ext = os.path.splitext(file)
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        
        img_path = os.path.join(img_dir, file)
        txt_filename = file_base + ".txt"
        txt_path = os.path.join(ann_dir, txt_filename)
        
        if os.path.exists(txt_path):
            data.append((img_path, txt_path))
        else:
            print(f"[⚠️] Missing label for: {img_path}")
    return data

# Collect all valid (image, label) pairs
benign_pairs = collect_files(benign_img_dir, annotations_benign_dir)
maligant_pairs = collect_files(maligant_img_dir, annotations_maligant_dir)
all_pairs = benign_pairs + maligant_pairs

print(f"✅ Total valid image-label pairs: {len(all_pairs)}")

if len(all_pairs) < 2:
    raise ValueError("❌ Not enough data to split. Add more image-label pairs.")

# Train-val split
train_data, val_data = train_test_split(all_pairs, test_size=0.2, random_state=42)

def copy_pairs(pairs, split):
    for img_path, txt_path in pairs:
        img_name = os.path.basename(img_path)
        txt_name = os.path.basename(txt_path)
        
        shutil.copy(img_path, os.path.join(images_output_dir, split, img_name))
        shutil.copy(txt_path, os.path.join(labels_output_dir, split, txt_name))

copy_pairs(train_data, "train")
copy_pairs(val_data, "val")

print("✅ Dataset successfully prepared in 'processed' folder.")
