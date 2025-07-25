# Endoscopic Cancer Classification using YOLOv8n

Classifying endoscopic images into **benign** and **malignant** using **YOLOv8n**.

## 📌 Dataset
- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10590430/
- Classes: `benign`, `malignant`

## 🧱 Project Structure
endoscopic-cancer-detection-yolov8/
│
├── dataset/                      
│   ├── benign/
│   └── malignant/
│
├── annotations/
│   ├── benign/
│   └── malignant/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
├── Train2/                       
│   ├── results.png
│   ├── results.csv
│   ├── labels_correlogram.jpg
│   ├── confusion_matrix.png
│   └── weights/                  
│
├── scripts/
│   ├── annotate_images.py
│   ├── split_dataset.py
│   └── train_yolov8.py
│
├── yolov8_config/
│   └── data.yaml
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE   (MIT)

## 🛠 Pipeline
1. **Dataset prep**  
dataset/
├── benign/
└── malignant/

2. **Annotation (Phase 2)**  
- Wrote a Python script (`scripts/Auto_annotator.py`) using `os` & `cv2`  
- Generated labels into `annotations/benign` and `annotations/malignant`

3. **Split (train/val)**  
- `images/` and `labels/` both have `train/` and `val/` folders

4. **Training**
- **Model:** YOLOv8n  
- **Epochs:** 5  
- **Hardware:** CPU (took ~8.15 hours)  
- **Accuracy:** ~81.19%  

## 📊 Results (in `/Train2`)
- `results.png`
- `results.csv`
- `labels_correlogram.jpg`
- (confusion matrix, PR/F1 curves, etc.)

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# train (example)
yolo task=detect mode=train model=yolov8n.pt data=yolov8_config/data.yaml epochs=5 imgsz=640

📧 Contact
Dhruv Singhal
Email: dsinghal265@gmail.com
LinkedIn: <your-link>
