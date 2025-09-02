# 🧠 28-Day Computer Vision with PyTorch – Project Roadmap

This repo documents my **28-day deep dive into Computer Vision using PyTorch**.
The goal is to strengthen my **understanding of CV fundamentals, transfer learning, real-time detection, and medical/agriculture applications**, while also experimenting with advanced techniques like segmentation, embeddings, and pose estimation.

---

## 📌 Challenge Rules

* 🔹 1 project per day → **28 projects in 28 days**.
* 🔹 Only **PyTorch** for modeling.
* 🔹 Use **publicly available datasets** (linked below).
* 🔹 Cover a wide range of tasks → classification, detection, segmentation, pose estimation, image retrieval, etc.
* 🔹 Deliverables per project: **Jupyter Notebook + README + results (curves, confusion matrix, predictions)**.

---

## 📆 Daily Plan

### **Day 1 – MNIST Handwritten Digit Classifier**

* Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)
* Task: Train a CNN → classify 0–9 digits.
* Goal: Learn training loop, optimizer, loss.

---

### **Day 2 – Fashion-MNIST Classifier**

* Dataset: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
* Task: Classify T-shirts, shoes, dresses.
* Goal: Understand generalization beyond digits.

---

### **Day 3 – Fruit Classification**

* Dataset: [Fruits 360](https://www.kaggle.com/moltean/fruits)
* Task: Classify apples, bananas, oranges, etc.
* Goal: Multi-class CNN, data augmentation.

---

### **Day 4 – Flower Recognition with Transfer Learning**

* Dataset: [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
* Task: Classify flower species.
* Goal: Apply ResNet transfer learning.

---

### **Day 5 – Real-Time Face Detection**

* Dataset: [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) or webcam.
* Task: Detect faces in images/webcam.
* Goal: Detection pipeline (bounding boxes).

---

### **Day 6 – Image Colorization (Autoencoder)**

* Dataset: [COCO subset](https://cocodataset.org/#download) (grayscale version).
* Task: Train autoencoder to colorize grayscale.
* Goal: Encoder–decoder basics.

---

### **Day 7 – OCR (Text Extraction)**

* Dataset: [IIIT 5K Word](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)
* Task: Extract text using CRNN/EasyOCR.
* Goal: Sequence modeling + vision.

---

### **Day 8 – Image Search Engine**

* Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* Task: Extract embeddings → search similar images.
* Goal: Metric learning & embeddings.

---

### **Day 9 – Plant Disease Detection**

* Dataset: [PlantVillage](https://www.kaggle.com/emmarex/plantdisease)
* Task: Healthy vs diseased leaves.
* Goal: Agriculture + classification.

---

### **Day 10 – Crop Type Recognition (Satellite)**

* Dataset: [EuroSAT](https://github.com/phelber/eurosat)
* Task: Classify land use (wheat, maize, rice, etc.).
* Goal: Remote sensing imagery.

---

### **Day 11 – Poultry Disease Detection**

* Dataset: [Chicken Dataset](https://www.kaggle.com/search?q=chicken+disease)
* Task: Classify chicken health.
* Goal: Fine-tuning on small dataset.

---

### **Day 12 – Pneumonia Detection (Chest X-rays)**

* Dataset: [Chest X-Ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* Task: Normal vs pneumonia.
* Goal: Medical imaging basics.

---

### **Day 13 – Skin Cancer Classification**

* Dataset: [ISIC 2019](https://challenge.isic-archive.com/data/)
* Task: Melanoma vs benign.
* Goal: Transfer learning for health.

---

### **Day 14 – Retina Disease Detection**

* Dataset: [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
* Task: Detect diabetic retinopathy.
* Goal: Deeper medical CV.

---

### **Day 15 – Fruit Counting & Detection**

* Dataset: [Fruit Detection](https://www.kaggle.com/mbkinaci/fruit-images-for-object-detection)
* Task: Detect/count apples, mangoes, bananas.
* Goal: Object detection workflow.

---

### **Day 16 – Real-Time Object Detection (YOLOv8)**

* Dataset: [COCO subset](https://cocodataset.org/#download) or custom farm dataset.
* Task: Detect tractors, tools, people.
* Goal: Real-time video inference.

---

### **Day 17 – Weed vs Crop Segmentation**

* Dataset: [Weed Detection](https://www.kaggle.com/competitions/weed-detection/data)
* Task: Segment weeds from crops.
* Goal: U-Net for semantic segmentation.

---

### **Day 18 – Brain Tumor Segmentation**

* Dataset: [BRATS](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
* Task: Segment tumor regions in MRI scans.
* Goal: Medical segmentation.

---

### **Day 19 – Human Pose Estimation**

* Dataset: [MPII Human Pose](http://human-pose.mpi-inf.mpg.de/)
* Task: Keypoints for human skeleton.
* Goal: Regression outputs in CV.

---

### **Day 20 – Cattle Pose Estimation**

* Dataset: [Cow Pose Dataset](https://www.kaggle.com/datasets/andrewmvd/cows)
* Task: Detect standing, lying, walking.
* Goal: Agriculture + pose estimation.

---

### **Day 21 – Neural Style Transfer (AI Art)**

* Dataset: Any images (style + content).
* Task: Apply Van Gogh style to photo.
* Goal: Style transfer with CNNs.

---

### **Day 22 – Cartoonization of Images**

* Dataset: [Cartoon Dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-face-dataset)
* Task: Real → cartoon-like.
* Goal: Image-to-image translation.

---

### **Day 23 – Real-Time Emotion Detection**

* Dataset: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
* Task: Detect happy, sad, angry, etc.
* Goal: Multi-class + webcam inference.

---

### **Day 24 – Livestock Health Classification**

* Dataset: [Cow Body Condition](https://data.mendeley.com/datasets/2rnnz4fshx/1)
* Task: Healthy vs unhealthy cows.
* Goal: Agriculture + detection.

---

### **Day 25 – Medical X-Ray Dashboard**

* Dataset: \[Chest X-Ray Pneumonia]\(reuse Day 12 dataset).
* Task: Build model + visualization dashboard.
* Goal: Applied system workflow.

---

### **Day 26 – Image Embedding Search Engine (Advanced)**

* Dataset: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
* Task: Search most similar images.
* Goal: Embedding + FAISS search.

---

### **Day 27 – Retinal Blood Vessel Segmentation**

* Dataset: [DRIVE Dataset](https://drive.grand-challenge.org/)
* Task: Segment blood vessels in retina.
* Goal: Fine-grained medical segmentation.

---

### **Day 28 – Multi-Task Farm Monitoring System**

* Dataset: Combination of PlantVillage + Fruits + Cow dataset.
* Task: Detect fruits, track cows, classify leaves.
* Goal: Multi-task pipeline (YOLO + segmentation).

---

## ⚡ Deliverables

* 📁 `projects/dayX_project_name/` → notebook + code.
* 📑 `README.md` (this file) → progress tracker.
* 📊 Results → accuracy/loss curves, confusion matrix, predictions, screenshots.