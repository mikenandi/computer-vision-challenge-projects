# 🧠 14-Day Computer Vision with PyTorch – Project Roadmap

This repo documents my **14-day deep dive into Computer Vision using PyTorch**.
The goal is to strengthen my **understanding of CV fundamentals, transfer learning, real-time detection, and medical/agriculture applications**, while also experimenting with advanced techniques like segmentation, embeddings, and pose estimation.

---

## 📌 Challenge Rules

- 🔹 28 Projects in 14 Days (2 projects per day on average).
- 🔹 Only PyTorch for modeling.
- 🔹 Use publicly available datasets (listed below).
- 🔹 Focus on breadth (cover classification, detection, segmentation, pose, search).
- 🔹 Deliver README + Jupyter Notebook/code per project.

---

## 📆 Daily Plan

### **Day 1 – Basics of Classification**

1. **Handwritten Digit Classifier (MNIST)**

   - Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)
   - Task: Train a CNN → classify 0–9 digits.
   - Goal: Learn training loop, optimizer, loss.

2. **Fashion-MNIST Classifier (Clothes Recognition)**

   - Dataset: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
   - Task: Classify T-shirts, shoes, dresses.
   - Goal: Understand generalization beyond digits.

---

### **Day 2 – Simple Image Recognition**

3. **Fruit Classification**

   - Dataset: [Fruits 360](https://www.kaggle.com/moltean/fruits)
   - Task: Classify apples, bananas, oranges, etc.
   - Goal: Multi-class CNN, data augmentation.

4. **Flower Recognition**

   - Dataset: [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
   - Task: Classify flower species.
   - Goal: Transfer learning with pretrained ResNet.

---

### **Day 3 – Face & Color Models**

5. **Real-Time Face Detection**

   - Dataset: [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) or webcam feed.
   - Task: Detect faces in images/webcam.
   - Goal: Detection pipeline (bounding boxes).

6. **Image Colorization (Autoencoder)**

   - Dataset: [COCO dataset small subset](https://cocodataset.org/#download) (use grayscale).
   - Task: Train autoencoder to add color to grayscale.
   - Goal: Encoder–decoder architecture basics.

---

### **Day 4 – Text & OCR**

7. **Text Extraction (OCR)**

   - Dataset: [IIIT 5K Word Dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)
   - Task: Extract text from images using CRNN/EasyOCR.
   - Goal: Sequence modeling with CV.

8. **Image Search Engine (Reverse Search)**

   - Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
   - Task: Extract embeddings → search for similar images.
   - Goal: Learn metric learning & embeddings.

---

### **Day 5 – Agriculture Applications**

9. **Plant Disease Detection**

   - Dataset: [PlantVillage](https://www.kaggle.com/emmarex/plantdisease)
   - Task: Healthy vs diseased leaves.
   - Goal: Agriculture + classification.

10. **Crop Type Recognition (Satellite)**

- Dataset: [EuroSAT](https://github.com/phelber/eurosat)
- Task: Classify land use (wheat, maize, rice, etc.).
- Goal: Work with aerial imagery.

---

### **Day 6 – Agriculture & Health**

11. **Poultry Disease Detection**

- Dataset: [Custom/Kaggle Chicken Dataset](https://www.kaggle.com/search?q=chicken+disease)
- Task: Classify chicken health from images.
- Goal: Small dataset fine-tuning.

12. **Pneumonia Detection (Chest X-rays)**

- Dataset: [Chest X-Ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- Task: Predict normal vs pneumonia.
- Goal: Medical imaging basics.

---

### **Day 7 – Health Imaging**

13. **Skin Cancer Classification**

- Dataset: [ISIC 2019](https://challenge.isic-archive.com/data/)
- Task: Classify melanoma vs benign.
- Goal: Transfer learning in health.

14. **Retina Disease Detection**

- Dataset: [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
- Task: Detect diabetic retinopathy.
- Goal: Deeper medical domain CV.

---

### **Day 8 – Object Detection**

15. **Fruit Counting & Detection**

- Dataset: [Fruits Dataset](https://www.kaggle.com/mbkinaci/fruit-images-for-object-detection)
- Task: Detect/count apples, mangoes, bananas.
- Goal: Object detection workflow.

16. **Real-Time Object Detection (YOLOv8)**

- Dataset: [COCO subset](https://cocodataset.org/#download) / custom farm tools dataset.
- Task: Detect tractors, tools, people.
- Goal: Real-time video inference.

---

### **Day 9 – Segmentation**

17. **Weed vs Crop Segmentation**

- Dataset: [Weed Detection Dataset](https://www.kaggle.com/competitions/weed-detection/data)
- Task: Segment weeds from crops.
- Goal: U-Net for semantic segmentation.

18. **Brain Tumor Segmentation**

- Dataset: [BRATS Dataset](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
- Task: Segment tumor regions in MRI scans.
- Goal: Medical segmentation.

---

### **Day 10 – Pose Estimation**

19. **Human Pose Estimation**

- Dataset: [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/)
- Task: Keypoints for human skeleton.
- Goal: Regression outputs in CV.

20. **Cattle Pose Estimation (Farm Monitoring)**

- Dataset: [Custom cow dataset](https://www.kaggle.com/datasets/andrewmvd/cows) or adapt MPII.
- Task: Detect standing vs lying vs walking postures.
- Goal: Pose estimation in agriculture.

---

### **Day 11 – Creative CV**

21. **AI Art Style Transfer**

- Dataset: Any images (style + content).
- Task: Apply Van Gogh style to photo.
- Goal: Neural style transfer.

22. **Cartoonization of Images**

- Dataset: [Cartoon Dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-face-dataset)
- Task: Convert real images to cartoon-like.
- Goal: Image-to-image translation.

---

### **Day 12 – Emotion & Behavior**

23. **Real-Time Emotion Detection**

- Dataset: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Task: Detect happy, sad, angry, etc.
- Goal: Multi-class classification + webcam.

24. **Livestock Health Classification**

- Dataset: [Cow Body Condition Dataset](https://data.mendeley.com/datasets/2rnnz4fshx/1)
- Task: Detect healthy vs unhealthy cows.
- Goal: Agriculture + detection.

---

### **Day 13 – Advanced Systems**

25. **Medical X-Ray Dashboard (Doctor View)**

- Dataset: Chest X-Ray Pneumonia (reuse #12).
- Task: Build model + add confidence plots.
- Goal: Full applied system workflow.

26. **Image Embedding Search Engine**

- Dataset: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- Task: Search most similar images.
- Goal: Embedding + FAISS.

---

### **Day 14 – Final Expert Projects**

27. **Semantic Segmentation for Retinal Blood Vessels**

- Dataset: [DRIVE Dataset](https://drive.grand-challenge.org/)
- Task: Segment blood vessels in retina.
- Goal: Fine-grained medical segmentation.

28. **Multi-Task Farm Monitoring System**

- Dataset: Combination of Fruit detection + Cows dataset + PlantVillage.
- Task: Detect fruits, track cows, classify leaves.
- Goal: Multi-task YOLO + segmentation pipeline.

---

## ⚡ Deliverables

- 📁 `projects/dayX_project_name/` with code + notebook.
- 📑 `README.md` (this file) → progress tracker.
- 📊 Screenshots, loss/accuracy curves, sample predictions.
