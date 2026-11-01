# 🎭 EmotionLens: Real-Time Facial Emotion Recognition with Transfer Learning

**EmotionLens** is an end-to-end computer-vision project that detects a face from a live webcam feed and classifies the person's emotion in real time into **7 categories** (angry, disgust, fear, happy, neutral, sad, surprise). It uses **MobileNetV2** (ImageNet pre-trained) with a two-phase **transfer learning + fine-tuning** pipeline, trained on **FER-2013**.

> **From training to a live demo**: the notebook covers the whole workflow: data pipeline → augmentation → transfer learning → fine-tuning → evaluation → model export → **real-time webcam inference with live probability bars**.

| | |
|---|---|
| **Task** | Multi-class image classification (7 emotions) |
| **Base model** | MobileNetV2 (ImageNet weights), a lightweight architecture suited to edge and real-time use |
| **Dataset** | FER-2013, 28,709 train / 7,178 test images |
| **Training strategy** | Phase A: feature extraction → Phase B: fine-tuning the last 30 layers |
| **Deployment** | Live webcam inference with OpenCV (face detection + emotion overlay) |
| **Final accuracy** | **51.9 %** on the 7,178-image test set (see [honest evaluation](#-results--honest-evaluation)) |

---

## 🎯 Objective

Understanding emotion from faces is useful wherever software has to respond to people. Typical applications:

- 📞 **Customer experience**: gauging sentiment in video support and service
- 🎓 **EdTech**: engagement and attention feedback in online classes
- 🧠 **Wellbeing and HCI**: emotion-aware apps and adaptive interfaces
- 🚗 **Automotive**: driver-state awareness
- 🎮 **Gaming and interactive media**: reactive experiences
- 🛍️ **Market research**: reaction analysis (with consent)

**What this project aims to do**
1. Build an accurate, *lightweight* emotion classifier using transfer learning instead of training a huge network from scratch.
2. Show the full path from a trained model to a **working real-time application**, not just a notebook score.
3. Evaluate the result honestly, including class imbalance and the limits of the dataset.

---

## 🗂️ Dataset: FER-2013

[FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) (Facial Expression Recognition 2013): small face crops labelled with one of seven emotions.

| Property | Value |
|---|---|
| Train images | 28,709 |
| Test images | 7,178 |
| Classes | 7: `angry, disgust, fear, happy, neutral, sad, surprise` |
| Source format | Small grayscale face images in class folders (`train/<emotion>/`, `test/<emotion>/`) |
| Used as | Resized to **96 × 96**, converted to **RGB** (MobileNetV2 expects 3 channels) |
| Known issue | **Class imbalance** (far more `happy` than `disgust`) and noisy, in-the-wild labels |

---

## 🧠 Approach

### 1. Data pipeline
- `keras.utils.image_dataset_from_directory` loads the folders into batched `tf.data` datasets (batch size 64, 96×96, RGB, one-hot labels).
- Class names are saved to `class_names.json` so the **live demo uses exactly the same label order as training**.

### 2. Model architecture
```
Input (96×96×3)
→ Data augmentation (RandomFlip horizontal · RandomRotation 0.1 · RandomZoom 0.1)
→ Rescaling to [-1, 1]  (the preprocessing MobileNetV2 expects)
→ MobileNetV2 backbone (ImageNet weights, include_top=False)
→ GlobalAveragePooling2D
→ Dropout(0.3)
→ Dense(7, softmax)
```
Total about **2.27 M parameters**; only **8,967** are trained in phase A.

### 3. Two-phase transfer learning
| Phase | What is trained | Learning rate | Epochs |
|---|---|---|---|
| **A: Feature extraction** | Only the new classifier head (backbone frozen) | 1e-3 | 15 |
| **B: Fine-tuning** | Last **30 layers** of MobileNetV2 + head (early layers stay frozen) | 1e-5 | 15 |

The tiny fine-tuning learning rate stops the pre-trained features from being destroyed while adapting them to faces.

### 4. Real-time inference (live demo)
A four-step loop runs on every webcam frame:
1. **Detect** faces with OpenCV's Haar Cascade (`haarcascade_frontalface_default.xml`)
2. **Crop and preprocess** the face the same way as the training images (grayscale → resize 96×96 → RGB)
3. **Predict** the emotion with the trained model
4. **Render** a colour-coded bounding box, the emotion label with confidence, and **live probability bars for all 7 emotions**

Controls: `q` quit · `s` save a screenshot.

---

## 📊 Results & honest evaluation

| Stage | Test accuracy |
|---|---|
| Phase A: frozen backbone | about 43 % |
| **Phase B: after fine-tuning** | **51.9 %** (loss 1.275) |

**Fine-tuning gave about 9 points** over the frozen-backbone result (about 43 % → 51.9 %), which shows that adapting the pre-trained features to faces matters.

**How to read the number**
- Seven-class random guessing is about 14 %, so the model has learned real signal.
- FER-2013 is a **hard, noisy benchmark**: low-resolution images, ambiguous expressions and imperfect labels cap what simple pipelines reach.
- Accuracy hides weak spots: the dataset is **imbalanced**, so rare classes such as `disgust` and `fear` are harder than `happy`.
- The `test` folder doubles as the validation set in this notebook, so the score is a **development benchmark**, not a final blind test.

> This is a **working baseline with a clear improvement path**, not a production-grade classifier. See the roadmap below.

---

## 🛠️ Tech stack
**Python** · **TensorFlow / Keras** (MobileNetV2, transfer learning) · **OpenCV** (face detection, live video) · **NumPy** · **Matplotlib**

---

## 🚀 Getting started

### 1. Clone and install
```bash
git clone https://github.com/suhanisaxe14-prog/Emotion-Detection-Transfer-Learning.git
cd Emotion-Detection-Transfer-Learning
pip install tensorflow "opencv-python<5" matplotlib numpy
```
> Use `opencv-python<5`: version 5 does not include `CascadeClassifier`, which the webcam demo needs.

### 2. Get the data
Download FER-2013 from Kaggle and extract it so the folder looks like:
```
archive (2)/
├── train/  angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
└── test/   angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
```
Set `DATA_FOLDER` in the settings cell if your folder is named differently.

### 3. Run
Open `Emotion_Detection_transfer_learning.ipynb` and run the cells in order. This trains the model, saves `emotion_model.keras` and `class_names.json`, then launches the webcam demo (needs a webcam; click the video window and press `q` to quit).

---

## 📁 Repository structure
```
├── Emotion_Detection_transfer_learning.ipynb   # data → model → training → evaluation → live demo
├── emotion_model.keras                         # trained model (generated by the notebook)
├── class_names.json                            # label order shared by training and live demo
└── README.md
```

---

## 🔭 Roadmap (planned improvements)
- [ ] **Handle class imbalance**: class weights or oversampling for rare emotions
- [ ] **Per-class metrics**: confusion matrix, precision/recall/F1 for each emotion
- [ ] **Separate validation split** so the test set stays untouched
- [ ] **Stronger backbones and higher resolution** (e.g. EfficientNet, 128×128+), plus longer fine-tuning with LR scheduling
- [ ] **Temporal smoothing** of predictions across frames for steadier live output
- [ ] **Better face detection** (e.g. DNN/MediaPipe-based) than Haar Cascade
- [ ] **Web app**: Streamlit or Flask demo with image upload and webcam
- [ ] **Export to TFLite** for mobile and edge deployment

---

## ⚠️ Ethical note
Facial emotion recognition is an **approximation** based on facial expression. It does not read a person's true inner state, and accuracy varies across people and conditions. Use it only with **informed consent**, and not for high-stakes decisions such as hiring, grading or surveillance.

---

## 👩‍💻 Author
**Suhani Saxena (Tisha)**: B.Tech (AI & Data Science), Mahakal Institute of Technology, Ujjain (RGPV)
GitHub: [@suhanisaxe14-prog](https://github.com/suhanisaxe14-prog)

## 🙏 Acknowledgements
- FER-2013 dataset (Kaggle) · MobileNetV2 (Sandler et al., 2018) · TensorFlow/Keras and OpenCV communities.
