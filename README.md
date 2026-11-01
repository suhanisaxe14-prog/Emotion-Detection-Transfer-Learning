# 🎭 EmotionLens

### 👁️ Real-Time Facial Emotion Recognition with MobileNetV2 Transfer Learning & OpenCV

**A computer-vision application that finds faces in a live webcam feed and reads their expressions, using a fine-tuned deep-learning model.** 🚀

---

## 🎯 About the Project

EmotionLens classifies facial expressions into **7 emotions**: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad` and `surprise`. It works on image folders for training and on a **live webcam stream** for real-time use.

Instead of training a large network from scratch, the project borrows the visual knowledge of **MobileNetV2** (pre-trained on ImageNet) and teaches it to read faces in two stages: first a new classifier head, then a careful fine-tuning of the backbone. The result is lightweight enough for real-time use on an ordinary laptop.

The aim is a complete workflow: **data → model → evaluation → saved model → working live demo**, not just a score in a notebook.

## ✨ Key Features

- 🎭 **7-class emotion recognition** on the FER-2013 dataset
- 🧠 **Transfer learning** with pre-trained MobileNetV2
- 🔧 **Two-phase training**: feature extraction, then fine-tuning the last 30 layers
- 🔄 **Built-in augmentation** (flip, rotation, zoom) for better generalisation
- 👤 **Automatic face detection** with OpenCV Haar Cascades
- 📹 **Live webcam inference** with colour-coded boxes and confidence scores
- 📊 **Live probability bars** for all 7 emotions on screen
- 📸 **Screenshot hotkey** (`s`) and clean quit (`q`)
- 💾 **Model and label export** (`emotion_model.keras`, `class_names.json`)

## 🛠️ Technologies Used

| Technology | Role in the project |
|---|---|
| 🐍 Python | Core language |
| 🧠 TensorFlow & Keras | Model building, training, fine-tuning |
| 📱 MobileNetV2 | Pre-trained backbone for feature extraction |
| 👁️ OpenCV | Face detection and real-time video |
| 🔢 NumPy | Array and probability handling |
| 📊 Matplotlib | Training curves and visual results |
| 📓 Jupyter Notebook | Development and experimentation |

## 🗂️ Dataset

**FER-2013** (Facial Expression Recognition, Kaggle)

| | |
|---|---|
| Training images | 28,709 |
| Test images | 7,178 |
| Classes | 7 emotions |
| Preprocessing | Resized to 96 × 96, converted to RGB |
| Challenge | Imbalanced classes (many more `happy` than `disgust`) and noisy real-world labels |

## ⚙️ How It Works

📸 **Webcam frame**
⬇️
👤 **Haar Cascade finds the face**
⬇️
✂️ **Crop → grayscale → resize 96×96 → RGB**
⬇️
🧠 **MobileNetV2 extracts facial features**
⬇️
🎯 **Softmax gives 7 emotion probabilities**
⬇️
🖥️ **Box, label, confidence and live bars drawn on the video**

## 🧠 Model Architecture

```
Input 96×96×3
 → Augmentation (RandomFlip · RandomRotation 0.1 · RandomZoom 0.1)
 → Rescaling to [-1, 1]
 → MobileNetV2 (ImageNet weights)
 → GlobalAveragePooling2D
 → Dropout (0.3)
 → Dense (7, softmax)
```

| Detail | Value |
|---|---|
| Total parameters | about 2.27 M |
| Trainable in Phase A | 8,967 (head only) |
| Loss | Categorical cross-entropy |
| Optimizer | Adam |

### 🔁 Training strategy

| Phase | What learns | Learning rate | Epochs |
|---|---|---|---|
| **A: Feature extraction** | New classifier head only | 1e-3 | 15 |
| **B: Fine-tuning** | Last 30 backbone layers + head | 1e-5 | 15 |

A very small learning rate in Phase B adapts the pre-trained features to faces without wiping out what they already know.

## 📈 Model Evaluation

| Stage | Test accuracy |
|---|---|
| Phase A: frozen backbone | about 43 % |
| **Phase B: fine-tuned** | **51.9 %** (loss 1.275) |

- ✅ Fine-tuning improved accuracy by roughly **9 percentage points**.
- ✅ Random guessing over 7 classes gives about 14 %, so the model has learned real facial cues.
- ⚠️ FER-2013 is a **hard, noisy benchmark** (low resolution, ambiguous expressions, imperfect labels).
- ⚠️ The dataset is **imbalanced**, so overall accuracy hides weaker results on rare emotions.
- ⚠️ The test folder is also used as validation during training, so 51.9 % is a development benchmark. A separate validation split is the next improvement.

<!-- After running evaluation_cells.py, paste here: per-class precision/recall/F1 table, confusion matrix image, most confused pairs -->



## 👩‍💻 Connect With Me

**Suhani Saxena**
💡 B.Tech (AI & Data Science) · Aspiring AI/ML Engineer

- 🐙 **GitHub:** [suhanisaxe14-prog](https://github.com/suhanisaxe14-prog)
- 💼 **LinkedIn:** https://www.linkedin.com/in/suhani-saxena-735472311/
- 📧 **Email:** suhanisaxena1603@gmail.com

---

### 🚀 Turning pre-trained intelligence into real-time understanding

