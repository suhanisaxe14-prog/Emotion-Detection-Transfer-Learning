# 🚦 Traffic Sign Recognition on GTSRB: ANN vs Classical ML (HOG + SVM) vs CNN

Classifying German traffic signs into **43 classes** with three different approaches, trained and tested on the same data split so the comparison is fair.

| Model | Input | Test accuracy |
|---|---|---|
| ANN (fully connected) | raw pixels | **83.17 %** |
| Classical ML: HOG + colour histogram + SVM (RBF) | hand-crafted features | **89.64 %** |
| **CNN (final model)** | raw pixels, learned features | **96.01 %** |

---

## 📌 Objective

Traffic-sign recognition is a core perception task for driver-assistance systems and autonomous vehicles. A wrong or missed sign (a Stop read as a speed limit, say) is a safety problem, so the model has to be accurate across all classes, not just the common ones.

This project has two goals:

1. **Build an accurate traffic-sign classifier** that takes a 32×32 colour image and predicts one of 43 sign classes. This is the CNN.
2. **Find out why a CNN is the right tool** by solving the *same* problem with two other approaches and comparing them on identical data:
   - an **ANN** that sees raw pixels but has no notion of spatial structure;
   - a **classical ML pipeline** (hand-crafted HOG shape features + HSV colour features → SVM), the approach used before deep learning became common.

Everything is built only with fundamentals: ANN, CNN, and classical ML. There is no transfer learning or fine-tuning.

---

## 🗂️ Dataset

**GTSRB: German Traffic Sign Recognition Benchmark** (introduced for the IJCNN 2011 competition), downloaded from Kaggle via `kagglehub`:
[`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

| Property | Value |
|---|---|
| Classes | 43 (speed limits, prohibitions, warnings, mandatory signs, etc.) |
| Training images | 39,209 |
| Test images | 12,630 (labels in `Test.csv`) |
| Training "tracks" | 1,307 (each physical sign photographed in a sequence of frames) |
| Original image size | Varies; all images resized to **32 × 32 × 3** (RGB) |
| Class balance | **Imbalanced**: from 210 images (Speed limit 20 km/h) to 2,250 (Speed limit 50 km/h) |

Folder layout used by the notebook: `Train/<class_id>/`, `Test/`, `Train.csv`, `Test.csv`, `Meta/`.

### Important detail: avoiding data leakage
Training file names follow `<class>_<track>_<frame>.png`. Frames from the same track are **near-identical**, so a random split would put almost the same image in both train and validation and inflate the validation score. Instead I use **`GroupShuffleSplit`** to split **by track**:

- Train: **31,349** images
- Validation: **7,860** images (20 %, no track appears in both, checked with an `assert`)
- Test: **12,630** images (the official test set, never used for tuning)

---

## 🔬 Approach

### 1. Data preparation (shared by all three models)
- Load images in parallel (`ThreadPoolExecutor`), resize to 32×32 RGB, and cache to `gtsrb_data.npz` so reloading is fast.
- EDA: sample images, class-distribution plot, duplicate check, NaN/Inf checks.
- Normalise pixels to `[0, 1]`.
- Track-wise train/validation split (above).

### 2. Model A: ANN (baseline)
`Flatten → Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dense(43, softmax)`, about 825 K parameters, Adam, 10 epochs.
It flattens the image, so it throws away the 2-D layout of pixels. That is why it is the weakest model.

### 3. Model B: Classical ML (HOG + colour + SVM)
- **HOG** (Histogram of Oriented Gradients) on a CLAHE-equalised grayscale image captures **shape and edges** (1,764 features).
- **HSV colour histograms** (8 bins × 3 channels) capture **colour** (24 features).
- `StandardScaler` → **PCA** (keeps 95 % variance: 1,788 → **505** features) → **SVM with RBF kernel**, `class_weight="balanced"` to handle class imbalance.
- Hyper-parameters (`C`, `gamma`) tuned on the **track-wise validation set**: best **C = 10, gamma = scale × 0.5**.
- Extra experiments: linear vs RBF kernel, and a **feature ablation** (colour only / HOG only / both).

### 4. Model C: CNN (final model)
```
Input 32×32×3
→ Data augmentation (RandomRotation 0.08, RandomZoom 0.10, RandomTranslation 0.10)   [training only]
→ Conv2D(32)  → BatchNorm → MaxPool
→ Conv2D(64)  → BatchNorm → MaxPool
→ Conv2D(128) → BatchNorm → MaxPool
→ Flatten → Dense(128, ReLU) → Dropout(0.4) → Dense(43, softmax)
```
About 362 K parameters (less than half the ANN), Adam (lr 1e-3), up to 30 epochs with **EarlyStopping** on `val_loss` (`patience=5`, `restore_best_weights=True`).

**Why each piece is there**
| Component | Reason |
|---|---|
| Conv layers | Learn edges → shapes → sign parts, and share weights across the image |
| BatchNormalization | More stable, faster training |
| MaxPooling | Small shifts no longer matter; fewer parameters |
| Data augmentation | Signs appear at different angles, scales and positions; makes the model more robust |
| Dropout | Reduces overfitting in the dense layer |
| EarlyStopping | Stops training and restores the best weights |

---

## 📊 Results

### Model comparison (test set, 12,630 images)

| Model | Test accuracy | Notes |
|---|---|---|
| ANN (raw pixels) | 83.17 % | Ignores spatial structure |
| HOG + colour + SVM | 89.64 % | Val 90.57 %; 1,308 misclassified; slow prediction (108.6 s for 12,630 images) |
| **CNN** | **96.01 %** | Val 98.33 %; **504** wrong out of 12,630 |

**Takeaways**
- Hand-crafted features beat raw pixels by about 6.5 points, so *structure matters*.
- A CNN that learns its own features beats both, by about 6.4 points over the SVM and about 12.8 over the ANN, with **fewer parameters than the ANN**.
- **Feature ablation (SVM, validation):** colour alone 13.8 %, HOG alone 89.7 %, HOG + colour 90.0 %. Shape carries almost all the signal; colour gives a small boost.
- **RBF vs linear SVM:** 90.6 % vs 86.3 %. The classes are not linearly separable in HOG space.

### Where the CNN struggles (weakest test classes)
| Class | Test images | Accuracy |
|---|---|---|
| Pedestrians | 60 | 50.0 % |
| Beware of ice/snow | 150 | 56.7 % |
| End of no passing | 60 | 78.3 % |
| Double curve | 90 | 82.2 % |
| Children crossing | 150 | 84.0 % |

These are mostly **warning triangles with small inner pictograms** that become blurry at 32×32, and several of them have **few training images**.

---

## 🧪 Evaluation tools used
Accuracy on the held-out test set, per-class precision/recall/F1 (`classification_report`), confusion matrices, weakest-class table, gallery of misclassified images, and **top-3 predictions with confidence** for single images.

---

## 🛠️ Tech stack
**Python**, **Keras 3 / TensorFlow** (ANN, CNN), **scikit-learn** (SVM, PCA, scaler, metrics, `GroupShuffleSplit`), **scikit-image** (HOG, CLAHE, HSV), **NumPy**, **Pandas**, **Matplotlib**, **Pillow**, **joblib**, **tqdm**, **kagglehub**.

---

## 🚀 How to run

```bash
git clone https://github.com/suhanisaxe14-prog/<your-repo-name>.git
cd <your-repo-name>
pip install keras tensorflow scikit-learn scikit-image numpy pandas matplotlib pillow tqdm joblib kagglehub
jupyter notebook traffic_sign_recognition.ipynb
```

1. Run the cells top to bottom. The dataset is downloaded automatically through `kagglehub` (a Kaggle account/API access may be required).
2. The first run extracts and caches the data (`gtsrb_data.npz`). Later runs load from the cache.
3. **Part 1** is ANN and CNN. **Part 2** is HOG + SVM, plus the final three-way comparison.

> Note: HOG feature extraction (about 6 min) and SVM tuning (about 30 min for the 3×3 grid) are the slowest parts on CPU.

### Use the saved models
```python
import keras, numpy as np
from PIL import Image

model = keras.models.load_model("gtsrb_cnn.keras")
img = np.array(Image.open("sign.png").convert("RGB").resize((32, 32)), dtype="float32") / 255.0
probs = model.predict(img[None], verbose=0)[0]
print(np.argmax(probs), probs.max())
```
The SVM bundle `traffic_sign_svm.joblib` holds the scaler, PCA, SVM and class names.

---

## 📁 Repository structure
```
├── traffic_sign_recognition.ipynb   # full notebook: EDA → ANN → CNN → HOG+SVM → comparison
├── gtsrb_cnn.keras                  # trained CNN
├── traffic_sign_svm.joblib          # scaler + PCA + SVM + class names
└── README.md
```

---

## 🔭 Future improvements
- Train at a higher resolution (48×48 or 64×64) so small pictograms stay readable.
- Rebalance the weak classes (more augmentation or class weights for the CNN).
- Try deeper CNNs or learning-rate scheduling, and tune the augmentation.
- Later: transfer learning and fine-tuning, once covered.
- Deploy a small **Streamlit** app for image upload and top-3 predictions.

---

## 👩‍💻 Author
**Suhani Saxena (Tisha)**: B.Tech (AI & Data Science), Mahakal Institute of Technology, Ujjain (RGPV)
GitHub: [@suhanisaxe14-prog](https://github.com/suhanisaxe14-prog)

## 🙏 Acknowledgements
- J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, *"The German Traffic Sign Recognition Benchmark: A multi-class classification competition"*, IJCNN 2011.
- Kaggle dataset mirror by `meowmeowmeowmeowmeow`.
