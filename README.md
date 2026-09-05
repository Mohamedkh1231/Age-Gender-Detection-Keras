# 🧑‍🦳👨 Age and Gender Detection using CNN & Keras 3

An end-to-end Computer Vision application that detects faces in real-time or from images, predicting both **Age** (Regression) and **Gender** (Binary Classification) simultaneously using a multi-output Convolutional Neural Network built with **Keras 3** and **TensorFlow**.

---

## 🌟 Key Features

- **Multi-Output CNN**: Single backbone network predicting Age and Gender simultaneously.
- **Face Detection Integration**: Uses OpenCV's Haar Cascade classifier for real-time face localization.
- **Interactive App**: Run prediction scripts directly using the included `APP.py`.
- **Built-in Data Augmentation**: Includes horizontal flips, rotations, zooms, and translations.

---

## 📐 Model Architecture

- **Input Shape**: `(128, 128, 3)`
- **Data Augmentation**:
  - `RandomFlip("horizontal")`
  - `RandomRotation(factor=0.08)`
  - `RandomZoom(height_factor=0.1)`
  - `RandomTranslation(height_factor=0.08, width_factor=0.08)`
- **Feature Extractor**: 
  - 4x `Conv2D` blocks (32, 64, 128, 256 filters with ReLU)
  - `BatchNormalization` + `MaxPooling2D`
  - `GlobalAveragePooling2D`
- **Output Heads**:
  - **Gender Output**: `Dense(1, activation='sigmoid')` (Loss: `binary_crossentropy`)
  - **Age Output**: `Dense(1, activation='linear')` (Loss: `mae`)
- **Optimizer**: Adam ($\text{learning\_rate} = 1.25 \times 10^{-4}$)

---

## 📁 Repository Structure

```text
├── APP.py                               # Application script for running inference
├── gender_age_prediction_model.keras    # Trained Keras 3 model weights & architecture
├── haarcascade_frontalface_default.xml # OpenCV Haar Cascade face detection file
├── age-gender-detection-keras.ipynb     # Jupyter Notebook for model training & evaluation
├── requirements.txt                     # Project dependencies
└── README.md                            # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Age-Gender-Detection-Keras.git](https://github.com/YOUR_USERNAME/Age-Gender-Detection-Keras.git)
cd Age-Gender-Detection-Keras
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python APP.py
```

---

## 📊 Evaluation Metrics

- **Gender Classification**: Evaluated using `BinaryAccuracy`.
- **Age Estimation**: Evaluated using `MAE` (Mean Absolute Error).

---

## 👤 Author

- **Mohamed Khaled Al-Desouki**
- GitHub: [@YOUR_GITHUB_USERNAME](https://github.com/YOUR_GITHUB_USERNAME)
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/YOUR_LINKEDIN)
