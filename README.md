# 🌾 Rice Variety Identification using Swin Transformer

## 📌 Project Overview

This project presents a deep learning-based approach for **fine-grained rice variety classification** using a **Swin Transformer (Vision Transformer)** model. The system allows users to upload an image of rice grains and predicts the corresponding rice variety.

The application is built using **Flask** for the backend and provides a simple web interface for real-time predictions.

---

## 🎯 Objectives

* Classify rice grains into different varieties
* Utilize advanced deep learning (Vision Transformers)
* Provide a user-friendly web interface
* Achieve high accuracy in fine-grained classification

---

## 🧠 Model Details

* Model: **Swin Transformer (Tiny)**
* Framework: **PyTorch + Hugging Face Transformers**
* Input Size: **224 × 224**
* Output Classes:

  * Arborio
  * Basmati
  * Ipsala
  * Jasmine
  * Karacadag

---

## ⚙️ Tech Stack

* Python
* Flask
* PyTorch
* Torchvision
* Hugging Face Transformers
* HTML / CSS

---

## 📂 Project Structure

```
project/
│── app.py
│── swin_rice_model.pth
│── requirements.txt
│── README.md
│
├── static/
│   └── video/
│       └── background.mp4
│
├── templates/
│   └── index.html
│
├── uploads/
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```
git clone https://github.com/Shyam-Prasath/Rice-Identification-Using-Swin-Transformer.git
cd Rice-Identification-Using-Swin-Transformer
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```
python app.py
```

### 4️⃣ Open in Browser

```
http://127.0.0.1:5000/
```

---

## 🖼️ How It Works

1. User uploads a rice grain image
2. Image is preprocessed (resize + normalization)
3. Passed into Swin Transformer model
4. Model predicts rice variety
5. Result + confidence score displayed

---

## 📊 Features

* Real-time image classification
* Deep learning-based prediction
* Clean and interactive UI
* Confidence score display
* Supports multiple rice varieties

---

## ⚠️ Notes

* Ensure `swin_rice_model.pth` is present in the root directory
* Large datasets (`train/`, `test/`) are excluded from GitHub
* Use GPU for faster inference (optional)

---

## 🔮 Future Enhancements

* Add more rice varieties
* Deploy as a web service
* Mobile app integration
* Model optimization for faster inference
* Add Grad-CAM visualization
