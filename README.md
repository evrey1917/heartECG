# 🚀 ECG Classifier — Multi-label Diagnosis from 12-lead Clinical ECGs

## Project Overview

This project focuses on automatic classification of ECG signals using a convolutional autoencoder followed by a multi-label classifier. The model is trained on a large-scale clinical dataset following the AHA (American Heart Association) diagnostic standard. A web-based application allows users to upload ECG recordings and patient information to receive diagnostic outputs.

---

## 📊 Dataset Description

- **Source**: Shandong Provincial Hospital  
- **Collection Period**: August 2019 to August 2020  
- **Size**: 25,770 12-lead ECG records from 24,666 patients  
- **Format**: 12-lead, 500 Hz sampling frequency, 10 to 60 seconds duration  
- **Demographics**: 55.36% male, 44.64% female  
- **Abnormalities**:  
  - 46.04% of records contain abnormalities  
  - 44 primary diagnostic statements + 15 modifiers (per AHA standard)  
  - 14.45% of all records and 31.39% of abnormal records contain multiple diagnostic labels (multi-label)

# Original dataset
https://springernature.figshare.com/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802/1

---

## ⚙️ Data Preprocessing

Each ECG signal undergoes the following preprocessing steps:

- **Low-pass filtering** (to remove high-frequency noise)
- **High-pass filtering** (to remove baseline wander and low-frequency drift)
- **Detrending** of the signal

These steps improve the signal quality and aid in robust feature extraction.

---

## 🧠 Model Architecture

The model includes two main components:

1. **Convolutional Autoencoder (CAE)**:
   - Learns compressed representations of ECG signals
   - Reduces dimensionality while preserving diagnostic information

2. **Classifier**:
   - Fully-connected neural network
   - Supports multi-label output with sigmoid activation
   - Predicts diagnostic statements and modifiers

---

## 🌐 Web Application

**Interface Features**:

- Upload 12-lead ECG recordings
- Provide patient demographics (age, sex)
- Receive predicted diagnostic labels (based on AHA standard)
- Output includes class probabilities for each diagnosis

---


# **ECG Health Check (Django App)**  

A Django-based application for uploading `.h5` files containing ECG data, entering height and weight, and analyzing the health status.  

## 🚀 **How to Run the Project**  

### **1. Clone the Repository**  
```bash
git clone -b site https://github.com/evrey1917/heartECG.git
cd heartECG
```
### **2. Install dependences**  
```bash
pip install -r req.txt

```
### **3. Apply Database Migrations**
```bash
python manage.py makemigrations site_ECG
python manage.py migrate
```
### **4. Run the Server**
```bash
python manage.py runserver
```
Now, open your browser and go to:
http://127.0.0.1:8000/


## 🛠 **How to Use**
### **1.Upload an .h5 file containing ECG data and enter height and weight.**
### **2.Click the "Submit" button.**
### **3.The server processes the data and provides a result: "Healthy" or "Abnormalities detected".**

