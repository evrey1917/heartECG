# **Month report**  

https://docs.google.com/presentation/d/1avHUWtMO73m8kjQsVnwcHC3RDxnyX3yIQ44hBJESsgc/edit#slide=id.p

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
