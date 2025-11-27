# Face Verification & Recognition with DeepFace:
A Python command-line tool that performs **face verification** (1:1 comparison) and **face recognition** (1:N search in a database) using DeepFace. Supports input from **image files** or **live webcam**.

## Features:

- **Verification**: Compare two faces (e.g., "Is this the same person?")
- **Recognition**: Identify a person from a database of known faces
- **Webcam support**: Capture live image for analysis

## 📦 Requirements:

- Python 3.10
- A working webcam (for `--webcam` mode)
- `pip` package manager
- All required Packages in requirements.txt

## Start Guide:
-Open Terminal

# 1. Create Virtual Environment:
```python 
python -m venv venv
venv\Scripts\activate
``` 
# 2. Install Dependencies:
```python
pip install -r requirements.txt
```
No requirments.txt:
```python
deepface==0.0.96
opencv-python
matplotlib
tensorflow
```
# 3. Prepare Data:
***Note: This repo does not come with test images or database***
**For face verification :**
 Place two **images** eg. test and random image in project folder or can use **live webcam** for input **image**.

 **For Recognition :**
 Can use the same test **image** or **live webcam** and make the following database.


```python
database/
├── Ali/
│   └── ali1.jpg
├── Ahmad/
│   └── ahmad1.jpg
└── Zain/
    └── zain1.jpg
```
# 4. Usage:
***Verification:***

**From image files :**
```python
python main.py --test test.jpg --random random.jpg
```

**From webcam + reference image :**
```python
python main.py --webcam --random random.jpg
```
***Recognition:***

**From image files :**
```python
python main.py --test test.jpg --database database
```
**From webcam:**
```python
python main.py --webcam --database database
```

**Notes:**
Press SPACE to capture image from webcam
Press ESC to cancel webcam capture
First run may take longer (DeepFace builds face embeddings)