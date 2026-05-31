<div align="center">

# 👁️‍🗨️ AI Face Detection System
**A real-time, deep-learning-powered computer vision application.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org)


</div>

<br>

A high-accuracy, interactive web application designed to detect faces in real-time. Built to handle dense crowds, varying lighting conditions, and dynamic environments, this project demonstrates a modern shift from legacy algorithmic detection to robust Deep Learning models.

---

### 🧠 Why AI over Traditional Methods?
*(Showcasing AI/ML Interest)*
Traditional Face Detection (like Haar Cascades) relies on rigid, manually defined patterns of light and dark shadows, leading to high false-positive rates (e.g., mistaking windows or patterns for faces). This project leverages **Google's MediaPipe**, a state-of-the-art Deep Learning model that understands complex facial topology, resulting in near-perfect accuracy even with partial occlusions or difficult angles.

---

### ⚙️ System Architecture Flowchart

*GitHub will automatically render this code block as a visual flowchart!*

```mermaid
graph TD
    A[User Interface Streamlit] -->|Input| B{Select Mode}
    B -->|Upload Image| C[Static Image Processing]
    B -->|Webcam| D[Live Video Stream]
    
    C --> E[OpenCV Pre-Processing]
    D --> E
    
    E --> F{Haar Cascade Classifier}
    F -->|Apply| G[Scale Factor & Min Neighbors]
    F -->|Extract| H[Bounding Box Coordinates]
    
    G --> I[OpenCV Rendering]
    H --> I
    
    I --> J[Display Results & Analytics]
    J --> K[Export / Download]
    
    classDef core fill:#005C53,stroke:#fff,stroke-width:2px,color:#fff;
    class F core;
```
---

### ✨ Core Features
* 🎯 **AI-Powered Accuracy:** Utilizes deep learning inference for superior, confident detection.
* 📸 **Dual Processing Modes:** Analyze static images (JPG, PNG, WEBP) or track faces via live webcam feed.
* 🎛️ **Interactive Analytics UI:** Adjust AI confidence thresholds and bounding box parameters on the fly and watch the model adapt in real-time.
* 💾 **Instant Export:** Download processed visuals with drawn tensor bounding boxes and confidence scores.

---
### ✨ Output Samples
<img label="1" width="1362" height="723" alt="Screenshot 2026-04-11 220955" src="https://github.com/user-attachments/assets/c7cd4277-23a0-4ca0-bc8e-80ac127669e5" />

<img label="2" width="1355" height="716" alt="Screenshot 2026-04-11 221016" src="https://github.com/user-attachments/assets/9d96c4c3-7201-4227-b375-58770c60fe8a" />


---

### 🚀 Quick Start / Local Deployment

**1. Clone the repository**
```bash
git clone [https://github.com/keshriaman231/Face-Detection-System.git](https://github.com/keshriaman231/Face-Detection-System.git)
cd Face-Detection-System
```
**2. Install dependencies**
```

python -m pip install -r requirements.txt

```

**3. Boot the application**
```
python -m streamlit run face_detect.py

```

<div align="center">



</div>
