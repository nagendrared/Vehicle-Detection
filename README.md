
<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=1000&color=00D4FF&center=true&vCenter=true&width=800&lines=🚗+Vehicle+Detection+System;YOLOv11+%7C+Indian+Traffic+AI;Real-Time+%7C+113+FPS+%7C+99.36%25+mAP" />
</p>

<br/>

<img src="https://img.shields.io/badge/YOLOv11-Custom%20Trained-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/mAP%4050-99.36%25-00C851?style=for-the-badge&logo=checkmarx&logoColor=white"/>
<img src="https://img.shields.io/badge/Speed-~113%20FPS-FFD700?style=for-the-badge&logo=lightning&logoColor=black"/>
<img src="https://img.shields.io/badge/React-TypeScript-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
<img src="https://img.shields.io/badge/Flask-Python-000000?style=for-the-badge&logo=flask&logoColor=white"/>

<br/><br/>

> **A production-grade, end-to-end AI system for real-time multi-class vehicle detection in Indian traffic conditions — powered by a custom YOLOv11 model with a modern full-stack web interface.**

<br/>

[📸 Demo](#-demo--screenshots) · [⚡ Quick Start](#️-installation--setup) · [📡 API Docs](#-api-reference) · [📊 Results](#-performance-metrics) · [🔮 Roadmap](#-future-roadmap)

</div>

---

## 📌 Table of Contents

- [Why This Project?](#-why-this-project)
- [System Architecture](#️-system-architecture)
- [Model Deep Dive](#-model-deep-dive)
- [Performance Metrics](#-performance-metrics)
- [Demo & Screenshots](#-demo--screenshots)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#️-installation--setup)
- [API Reference](#-api-reference)
- [Core Features](#-core-features)
- [Real-World Challenges Solved](#-real-world-challenges-solved)
- [Future Roadmap](#-future-roadmap)
- [Team](#-team)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Why This Project?

Indian roads present **some of the most complex traffic scenarios in the world** — extreme vehicle density, non-standard vehicle types like auto-rickshaws, frequent occlusion, erratic motion, low-light conditions, and high variability in road/camera quality.

Off-the-shelf models trained on Western datasets (e.g., COCO) fail here. We built **a domain-specific solution from the ground up**:

| Problem | Our Approach |
|---|---|
| Auto-rickshaws not recognized by generic models | Custom 6-class training on Indian dataset |
| Dense traffic with high occlusion | PAN-FPN bidirectional feature fusion |
| Night & poor lighting | Augmentation with HSV jitter + diverse data |
| Motion blur from fast vehicles | Mosaic + MixUp augmentation strategies |
| Slow inference on real-time feeds | YOLOv11 optimized at ~8.8 ms/image |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│   React (Vite + TypeScript) · Tailwind CSS · Lucide Icons       │
│   ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌─────────┐  │
│   │  Upload  │  │   Preview    │  │ Detection  │  │Analytics│  │
│   │   Zone   │  │   + Zoom     │  │  History   │  │Dashboard│  │
│   └──────────┘  └──────────────┘  └────────────┘  └─────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (multipart/form-data)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API LAYER (Flask)                        │
│   POST /detect · Confidence Filter · Class Filter · CORS        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE ENGINE                            │
│   Ultralytics YOLOv11 · OpenCV · NumPy                          │
│                                                                 │
│   Input (any size)                                              │
│       ↓  Resize to 640×640                                      │
│   C3k2 Backbone  →  PAN-FPN Neck  →  Multi-scale Head          │
│   (Feature Ext.)    (P3/P4/P5)        (NMS + Decode)           │
│       ↓                                                         │
│   6-Class Detections + Bounding Boxes + Confidence Scores       │
│       ↓                                                         │
│   Annotated Image (Base64) + JSON Response                      │
└─────────────────────────────────────────────────────────────────┘
```

## 🏗️ Full Architecture Diagram

> 📎 Full architecture diagram: ![Architecture Diagram](./architecture.png)


---

## 🧬 Model Deep Dive

### Architecture: YOLOv11

YOLOv11 introduces architectural refinements over YOLOv8, particularly in the backbone's C3k2 blocks and the enhanced multi-scale detection head — delivering faster convergence and higher accuracy on domain-specific datasets.

```
Input Image (640 × 640 × 3)
        │
        ▼
┌──────────────────────┐
│   C3k2 Backbone      │  ← Cross-Stage Partial with kernel-2 attention
│   (Feature Extractor)│    Extracts semantic + spatial features
└──────────┬───────────┘
           │  P3 (large obj) / P4 (medium) / P5 (small)
           ▼
┌──────────────────────┐
│   PAN-FPN Neck       │  ← Bidirectional feature pyramid
│   (Feature Fusion)   │    Fuses high-level semantics + low-level detail
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Detection Head     │  ← Multi-scale predictions
│   NMS + Decode       │    Bounding boxes, class scores, confidence
└──────────────────────┘
```

### Dataset

| Property | Details |
|---|---|
| Total Images | 5,000+ annotated |
| Source | Indian Vehicle Dataset (domain-specific) |
| Annotation Format | YOLO (normalized `xywh`) |
| Train / Val Split | 80% / 20% |

**Detected Classes:**

```
🚗 Car          🚌 Bus          🚚 Truck
🏍️ Motorcycle   🚲 Bicycle      🛺 Auto-rickshaw
```

### Training Configuration

```yaml
model:       YOLOv11
optimizer:   AdamW
epochs:      30
batch_size:  16
input_size:  640 × 640
lr0:         0.01
lrf:         0.01

augmentations:
  - Mosaic blending (4-image)
  - MixUp
  - Random rotation & scaling
  - HSV jitter (hue, saturation, value)
  - Horizontal flip
  - Perspective transform
```

---

## 📈 Performance Metrics

<div align="center">

| Metric | Value | Benchmark |
|---|---|---|
| **mAP@50** | **0.9936** | 🏆 State-of-the-art |
| **mAP@50–95** | **0.9650** | 🏆 Excellent |
| **Precision** | **0.9980** | ✅ Near perfect |
| **Recall** | **0.9928** | ✅ Near perfect |
| **Latency** | **~8.8 ms/image** | ⚡ Real-time |
| **Throughput** | **~113 FPS** | ⚡ Production-ready |

</div>

> 📊 Training logs and curves: [`./notebook/training.ipynb`](./notebook/training.ipynb)

**Confusion Matrix Highlights:**
- Zero false negatives on auto-rickshaw class (hardest class for generic models)
- Precision holds above 0.99 across all 6 classes under varied lighting

---

## 📸 Demo & Screenshots

| Interface | Preview |
|---|---|
| 🖼️ Upload Interface | ![Upload UI](./upload.png) |
| 🎯 Detection Output | ![Detection Result](./result.png) |
| ⚙️ Settings Panel | ![Settings](./settings.png) |
| 📦 Batch Processing   | ![Batch](./batch.png) |

> 🎥 Demo Video: [Add link]
> 🌐 Live Demo: [Add deployment link]

---

## 🧩 Tech Stack

```
┌─────────────┬────────────────────────────────────────┐
│  Layer      │  Technologies                          │
├─────────────┼────────────────────────────────────────┤
│  Frontend   │  React 18 · Vite · TypeScript          │
│             │  Tailwind CSS · Lucide Icons            │
│             │  Dark Mode · Responsive UI              │
├─────────────┼────────────────────────────────────────┤
│  Backend    │  Flask · Python 3.10+                  │
│             │  OpenCV · NumPy · Flask-CORS            │
├─────────────┼────────────────────────────────────────┤
│  AI / ML    │  Ultralytics YOLOv11                   │
│             │  Custom-trained on Indian dataset       │
│             │  PyTorch (inference backend)            │
└─────────────┴────────────────────────────────────────┘
```

---

## 📂 Project Structure

```bash
VehicleDetection/
│
├── 📁 frontend/                    # React application
│   ├── 📁 src/
│   │   ├── App.tsx                 # Root component
│   │   ├── main.tsx                # Entry point
│   │   ├── index.css               # Global styles
│   │   └── vite-env.d.ts           # Vite type definitions
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── tsconfig*.json
│
├── 📁 backend/                     # Flask API server
│   ├── app.py                      # Main Flask app + /detect endpoint
│   ├── yolov11_model.pt            # Custom-trained weights (not tracked in git)
│   ├── 📁 input/                   # Uploaded images (temp)
│   └── 📁 output/                  # Annotated output images
│
├── 📁 notebook/
│   └── training.ipynb              # Full training pipeline + metrics
│
├── architecture.png                # System architecture diagram
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

| Tool | Version |
|---|---|
| Python | 3.9+ |
| Node.js | 18+ |
| pip | Latest |
| npm | Latest |

---

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/vehicle-detection.git
cd vehicle-detection
```

---

### 2️⃣ Backend Setup

```bash
cd backend

# Install dependencies
pip install ultralytics flask opencv-python flask-cors

# Start the API server
python app.py
```

Backend runs at: **http://localhost:5000**

> ⚠️ Ensure `yolov11_model.pt` is present in the `backend/` directory before starting.

---

### 3️⃣ Frontend Setup

```bash
cd frontend

# Install Node dependencies
npm install

# Start the development server
npm run dev
```

Frontend runs at: **http://localhost:5173**

---

### ✅ Verify Setup

Open your browser at `http://localhost:5173`, upload a traffic image, and you should see bounding boxes rendered with confidence scores.

---

## 📡 API Reference

### `POST /detect`

Runs vehicle detection on a submitted image.

**Request** (`multipart/form-data`)

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `File` | ✅ | Image file (JPEG, PNG) |
| `min_confidence` | `float` | ❌ | Minimum confidence score (0–1). Default: `0.5` |
| `max_results` | `int` | ❌ | Maximum detections to return. Default: `100` |
| `classes` | `string` | ❌ | Comma-separated class names to filter (e.g. `"car,bus"`) |

**Response** (`application/json`)

```json
{
  "success": true,
  "count": 4,
  "processing_time": 0.0088,
  "detections": [
    {
      "label": "car",
      "score": 0.9912,
      "box": [120, 85, 340, 210],
      "dimensions": {
        "width": 220,
        "height": 125,
        "area": 27500
      }
    },
    {
      "label": "auto-rickshaw",
      "score": 0.9734,
      "box": [400, 100, 580, 260],
      "dimensions": {
        "width": 180,
        "height": 160,
        "area": 28800
      }
    }
  ],
  "image_base64": "<base64-encoded annotated image>"
}
```

**Error Response**

```json
{
  "success": false,
  "error": "No file provided"
}
```

---

## ✨ Core Features

### 🔍 Detection Engine
- Multi-class vehicle detection across 6 Indian traffic categories
- Configurable confidence threshold and result count
- Optional class-based filtering
- Precise bounding box coordinates with pixel dimensions

### 🖥️ Frontend Interface
- **Drag & drop** image upload zone
- Real-time detection preview with zoom + fullscreen
- **Batch processing** support for multiple images
- Detection history log per session
- Dark mode toggle

### ⚙️ Settings & Customization
- Min confidence slider
- Max results control
- Class filter checkboxes
- Live parameter adjustment without page reload

### ⚡ Performance Optimizations
- Standardized 640×640 inference resolution for consistent speed
- Efficient Ultralytics YOLO pipeline (C++ backend via PyTorch)
- Base64 image transfer (no disk I/O on response)
- Non-Maximum Suppression (NMS) for clean bounding boxes

---

## 🛡️ Real-World Challenges Solved

| Challenge | Solution Applied |
|---|---|
| **Occluded vehicles in dense traffic** | PAN-FPN bidirectional fusion retains multi-scale context |
| **Night / low-light conditions** | HSV jitter augmentation + diverse lighting in dataset |
| **Motion blur on fast-moving vehicles** | MixUp + Mosaic augmentation for robust feature learning |
| **Auto-rickshaws unseen by COCO models** | Domain-specific custom class training |
| **Varying camera angles & distances** | Random rotation, scaling, perspective transform |
| **Real-time processing requirement** | YOLOv11 with ~8.8ms latency (~113 FPS) |

---

## 🔮 Future Roadmap

- [ ] 🎥 **Video stream detection** — real-time frame-by-frame processing
- [ ] 🔁 **Multi-object tracking** (SORT / DeepSORT integration)
- [ ] 🔢 **Number plate recognition** — ANPR module for Indian plates
- [ ] 📊 **Traffic analytics dashboard** — vehicle counts, lane density, heatmaps
- [ ] 🌐 **Cloud deployment** — Docker + AWS/GCP/Azure
- [ ] 📱 **Edge deployment** — TensorRT / ONNX export for Jetson Nano / Raspberry Pi
- [ ] 📷 **Multi-camera support** — simultaneous RTSP stream inference
- [ ] 🚦 **Traffic signal integration** — adaptive signal control prototype

---

## 👨‍💻 Team

<div align="center">

| Name |
|------|
| **Keshavareddy Nagendra Reddy** |
| **Poladasu Ramesh Goud** |
| **Jillella Sahithi** |

</div>

---

## 🙏 Acknowledgements

We express our sincere gratitude to:

**Mr. Y. P. Srinath Reddy**
*Assistant Professor, CSE (Data Science)*
*RGMCET, Nandyal*

For his continuous mentorship, technical guidance, and support throughout this project.

---

## ⭐ Why This Project Stands Out

```
✅ Real-world problem   — Built for Indian traffic, not a generic demo
✅ End-to-end system    — From raw data to trained model to deployed web app
✅ Full-stack + ML      — React frontend + Flask API + Custom YOLO model
✅ Near-perfect metrics — 99.36% mAP@50, 99.80% Precision
✅ Production-ready     — ~113 FPS inference, responsive UI, RESTful API
✅ Open & extensible    — Clean codebase ready for tracking, ANPR, video
```

---

## 📜 License

This project is intended for **academic and research use only**.
Please do not use for commercial purposes without explicit permission from the authors.

---

## 💬 Feedback & Support

Found a bug? Have a suggestion?

- 🐛 Open an [Issue](https://github.com/your-username/vehicle-detection/issues)
- 💡 Submit a [Pull Request](https://github.com/your-username/vehicle-detection/pulls)
- ⭐ Star this repo if it helped you!

---

<div align="center">

**Made with ❤️ at RGMCET, Nandyal — for smarter Indian roads**

<img src="https://img.shields.io/github/stars/your-username/vehicle-detection?style=social"/>
<img src="https://img.shields.io/github/forks/your-username/vehicle-detection?style=social"/>

</div>
