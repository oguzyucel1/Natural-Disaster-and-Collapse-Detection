# Disaster & Building Collapse Detection Models

## Table of Contents
- [Overview](#overview)
- [Models](#models)  
  - [Natural Disaster Detection](#natural-disaster-detection-model)  
  - [Building Collapse Detection](#building-collapse-detection-model)  
  - [Fire Detection](#fire-detection-model)  
- [Installation](#installation)
- [Usage](#usage)  
- [Training Details](#training-details)
- [Evaluation & Metrics](#evaluation--metrics)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview
This repository contains three computer-vision models for real-time detection and classification tasks:

1. **Natural Disaster Detection Model** – identifies events like volcanoes, avalanches, floods, wildfires, earthquakes, landslides, tsunamis, etc., using YOLOv11.  
2. **Building Collapse Detection Model** – finds and classifies collapsed vs. intact structures by combining YOLOv11 with MobileNetV2 and ResNet50 heads.  
3. **Fire Detection Model** – specifically detects active fires in images/video streams using YOLOv11.

---

## Models

### Natural Disaster Detection Model
- **Architecture**: YOLOv11  
- **Input**: Single images or video frames  
- **Output**: Bounding boxes + class labels for each disaster type  
- **Classes**:  
  - Volcano  
  - Avalanche  
  - Flood  
  - Wildfire  
  - Earthquake fault rupture  
  - Landslide  
  - Tsunami  
  - …and more  
- **Dataset**:  
  - Aggregated from public sources (NASA, USGS, NOAA) and web-scraped images, annotated in COCO format.  
- **Key Features**:  
  - Real-time inference (~45 FPS on a GTX 1080 Ti)  
  - Non-maximum suppression to reduce duplicates  
  - Adjustable confidence threshold via CLI/API  

### Building Collapse Detection Model
- **Detection Backbone**: YOLOv11 for coarse localization  
- **Classification Heads**:  
  - **MobileNetV2** (lightweight, low-latency)  
  - **ResNet50** (higher capacity, improved accuracy)  
- **Workflow**:  
  1. YOLOv11 proposes bounding boxes around building regions.  
  2. Cropped regions are passed to MobileNetV2 or ResNet50 classifiers to confirm “Collapsed” vs. “Intact.”  
- **Dataset**:  
  - Pre- and post-collapse building images from disaster-response sets, labeled for collapse status.  
- **Transfer Learning**:  
  - Base networks pretrained on ImageNet, fine-tuned on collapse dataset (224×224 inputs, batch size 32).

### Fire Detection Model
- **Architecture**: YOLOv11  
- **Input**: Images or video streams  
- **Output**: Bounding boxes around active fire regions  
- **Classes**:  
  - Fire  
- **Dataset**:  
  - Public fire imagery (forest fires, building fires, controlled burns), annotated in COCO format.  
- **Key Features**:  
  - Real-time inference (~45 FPS on a GTX 1080 Ti)  
  - Tunable confidence threshold  
  - Robust to smoke and varied lighting  

---

## 📸 Screenshots

Here are some example outputs from the models:

### Natural Disaster Detection
![Volcano Detection](screenshots/screenshot1.png)  
*Volcano detected in aerial imagery.*

![Flood Detection](screenshots/screenshot2.png)  
*Flooded area highlighted.*

### Building Collapse Detection
![Collapsed Building](screenshots/screenshot3.png)  
*Detected collapse on building façade.*

![Intact Building](screenshots/screenshot4.png)  
*Correctly classified intact structure.*

### Fire Detection
![Forest Fire](screenshots/screenshot5.png)  
*Detected active fire in forest.*

![Building Fire](screenshots/screenshot6.png)  
*Detected blaze in urban environment.*

### Natural Disaster Detection
![Volcano Detection](screenshots/screenshot1.png)  
*Volcano detected in aerial imagery.*

![Flood Detection](screenshots/screenshot2.png)  
*Flooded area highlighted.*

### Building Collapse Detection
![Collapsed Building](screenshots/screenshot3.png)  
*Detected collapse on building façade.*

![Intact Building](screenshots/screenshot4.png)  
*Correctly classified intact structure.*

### Fire Detection
![Forest Fire](screenshots/screenshot5.png)  
*Detected active fire in forest.*

![Building Fire](screenshots/screenshot6.png)  
*Detected blaze in urban environment.*


---


