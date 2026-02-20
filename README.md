# road_anomaly_detection
Edge AI system for detecting real-time road anomalies using YOLOv5 and Raspberry Pi
# Edge AI Based Road Anomaly Detection System

##  Overview

This project presents an Edge AI-based Road Anomaly Detection System designed to detect and log road irregularities such as potholes, cracks, stones, in real time. The system integrates object detection, temporal validation, risk scoring, and heatmap visualization to enhance road safety and driver awareness.

The solution is optimized for edge deployment (e.g., Raspberry Pi) and is suitable for real-time dashcam-based monitoring applications.

---

##  Objective

- Detect road anomalies such as:
  - Potholes
  - Road obstacles (stones)
  - Cracks
- Perform detection under low-light/night conditions
- Generate heatmaps for frequently detected anomalies
- Provide alerts to the driver
- Log events and capture clips for future analysis

---

##  System Architecture

**Flow:**

Camera  
→ Object Detection  
→ Temporal Validation  
→ Risk Scoring  
→ Heatmap Generation  
→ Alert System  
→ Logging & Clip Capture  

### Module Description

- **Camera Input:** Captures real-time video frames.
- **Object Detection:** YOLOv5-based model detects anomalies.
- **Temporal Validation:** Reduces false positives using frame consistency checks.
- **Risk Scoring:** Assigns severity levels based on object type, size, and proximity.
- **Heatmap Generation:** Visualizes frequently occurring anomaly zones.
- **Alert System:** Notifies the driver of high-risk detections.
- **Logging & Clip Capture:** Stores detected events with timestamps.

---

##  Model Used

- YOLOv5 for object detection
- Custom-trained dataset (road anomaly images)
- Data augmentation applied (blur, rain simulation, brightness adjustments)

---
