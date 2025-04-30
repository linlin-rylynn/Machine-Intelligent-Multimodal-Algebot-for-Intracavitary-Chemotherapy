# Machine-intelligent-Multimodal-Algebot-for-Intracavitary-Chemotherapy

This project implements a deep learning (DL)-based intelligent navigation framework designed for autonomous microrobot control under real-time medical imaging guidance. The system integrates advanced image segmentation, object detection, motion planning, and robot actuation into a closed-loop control pipeline capable of tracking and guiding microrobot swarms toward specific targets (e.g., tumors).

It supports two control configurations:

- Coil-based control – applied to micro-scale ex-vivo multimodal control demonstrations

- Robotic magnet system (RMS)-based control – applied to macro-scale in-vivo tumor navigation

## Key Features

- DL-based semantic segmentation of medical images to identify targets (e.g., tumors), microrobots swarms, obstacles, and navigable space

- Real-time object detection using YOLOv5 to continuously locate the microrobot swarm

- Autonomous navigation via BFS-based path planning and dynamic re-routing

- Closed-loop visual servoing with dynamic pose correction

- Machine interaction for both coil and RMS—based control system via serial or TCP/IP communication

## System Requirements
  ### Core Framework Requirements
- Python: 3.7+ (Recommended 3.8 for TensorFlow 2.4 compatibility)
- TensorFlow: 2.4.0 (GPU version recommended)
- OpenCV: 4.2.0+ (Critical for image processing)
- ONNX Runtime: 1.14.1 (For model conversion/export)
  ### Deep Learning Components
- tensorflow-gpu: 2.4.0	
- h5py: 2.10.0
- onnx: 1.14.1
- tf2onnx: 1.16.1
  ### Deep Learning Components
- opencv-python: 4.2.0.34	
- Pillow: 8.2.0
- numpy: 1.21.6
  ### Control System Dependencies
- pygam: 2.6.1     
- pyserial: 3.5      
- openpyxl: 3.1.3   
- mss: 6.1.0       


