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

## Key Features

