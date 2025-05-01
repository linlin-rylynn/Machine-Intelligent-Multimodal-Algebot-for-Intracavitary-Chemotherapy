# Machine-intelligent-Multimodal-Algebot-for-Intracavitary-Chemotherapy

This project implements a deep learning (DL)-based intelligent navigation framework designed for autonomous microrobot control under real-time Ultrasound imaging guidance. The system integrates advanced image segmentation, object detection, motion planning, and robotic actuation into a closed-loop control pipeline capable of tracking and guiding microrobot swarms toward specific targets (e.g., tumors).                                    
  ### Demonstration of segmentation
   ![Demo of segmentation](Demonstration%20results/Demo%20of%20bladder%20tumor%20segmentation.png)
  ### Demonstration of Navigation
   ![Demo of Navigation](Demonstration%20results/Demo%20of%20microrobot%20swarm%20autonomous%20navigation%20to%20tumor%20regime.png)

This project supports two control configurations:

- Coil-based control – applied to micro-scale ex-vivo multimodal control demonstrations

- Robotic magnet system (RMS)-based control – applied to macro-scale in-vivo tumor navigation


## Key Features

- DL-based semantic segmentation of Ultrasound images to identify targets (e.g., tumor regime), microrobots (e.g., DMCG) swarms , obstacles (e.g., Non-tumor regime), and navigable space (e.g., bladder lumen), etc.

- Real-time object detection using YOLOv5 to continuously locate the microrobot swarm

- Autonomous navigation via BFS-based path planning and dynamic re-routing

- Closed-loop visual servoing with dynamic pose correction
  
- Machine interaction for both coil and RMS—based control system via serial or TCP/IP communication
## Directory Structure and Key Components

### 1. Core Control Modules

| File/Folder                  | Description                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------|
| `Main_Coil_based_control.py` | Entry script for ex-vivo multimodal control (electromagnetic coil-driven microrobot swarms)     |
| `Main_RMS_based_control.py`  | Entry script for in-vivo tumor navigation (Robotic Magnetic System control loop)                |
| `Robotcient.py`              | Core hardware communication (serial/TCP-IP) for coils/RMS and servo control                    |
| `testServoPtxt.py`           | Servo motor/actuator testing script                                                            |

### 2. Deep Learning Models

#### UNet-based Segmentation

| File/Folder            | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| `Unet/unet.py`         | UNet architecture for medical image segmentation (tumors/obstacles)                            |
| `Unet/train.py`        | Training script                                     |
| `Unet/resnet50.py`     | Pre-trained ResNet50 backbone for UNet encoder                                                 |
| `Unet/vgg16.py`        | Pre-trained VGG16 backbone for UNet encoder                                                    |
| `Unet/unet_training.py`| Extended training logic (multi-GPU training, loss function optimization)                        |
| `Unet_utils/`          | Utilities (data augmentation, metrics, visualization)                                           |

#### YOLO-based Detection

| File/Folder            | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| `YOLO/yolo.py`         | YOLOv5 implementation for real-time microrobot swarm detection                                  |
| `YOLO/CSPdarknet.py`   | CSPDarknet backbone for YOLO                                                                    |
| `YOLO/train.py`        | Training script                                               |
| `YOLO/yolo_training.py`| Extended training logic (anchor box optimization, learning rate scheduling)                     |
| `YOLO_utils/`          | Helper tools (NMS post-processing, dataset formatting)                                          |


## System Requirements

  ### Core Framework 
  - Python: 3.7+ 
  - TensorFlow: 2.4.0 
  - OpenCV: 4.2.0+ 
  - ONNX Runtime: 1.14.1 
  ### Deep Learning Components
  - tensorflow-gpu: 2.4.0	
  - h5py: 2.10.0
  - onnx: 1.14.1
  - tf2onnx: 1.16.1
  ### Image Processing Stack
  - opencv-python: 4.2.0.34	
  - Pillow: 8.2.0
  - numpy: 1.21.6
  ### Control System Dependencies
  - pygame: 2.6.1     
  - pyserial: 3.5      
  - openpyxl: 3.1.3   
  - mss: 6.1.0

## Acknowledgment
  tf_unet    [https://github.com/jakeret/tf_unet](https://github.com/jakeret/tf_unet).

  Yolov5_tf   [https://github.com/jakeret/tf_unet](https://github.com/jakeret/tf_unet).

    


