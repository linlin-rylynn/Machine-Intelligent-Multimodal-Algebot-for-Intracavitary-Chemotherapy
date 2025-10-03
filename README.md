
# 1. Overview
This project implements a deep learning (DL)-based intelligent navigation framework designed for autonomous microrobot control under real-time Ultrasound imaging guidance. The system integrates advanced image segmentation, object detection, motion planning, and robotic actuation into a closed-loop control pipeline capable of tracking and guiding microrobot swarms toward specific targets (e.g., tumors). This project supports two control configurations:

- Coil-based control – applied to micro-scale ex-vivo multimodal control demonstrations

- Robotic magnet system (RMS)-based control – applied to macro-scale in-vivo tumor navigation

## 1.1 Key Features

- DL-based semantic segmentation of Ultrasound images to identify targets (e.g., tumor region), microrobots (e.g., DMCG) swarms , obstacles (e.g., Non-tumor region), and navigable space (e.g., bladder lumen), etc.

- Real-time object detection using YOLOv5 to continuously locate the microrobot swarm

- Autonomous navigation via BFS-based path planning and dynamic re-routing

- Closed-loop visual servoing with dynamic pose correction
  
- Machine interaction for both coil and RMS—based control system via serial or TCP/IP communication

## 1.2 Directory Structure and Key Components

### a. Core Control Modules

| File/Folder                  | Description                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------|
| `Main_Coil_based_control.py` | Entry script for ex-vivo multimodal control (electromagnetic coil-driven microrobot swarms)     |
| `Main_RMS_based_control.py`  | Entry script for in-vivo tumor navigation (Robotic Magnetic System control loop)                |
| `Robotcient.py`              | Core hardware communication (serial/TCP-IP) for servo control of RMS                            |
| `testServoPtxt.py`           | RMS testing script                                                                              |

### b. Deep Learning Models

### UNet-based Segmentation

| File/Folder            | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| `Unet/unet.py`         | UNet architecture for medical image segmentation (tumors/obstacles)                            |
| `Unet/train.py`        | Training script                                     |
| `Unet/resnet50.py`     | Pre-trained ResNet50 backbone for UNet encoder                                                 |
| `Unet/vgg16.py`        | Pre-trained VGG16 backbone for UNet encoder                                                    |
| `Unet/unet_training.py`| Extended training logic (multi-GPU training, loss function optimization)                        |
| `Unet_utils/`          | Utilities (data augmentation, metrics, visualization)                                           |

### YOLO-based Detection

| File/Folder            | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| `YOLO/yolo.py`         | YOLOv5 implementation for real-time microrobot swarm detection                                  |
| `YOLO/CSPdarknet.py`   | CSPDarknet backbone for YOLO                                                                    |
| `YOLO/train.py`        | Training script                                               |
| `YOLO/yolo_training.py`| Extended training logic (anchor box optimization, learning rate scheduling)                     |
| `YOLO_utils/`          | Helper tools (NMS post-processing, dataset formatting)                                          |


# 2. System Requirements

  ## 2.1 Core Framework 
  - Python: 3.7+ 
  - TensorFlow: 2.4.0 
  - OpenCV: 4.2.0+ 
  - ONNX Runtime: 1.14.1 
  ## 2.2 Deep Learning Components
  - tensorflow-gpu: 2.4.0
  - cuda：10.1
  - h5py: 2.10.0
  - onnx: 1.14.1
  - tf2onnx: 1.16.1
  - CUDA: 10.1
  - cuDNN: 7.6.5
  ## 2.3 Image Processing Stack
  - opencv-python: 4.2.0.34	
  - Pillow: 8.2.0
  - numpy: 1.21.6
  ## 2.4 Control System Dependencies
  - pygame: 2.6.1     
  - pyserial: 3.5      
  - openpyxl: 3.1.3   
  - mss: 6.1.0

# 3. Installation guide
    git clone https://github.com/linlin-rylynn/Machine-Intelligent-Multimodal-Algebot-for-Intracavitary-Chemotherapy

# 4. Demo
  ### Demonstration of segmentation
   ![Demo of segmentation](Demo/Demo%20of%20bladder%20tumor%20segmentation.png)
  ### Demonstration of navigation
   ![Demo of Navigation](Demo/Demo%20of%20autonomous%20navigation%20in%20vivo.png)

# 5. Instructions for use
## 5.1 Running the Control System
Once training is completed and the trained weights are placed into the corresponding `./weights/` folder (replacing the provided pretrained weights), the full system can be executed directly. The main scripts automatically call **YOLO** and **UNet**, so there is no need to run them separately unless for testing.
### a) Coil-based Control

```
python Main_Coil_based_control.py
```

Input: Real-time Bright field or Ultrasound image stream

Output: Closed-loop navigation control signals sent to the electromagnetic coil driver

Communication: Serial port (adjust COM port in script)

### b) RMS-based Control

```
python Main_RMS_based_control.py
```

Input: Real-time Ultrasound image stream

Output: RMS actuation commands via TCP/IP

Communication: IP/Port configuration can be modified in Robotcient.py

To test servo communication with the RMS hardware:

```
python testServoPtxt.py
```

## 5.2 Running Deep Learning Models
Although the main scripts already integrate segmentation (UNet) and detection (YOLO), you may run or retrain them independently if needed.
### a) UNet Segmentation

```
cd Unet
python unet.py
```

To train a new UNet model:

```
python train.py
```

### b) YOLOv5 Detection

```
cd YOLO
python yolo.py
```

To train YOLOv5 on a new dataset:

```
python train.py
```

## 5.3 Notes

Ensure that the required dependencies listed in System Requirements are installed.

For GPU acceleration, confirm CUDA and cuDNN versions are correctly configured.

Example datasets and pre-trained weights should be placed in ./weights/ and ./dataset/ folders, respectively.

Communication settings (serial port, IP, baud rate) may need to be modified in Robotcient.py before running experiments.

# 6. License
 - This project is covered under the Apache 2.0 License.

# 7. Reference
[[1] Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534)

[[2] YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)

[[3] YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

[[4] Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

## 8. Acknowledgment
  tf_unet    [https://github.com/jakeret/tf_unet](https://github.com/jakeret/tf_unet).

  Yolov5_tf   [https://github.com/jakeret/tf_unet](https://github.com/jakeret/tf_unet).

    


