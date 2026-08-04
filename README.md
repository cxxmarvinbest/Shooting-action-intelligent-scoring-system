# Shooting-action-intelligent-scoring-system
## Project Overview
The system automatically completes player and basketball detection, body key point posture estimation, two-stage segmentation of shooting movements, multi-angle quantitative scoring, standard video action comparison, and AI intelligent coach comment generation, and outputs quantitative scoring reports and improvement suggestions for the shooters.  
## Features
• Video upload and automatic parsing  
• Human and basketball target detection  
• Pose estimation of key points of the human body  
• Two-stage automatic segmentation of shooting action  
• Multi-angle quantitative scoring  
• Standard video motion comparison  
• AI Smart Coach Review  
## Technologies Used 
**Front-end interaction/UI framework:** PyQt6  
**Core Logic Language:** Python  
**Computer Vision and Image Processing:** OpenCV, Numpy  
**Target detection:** YOLOv8n  
**Multi-Target Tracking:** ByteTracker  
**Pose Estimate:** YOLOv8n-pose  
**Action Sequence Alignment Algorithm:** FastDTW, SciPy  
**AI Large Language Model:** Doubao-Seed-2. 1-turbo  
## Usage
1.Install Miniconda on the C drive, and make sure the installation directory does not contain any Chinese characters(https://mirroes.tuna.tsinghua.edu.cn/anaconda/miniconda/)  
2.Create a virtual environment(conda create -n lanqiu python=3.8)  
3.Configure domestic mirrors(https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)  
4.Use `MakeSense` to annotate the image, dividing it into `0 player` and `1 basketball`  
5.Split the `images` and `Annotation` into `train` and `val`, and place them into `datasets`  
6.Configure and specify the paths of the dataset and category information:`1yolo-lanqiu.yaml`  
7.Train model:`1yolo-train.py`
8.Generative model:`train/weights.pt`  
9.`beginner_shooting_videos` and `demo.mp4` are videos for testing  
10.`left_posture_estimation_and_detection_result.mp4` is video from `demo.mp4` with target detection frame and pose estimation skeleton  
11.Before you use `main.py` to generate exe file, you need to put `train/weights.pt` and  
