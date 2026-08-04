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
11.Before you use `main.py` to generate exe file, you need to put `train/weights.pt` and `shot_1-29` into `dist`, with code `pyinstaller --clean -D -w --collect-data ultralytics main.py`  
12.`scroing_system.py` includes action segmentation, calculation of critical angle of each section, technical integrity judgment, shot height rating, shot angle evaluation, power chain synergy and force rhythm, knee flexion score, ai score.
![action segmentation](https://github.com/cxxmarvinbest/Shooting-action-intelligent-scoring-system/blob/main/scoring_system/action%20segmentation.jpg)
![calculation of critical angle of each section](https://github.com/cxxmarvinbest/Shooting-action-intelligent-scoring-system/blob/main/scoring_system/calculation%20of%20critical%20angle.jpg)
![scoring](https://github.com/cxxmarvinbest/Shooting-action-intelligent-scoring-system/blob/main/scoring_system/scoring.jpg)
![technical integrity judgment](https://github.com/cxxmarvinbest/Shooting-action-intelligent-scoring-system/blob/main/scoring_system/technical%20integrity%20judgment.jpg)
![shot height rating](https://github.com/cxxmarvinbest/Shooting-action-intelligent-scoring-system/blob/main/scoring_system/shot%20height%20rating.jpg)
![shot angle evaluation](https://github.com/cxxmarvinbest/Shooting-action-intelligent-scoring-system/blob/main/scoring_system/shot%20angle%20evaluation.jpg)
![power chain synergy and force rhythm](https://github.com/cxxmarvinbest/Shooting-action-intelligent-scoring-system/blob/main/scoring_system/power%20Chain%20synergy%20and%20force%20rhythm.jpg)
![knee flexion score](https://github.com/cxxmarvinbest/Shooting-action-intelligent-scoring-system/blob/main/scoring_system/knee%20flexion%20score.jpg)
![ai score](https://github.com/cxxmarvinbest/Shooting-action-intelligent-scoring-system/blob/main/scoring_system/ai%20score.jpg)
13.`1pyqt6.py` cut multiple shots into a single

