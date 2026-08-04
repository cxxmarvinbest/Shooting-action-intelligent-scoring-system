# 投篮动作智能评分系统
## 项目简介   
本项目是一个基于单目视觉与AI大模型的投篮动作智能评分系统，
系统集成了YOLO目标检测、人体姿态估计与动态时间规整（DTW）算法，
并融合了豆包大语言模型。只需输入一段投篮视频，
即可自动完成动作的二段式切分、多维度量化评分、逐帧骨架可视化，
并输出专业的AI教练指导反馈，为篮球爱好者提供一站式的技术评测平台。

### 一、技术栈   
**前端交互/UI框架:** PyQt6   
**核心逻辑语言：** Python   
**计算机视觉与图像处理：** OpenCV,Numpy       
**目标检测：** YOLOv8n   
**多目标跟踪：** ByteTracker  
**姿态估计：** YOLOv8n-pose   
**动作序列比对算法：** FastDTW,SciPy   
**AI大语言模型：** Doubao-Seed-2.1-turbo   

### 二、整体架构设计
**前端交互层：** 负责接受用户视频输入，展示逐帧交互画面、缩放大图、各类量化评分   
**算法处理层：** 负责逐帧解析视频数据，提出检测框和骨骼关键点，进行几何角度运算和动作切分   
**数据比对层：** 针对切分后的特征序列进行时序对齐，与内置的标准视频计算欧式距离与相似度   
**AI反馈层：** 调用豆包大模型API

### 三、功能模块   
**1.二段式动作智能切分：** 基于篮球底部与肩膀Y坐标的相对空间关系，
自动将一次投篮动作精准划分为“准备-下蹲”与“蹬伸-出手”两个阶段   
**2.YOLO多目标跟踪：** 每一帧同步检测运动员、篮球，过滤背景干扰   
**3.姿态估计：** 提取人体17个核心关键点   
**4.多角度量化评分体系：** 系统从6个维度综合打分，包括：
动作DTW相似度（对比内置标准动作库）、
核心环节完整度、
出手高度、
动力链协同发力节奏、
屈膝爆发力、
出手角度   
**5.AI智能教练评语生成：** 提取量化评分数据构建Prompt,
调用豆包大模型自动生成包含“亮点、问题、建议（辅助训练与修正细节）”三段式评语   
**6.可视化交互评估看板：** 提供分段视频的对比回放、动作骨架显示、逐帧动作查看  

### 四、应用场景   
**1.篮球投篮技术训练：** 帮助球员将抽象的发力感觉转化为“下肢-核心-上肢”的客观量化数据，
快速定位技术硬伤   
**2.个性化教学与评测：** 辅助篮球教练客观评估学员动作，降低专业动作分析门槛，提供可执行的改进方案   
**3.日常运动复盘：** 业余爱好者无需穿戴设备，仅需手机拍摄的单目视频即可获得“AI私教”级别的反馈   

### 五、支持的动作类型   
单次完整的篮球投篮动作（包含持球、下蹲蓄力、蹬伸发力到出手释放）   

### 六、项目目录结构   
lanqiu/   
├── datasets/                  # 训练所需的数据集目录   
│   └── lanqiu/                # 数据集  
│       ├── images/            # 图像数据    
│       └── labels/            # YOLO格式标签数据  
├── left-shot-clips/           # 左视角的标准投篮姿态切片视频库  
├── right-shot-clips/          # 右视角的标准投篮姿态切片视频库  
├── runs/                      # 训练记录与检测结果输出路径  
│   └── detect/train           # 训练生成的目标检测模型权重存放处   
├── 1api.py                    # 核心代码  
├── yolov8n.pt                 # 目标检测原始模型  
├── yolov8n-pose.pt            # 用于姿态估计的模型  
└── 1yolo-lanqiu.yaml          # 训练篮球与人体目标检测模型的配置文件

### 七、核心实现  
#### 动作完成度专项评分模块   
def compute_completeness(self, frame_metrics, fps)：   
    # 1.提取髋关节Y坐标   
    hip_ys = [m['hip_y'] for m in frame_metrics if m.get('hip_y') is not None]   
    # 2.提取人的高度   
    player_heights = [m['player_h'] for m in frame_metrics if m.get('player_h') is not None]   
    # 3.提取髋关节和膝关节角度   
    hip_angles  = [m['angles'][2] for m in frame_metrics if m.get('angles') is not None]  
    knee_angles = [m['angles'][3] for m in frame_metrics if m.get('angles') is not None]  
    # 4.找到髋关节最低点
    max_y_idx = np.argmax(smoothed_hips)  
    max_y_val = smoothed_hips[max_y_idx]  
    # 5.找到从视频开始到最低点期间以及从最低点到视频结束期间身体达到的最高位置  
    min_y_before = np.min(smoothed_hips[:max_y_idx + 1]) if max_y_idx > 0 else smoothed_hips[0]  
    min_y_after = np.min(smoothed_hips[max_y_idx:]) if max_y_idx < len(smoothed_hips) - 1 else max_y_val  
    # 6.下蹲蓄力环节
    is_coordinate_dropped = drop_ratio > 0.04  
    is_direction_changed = (drop_ratio > 0.02) and (rise_ratio > 0.02)  
    is_angle_flexed = (min_hip_angle < 160.0) or (min_knee_angle < 152.0)  
    if is_coordinate_dropped or is_direction_changed or is_angle_flexed:  
        has_squat = True  
    # 7.蹬伸发力环节  
    if post_min_knee and (np.max(post_min_knee) - np.min(knee_angles) > 15) and np.max(post_min_knee) > 155:  
        has_extension = True  
    elif rise_ratio > 0.05:  
        has_extension = True  
    # 8.出手释放环节  
    if m['wrist_y'] < m['shoulder_y'] and m['angles'][1] > 140:  
        has_release = True  

#### 屈膝爆发力专项评估模块
def compute_knee_power(self, frame_metrics, fps):  
    # 1.膝关节最大最小角度  
    min_knee = np.min(smoothed_knee)  
    max_knee = np.max(smoothed_knee)  
    # 2.膝关节变化幅度  
    amplitude = max_knee - min_knee  
    # 3.膝关节变化角速度  
    velocities = np.diff(smoothed_knee) * fps  
    max_velocity = np.max(velocities) if len(velocities) > 0 else 0  
    # 4.综合打分  
    amp_score = 100.0 - abs(amplitude - 75.0) * 1.5  
    amp_score = max(0.0, min(100.0, amp_score))  
    vel_score = (max_velocity / 350.0) * 100.0  
    vel_score = max(0.0, min(100.0, vel_score))  
    total_score = (amp_score * 0.5) + (vel_score * 0.5)  

#### 发力协调性评估模块  
def compute_coordination(self, frame_metrics):  
    # 1.找各关节角度峰值  
    t_hip = int(np.argmax(smooth(hip_angles)))  
    t_knee = int(np.argmax(smooth(knee_angles)))  
    t_shoulder = int(np.argmax(smooth(shoulder_angles)))  
    t_elbow = int(np.argmax(smooth(elbow_angles)))    
    t_release = int(np.argmin(smooth(wrist_ys)))  
    # 2.综合打分  
    score = 100.0  
    lags = [  
        ("膝-髋 (下肢)", t_knee - t_hip),  
        ("肩-膝 (躯干)", t_shoulder - t_knee),  
        ("肘-肩 (上肢)", t_elbow - t_shoulder),   
        ("腕-肘 (末端)", t_release - t_elbow)  
    ]  
    for name, lag in lags:  
        if lag < 0:  
            score += lag * 3  

#### 视觉处理模块  
def process_video(self, video_path):  
    # 1.读取视频  
    cap = cv2.VideoCapture(video_path)  
    # 2.启动模型遍历每一帧  
    while True:  
        det_results = self.det_model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False, conf=0.45)
        pose_results = self.pose_model.predict(frame, verbose=False)
    # 3.切分视频  
    idx_squat = frame_idx // 2  
    for m in frame_metrics:  
        if m.get('shoulder_y') is not None and len(m.get('ball_boxes', [])) > 0:  
            bx1, by1, bx2, by2 = m['ball_boxes'][0]  
            shoulder_y = m['shoulder_y']  
            if by2 < shoulder_y:  
                idx_squat = m['idx']  
                break   
    # 4.初始化列表用于储存第一阶段（准备-下蹲）和第二阶段（蹬伸-出手）的关节角度数据  
    seq1, seq2 = [], []   
    for m in frame_metrics:  
        if m['angles'] is not None:  
            if m['idx'] <= idx_squat:  
                seq1.append(m['angles'])  
            else:  
                seq2.append(m['angles'])   
    # 5.计算相对出手高度   
    valid_height_frames = [m for m in frame_metrics if 'wrist_y' in m and 'player_h' in m]   
    if valid_height_frames:   
        start_wrist_y = valid_height_frames[0]['wrist_y']   
        min_wrist_y = min(m['wrist_y'] for m in valid_height_frames)  
        player_h = valid_height_frames[0]['player_h']  
        rel_height = (start_wrist_y - min_wrist_y) / player_h if player_h > 0 else 0.0  
    else:
        rel_height = 0.0  
    # 6.定义视频编码器  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    # 7.输出视频的尺寸  
    out1 = cv2.VideoWriter(out1_path, fourcc, slow_fps, (480, 640))              
    out2 = cv2.VideoWriter(out2_path, fourcc, slow_fps, (480, 640))  
    # 8.定义骨架点的连接方式  
    skeleton_connections = [  
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  
    ]  
    # 9.解包出各关节角度  
    if data['angles'] is not None:  
        shoulder, elbow, hip, knee = data['angles']  
    # 10.循环结束，释放对象  
    cap.release()   
    s1 = np.array(seq1)  
    s2 = np.array(seq2)  

#### ai动作分析流程  
def run(self):  
    # 1.加载模型  
    self.det_model = YOLO("runs/detect/train/weights/best.pt")  
    self.pose_model = YOLO("yolov8n-pose.pt")  
    # 2.构建分析提示词  
    ai_prompt = f"""  
    你是一位专业的篮球教练。请根据以下我的投篮测试数据，给我一份简短、专业的中文改进建议：  
    1. 准备下蹲阶段动作相似度得分：{score1:.1f}/100  
    2. 蹬伸出手阶段动作相似度得分：{score2:.1f}/100  
    3. 核心技术环节完整度：{completeness_score:.1f}%  
    4. 动力链协同发力得分：{coord_score:.1f}/100  
    5. 屈膝爆发力得分：{knee_score:.1f}/100  
    6. 出手角度得分：{release_score:.1f}/100  
    [输出格式要求]  
    必须直接输出以下三段内容，禁止任何自我介绍或“收到”、“好的”等客套话，总字数控制在 150 字左右：  
    1.亮点：[结合得分最高的1-2个指标，夸奖动作做得好的地方]  
    2.问题：[结合得分偏低或不达标的指标，指出动作中的断档或核心硬伤]  
    3.建议：进行[具体的辅助训练]练习/辅助练习，加入[具体的修正细节]动作，体会[具体身体部位或技术环节]的发力。  
    """  
    # 3.设置大模型API    
    headers = {  
        "Content-Type": "application/json",  
        "Authorization": "xxx"  
    }  
    # 4.定义HTTP请求头  
    payload = {  
        "model": "ep-m-20260720143111-jtzg2",  
        "messages": [  
            {  
                "role": "user",  
                "content": [{"text": ai_prompt, "type": "text"}]  
            }  
        ]  
    }   
    # 5.发送POST请求  
    response = requests.post(api_url, headers=headers, json=payload, timeout=360)  

#### 设置UI界面  
def set_ui(self):  
    # 1.设置外边距  
    layout.setContentsMargins(20, 20, 20, 20)  
    # 2.设置标题  
    self.lbl_title = QLabel("🏀 投篮动作智能评分系统 (二段式评测)")  
    # 3.标题居中对齐  
    self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)  
    # 4.设置按键  
    self.btn_select = QPushButton("📁 选择测试视频")  
    self.btn_start = QPushButton("🚀 开始分段智能评分")  
    self.btn_play1 = QPushButton("▶️")  
    self.btn_play2 = QPushButton("▶️")  
    # 5.设置分段视频标题  
    self.lbl_video_title = QLabel("▶️ 分段姿态跟踪 (准备-下蹲 vs 蹬伸-出手)")   
    # 6.设置评分报告标题  
    self.lbl_score_title = QLabel("🏆 双阶段综合评测报告")  
    self.lbl_score_title.setObjectName("subtitle")  
    self.lbl_score_title.hide()  
    self.scroll_layout.addWidget(self.lbl_score_title)  
    # 7.设置各部分评分（动作完成度、出手高度、协调性、膝关节角度、出手角度）  
    self.lbl_completeness_score = QLabel("")  
    self.lbl_height_score = QLabel("")  
    self.lbl_coord_score = QLabel("")  
    self.lbl_knee_score = QLabel("")  
    self.lbl_release_score = QLabel("")  
    


    

    




    



