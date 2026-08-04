import os
import cv2
import numpy as np
import sys
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from ultralytics import YOLO

# ========== PyQt6 导入 ==========
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QProgressBar, QScrollArea, QMessageBox, QDialog,
                             QSlider, QGraphicsView, QGraphicsScene)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap


# ==========================================
# 1. UI组件
# ==========================================
class ClickableLabel(QLabel):
    # 传递当前图片的索引
    clicked = pyqtSignal(int)

    def __init__(self, index, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)


class ImageViewerDialog(QDialog):
    def __init__(self, image_paths, start_index, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.current_index = start_index

        self.setWindowTitle("🔍 查看骨架大图 (按住 Ctrl + 鼠标滚轮缩放，鼠标左键拖拽)")
        self.setStyleSheet("background-color: #1E1E2E; color: white;")
        self.resize(1000, 750)

        layout = QVBoxLayout(self)

        # 使用 QGraphicsView 支持高级缩放和拖拽
        self.view = QGraphicsView()
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # 允许鼠标拖拽平移
        self.view.setStyleSheet("border: none; background-color: #11111B;")
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.pixmap_item = self.scene.addPixmap(QPixmap())
        layout.addWidget(self.view)

        # 底部控制栏 (向左向右按键)
        control_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_prev.clicked.connect(self.show_prev)

        self.lbl_info = QLabel()
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_info.setStyleSheet("font-size: 16px; font-weight: bold; color: #A6E3A1;")

        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_next.clicked.connect(self.show_next)

        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.lbl_info)
        control_layout.addWidget(self.btn_next)
        layout.addLayout(control_layout)

        # 重写滚轮事件用于 Ctrl+滚轮 缩放
        self.view.wheelEvent = self.zoom_event

        self.load_current_image()

    def load_current_image(self):
        if 0 <= self.current_index < len(self.image_paths):
            pixmap = QPixmap(self.image_paths[self.current_index])
            self.pixmap_item.setPixmap(pixmap)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            # 自适应窗口大小
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

            self.lbl_info.setText(f"第 {self.current_index + 1} 帧 / 共 {len(self.image_paths)} 帧")
            self.btn_prev.setEnabled(self.current_index > 0)
            self.btn_next.setEnabled(self.current_index < len(self.image_paths) - 1)

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def show_next(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_current_image()

    def zoom_event(self, event):
        # 必须按住 Ctrl 键才能滚动缩放，否则是上下平移
        if QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier:
            zoom_in_factor = 1.15                     # 参数：Ctrl+滚轮放大的倍率
            zoom_out_factor = 1 / zoom_in_factor

            # 滚轮向上放大，向下缩小
            if event.angleDelta().y() > 0:
                self.view.scale(zoom_in_factor, zoom_in_factor)
            else:
                self.view.scale(zoom_out_factor, zoom_out_factor)
        else:
            # 调用原生的滚轮事件（页面上下滚动）
            QGraphicsView.wheelEvent(self.view, event)


# ==========================================
# 2. 后台处理线程 (避免 UI 卡顿)
# ==========================================
class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    # 传递: score1, score2, deg1, deg2, frames_dir, out_video1, out_video2
    finished = pyqtSignal(float, float, float, float, str, str, str, float, float, float, float, str, float, str, float, str)
    error = pyqtSignal(str)

    def compute_knee_power(self, frame_metrics, fps=30.0):
        # 提取有效膝关节角度
        knee_angles = [m['angles'][3] for m in frame_metrics if m.get('angles') is not None]
        if len(knee_angles) < 5:
            return 0.0, "<div align='center'>膝关节数据不足，无法评估屈膝发力</div>"

        # 平滑数据以消除抖动噪点
        def smooth(data, window=5):
            if len(data) < window: return np.array(data)
            return np.convolve(data, np.ones(window) / window, mode='same')

        smoothed_knee = smooth(knee_angles)

        # 计算最低点(最大屈膝)与最高点(完全伸展)
        min_knee = np.min(smoothed_knee)
        max_knee = np.max(smoothed_knee)
        amplitude = max_knee - min_knee

        # 计算角速度 (度/秒) -> 利用帧差分乘以帧率
        # np.diff 计算相邻两帧的角度差，正值代表伸膝过程
        velocities = np.diff(smoothed_knee) * fps
        max_velocity = np.max(velocities) if len(velocities) > 0 else 0

        # ================= 评分逻辑 =================
        # 1. 屈伸幅度得分 (假设理想屈伸幅度在 65° - 85° 之间，完全站直约180°，下蹲至100°)
        amp_score = 100.0 - abs(amplitude - 75.0) * 1.5
        amp_score = max(0.0, min(100.0, amp_score))

        # 2. 爆发角速度得分 (假设最大角速度 > 350°/s 为极佳)
        vel_score = (max_velocity / 350.0) * 100.0
        vel_score = max(0.0, min(100.0, vel_score))

        # 综合得分 (权重各占 50%)
        total_score = (amp_score * 0.5) + (vel_score * 0.5)

        html_report = f"""
        <table width='100%' style='color:#A6ADC8; font-size: 14px;'>
            <tr style='color:#89B4FA;'>
                <th align='center'><b>评价指标</b></th>
                <th align='center'><b>实测数据</b></th>
                <th align='center'><b>单项得分</b></th>
            </tr>
            <tr><td align='center'>最低下蹲角度</td><td align='center'>{min_knee:.1f}°</td><td align='center'>-</td></tr>
            <tr><td align='center'>膝关节屈伸幅度</td><td align='center'>{amplitude:.1f}°</td><td align='center'>{amp_score:.1f}</td></tr>
            <tr><td align='center'>最大蹬伸角速度</td><td align='center'>{max_velocity:.1f}°/s</td><td align='center'>{vel_score:.1f}</td></tr>
        </table>
        """
        return total_score, html_report

    def compute_coordination(self, frame_metrics):
        if len(frame_metrics) < 10:
            return 0.0, "数据量过少，无法分析动力链"

        # 提取关键数据序列
        hip_angles = [m['angles'][2] if m['angles'] else 180 for m in frame_metrics]
        knee_angles = [m['angles'][3] if m['angles'] else 180 for m in frame_metrics]
        shoulder_angles = [m['angles'][0] if m['angles'] else 180 for m in frame_metrics]
        elbow_angles = [m['angles'][1] if m['angles'] else 180 for m in frame_metrics]

        # 获取手腕Y坐标(缺失则使用上一帧)
        wrist_ys = []
        for m in frame_metrics:
            w_y = m.get('wrist_y', None)
            if w_y is not None:
                wrist_ys.append(w_y)
            else:
                wrist_ys.append(wrist_ys[-1] if wrist_ys else 9999)

        # 核心：使用一维卷积进行滑动平均平滑，避免异常帧噪点干扰极值判断
        def smooth(data, window=5):                    #参数：值越大越平滑但可能会造成极值点偏移
            if len(data) < window: return np.array(data)
            return np.convolve(data, np.ones(window) / window, mode='same')

        # 计算极值帧所在索引 (角度求最大值，Y坐标求最小值即最高点)
        t_hip = int(np.argmax(smooth(hip_angles)))
        t_knee = int(np.argmax(smooth(knee_angles)))
        t_shoulder = int(np.argmax(smooth(shoulder_angles)))
        t_elbow = int(np.argmax(smooth(elbow_angles)))
        t_release = int(np.argmin(smooth(wrist_ys)))

        peaks = {
            "髋部伸展": t_hip,
            "膝部蹬伸": t_knee,
            "肩部发力": t_shoulder,
            "肘部传递": t_elbow,
            "手腕释放": t_release
        }

        # 计分逻辑：理想状态下应为从下到上的顺序。若出现“倒挂”(如上肢先于下肢发力)，则扣分。
        score = 100.0
        lags = [
            ("膝-髋 (下肢)", t_knee - t_hip),
            ("肩-膝 (躯干)", t_shoulder - t_knee),
            ("肘-肩 (上肢)", t_elbow - t_shoulder),
            ("腕-肘 (末端)", t_release - t_elbow)
        ]

        for name, lag in lags:
            if lag < 0:
                # 倒挂惩罚：每倒置1帧扣除 3 分
                score += lag * 3

        score = max(0.0, min(100.0, score))

        # 归一化时间比例 (将整个动作用时映射为 0% - 100%)
        t_start = min(peaks.values())
        t_end = max(peaks.values())
        total_time = t_end - t_start if t_end > t_start else 1

        html_rows = ""
        # 按发生时间排序
        sorted_peaks = sorted(peaks.items(), key=lambda x: x[1])
        for name, t in sorted_peaks:
            rel_pct = ((t - t_start) / total_time) * 100
            # 为每个 td 添加 align='center'
            html_rows += f"<tr><td align='center'>{name}</td><td align='center'>第 {t} 帧</td><td align='center'>{rel_pct:.1f}%</td></tr>"

        report = f"""
                <table width='100%' style='color:#A6ADC8; font-size: 14px;'>
                    <tr style='color:#89B4FA;'>
                        <th align='center'><b>动力链环节</b></th>
                        <th align='center'><b>达峰节点</b></th>
                        <th align='center'><b>相对总耗时比例</b></th>
                    </tr>
                    {html_rows}
                </table>
                """
        return score, report

    def __init__(self, test_video_path, standard_videos):
        super().__init__()
        self.test_video_path = test_video_path
        self.standard_videos = standard_videos
        self.det_model = None
        self.pose_model = None

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def compute_release_angle(self, frame_metrics):
        # 提取包含手腕y坐标和关键点数据的有效帧
        valid_frames = [m for m in frame_metrics if m.get('wrist_y') is not None and m.get('kpts') is not None]
        if not valid_frames:
            return 0.0, "<div align='center'>缺失手腕或关键点数据，无法评估出手角度</div>"

        # 找到手腕最高点（图像Y坐标最小）的帧，即出手瞬间
        release_frame = min(valid_frames, key=lambda x: x['wrist_y'])

        kpts = release_frame['kpts']
        side = release_frame.get('side_str', 'Right')

        # 确定手肘和手腕的索引 (YOLOv8 骨架：左手肘7, 左手腕9; 右手肘8, 右手腕10)
        if side == 'Right':
            e_idx, w_idx = 8, 10
        else:
            e_idx, w_idx = 7, 9

        elbow = kpts[e_idx]
        wrist = kpts[w_idx]

        if elbow[0] == 0 or wrist[0] == 0:
            return 0.0, "<div align='center'>出手瞬间手臂关键点被遮挡，无法计算角度</div>"

        # 计算绝对差值 (dx: 水平距离, dy: 垂直距离。注意图像坐标Y向下为正，需要翻转减法方向)
        dx = abs(wrist[0] - elbow[0])
        dy = elbow[1] - wrist[1]  # 手肘的Y 减去 手腕的Y (正常投篮时手腕在上方，此值为正)

        # 使用 numpy 的 arctan2 计算相对于水平线的角度
        if dx == 0 and dy == 0:
            angle = 0.0
        else:
            angle = float(np.degrees(np.arctan2(dy, dx)))

        # ================= 评分逻辑 =================
        # 假设理想的出手角度在 45° - 55° 之间，50° 为满分
        score = 100.0 - abs(angle - 50.0) * 2.5
        score = max(0.0, min(100.0, score))

        html_report = f"""
        <table width='100%' style='color:#A6ADC8; font-size: 14px;'>
            <tr style='color:#89B4FA;'>
                <th align='center'><b>评价指标</b></th>
                <th align='center'><b>实测角度</b></th>
                <th align='center'><b>单项得分</b></th>
            </tr>
            <tr>
                <td align='center'>出手小臂对地夹角</td>
                <td align='center'>{angle:.1f}°</td>
                <td align='center'>{score:.1f}</td>
            </tr>
        </table>
        """
        return score, html_report

    def compute_dtw_score(self, true_avg_degree):           #参数：DTW动作相似度评分公式
        if true_avg_degree <= 10.0:
            return 100.0
        elif true_avg_degree >= 55.0:
            return 20.0
        else:
            return 100 - (true_avg_degree - 10) * (80 / 45)

    def process_video(self, video_path, save_visuals=False, out_dir=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None, None, None, None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frame_metrics = []
        frame_idx = 0

        # 第一遍：追踪与分析
        while True:
            success, frame = cap.read()
            if not success: break

            # 1. 模型预测
            det_results = self.det_model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False, conf=0.45)         #参数：置信度可做调整，可以调高以减少误检，可以调低以找回目标
            # 参数：YOLO追踪器配置bytetrack.yaml
            pose_results = self.pose_model.predict(frame, verbose=False)

            current_data = {
                'idx': frame_idx, 'hip_y': None, 'angles': None, 'kpts': None,
                'cx1': 0, 'cy1': 0, 'player_box': None, 'ball_boxes': []
            }

            # ====== 步骤 A：先处理人与姿态 (获取 player) ======
            main_kpts = None
            if pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
                pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                max_area = 0
                best_idx = -1

                for i, box in enumerate(pose_boxes):
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_idx = i
                        current_data['player_box'] = (x1, y1, x2, y2)

                if best_idx != -1 and pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
                    main_kpts = pose_results[0].keypoints.xy[best_idx].cpu().numpy()
                    current_data['kpts'] = main_kpts

            # ====== 步骤 B：后处理篮球======
            if det_results[0].boxes is not None and len(det_results[0].boxes) > 0:
                boxes = det_results[0].boxes.xyxy.cpu().numpy()
                clses = det_results[0].boxes.cls.cpu().numpy()
                confs = det_results[0].boxes.conf.cpu().numpy()

                valid_balls = []
                for box, cls, conf in zip(boxes, clses, confs):
                    if int(cls) == 1:  # 是篮球
                        bx1, by1, bx2, by2 = map(int, box)
                        w, h = bx2 - bx1, by2 - by1
                        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

                        # 1. 基础尺寸过滤
                        if aspect_ratio < 2.5 and w < width * 0.3:                    # 参数：篮球宽高比限制，若视频分辨率或拍摄距离变化，可调节此比例

                            # 2. 空间逻辑过滤 (利用刚刚拿到的人体框)
                            if current_data['player_box'] is not None:
                                px1, py1, px2, py2 = current_data['player_box']
                                ball_cx = (bx1 + bx2) / 2
                                ball_cy = (by1 + by2) / 2
                                player_w = px2 - px1

                                # 限制 2a：球的中心 Y 轴不能太低
                                if ball_cy > py2 + 10:
                                    continue
                                # 限制 2b：球不能离球员太远
                                if ball_cx < px1 - player_w * 1.5 or ball_cx > px2 + player_w * 1.5:
                                    continue

                            # 通过所有考验，加入候选列表
                            valid_balls.append((conf, (bx1, by1, bx2, by2)))

                # 3. 终极绝杀：如果有多个候选球，只取置信度最高的那【1 个】
                if valid_balls:
                    valid_balls.sort(key=lambda x: x[0], reverse=True)
                    best_ball_box = valid_balls[0][1]
                    current_data['ball_boxes'].append(best_ball_box)

            if main_kpts is not None:
                abs_kpts = np.zeros_like(main_kpts)
                for i in range(len(main_kpts)):
                    if main_kpts[i][0] > 0:
                        abs_kpts[i] = [main_kpts[i][0] + current_data['cx1'], main_kpts[i][1] + current_data['cy1']]

                r_s, r_e, r_w = abs_kpts[6], abs_kpts[8], abs_kpts[10]
                l_s, l_e, l_w = abs_kpts[5], abs_kpts[7], abs_kpts[9]

                if r_w[0] > 0 and r_s[0] > 0:
                    s, e, w, h, k, a = r_s, r_e, r_w, abs_kpts[12], abs_kpts[14], abs_kpts[16]
                    side_str = "Right"
                else:
                    s, e, w, h, k, a = l_s, l_e, l_w, abs_kpts[11], abs_kpts[13], abs_kpts[15]
                    side_str = "Left"

                if s[0] > 0 and h[0] > 0:
                    shoulder = self.calculate_angle(h, s, e) if e[0] > 0 else 160.0
                    elbow = self.calculate_angle(s, e, w) if (e[0] > 0 and w[0] > 0) else 180.0
                    hip = self.calculate_angle(s, h, k) if k[0] > 0 else 180.0
                    knee = self.calculate_angle(h, k, a) if (k[0] > 0 and a[0] > 0) else 180.0

                    current_data['angles'] = [shoulder, elbow, hip, knee]
                    current_data['side_str'] = side_str

                    current_data['wrist_y'] = w[1] + current_data['cy1']
                    current_data['shoulder_y'] = s[1] + current_data['cy1']
                    if current_data['player_box'] is not None:
                        current_data['player_h'] = current_data['player_box'][3] - current_data['player_box'][1]

                hips = [main_kpts[11], main_kpts[12]]
                valid_hips_y = [pt[1] + current_data['cy1'] for pt in hips if pt[0] > 0]
                if valid_hips_y:
                    current_data['hip_y'] = sum(valid_hips_y) / len(valid_hips_y)

            frame_metrics.append(current_data)
            frame_idx += 1

        idx_squat = frame_idx // 2  # 设置默认备用切分点，以防未检测到过肩动作

        for m in frame_metrics:
            # 确保当前帧同时存在肩膀 Y 坐标和篮球检测框
            if m.get('shoulder_y') is not None and len(m.get('ball_boxes', [])) > 0:
                # 获取置信度最高的篮球检测框
                bx1, by1, bx2, by2 = m['ball_boxes'][0]
                shoulder_y = m['shoulder_y']

                # 图像坐标系中 Y 轴向下为正。
                # 篮球检测框底部的 y 坐标为 by2，如果 by2 小于 shoulder_y，
                # 说明篮球检测框的最底端都已经高于肩膀，即满足“四个点都在肩部之上”。
                if by2 < shoulder_y:
                    idx_squat = m['idx']  # 找到符合条件的这一帧，设为分界点
                    break  # 停止遍历，保证是抬手过肩的【第一帧】

        seq1, seq2 = [], []
        for m in frame_metrics:
            if m['angles'] is not None:
                if m['idx'] <= idx_squat:
                    seq1.append(m['angles'])
                else:
                    seq2.append(m['angles'])

        if not seq1: seq1 = [[180.0, 180.0, 180.0, 180.0], [180.0, 180.0, 180.0, 180.0]]
        if not seq2: seq2 = [[180.0, 180.0, 180.0, 180.0], [180.0, 180.0, 180.0, 180.0]]

        valid_height_frames = [m for m in frame_metrics if 'wrist_y' in m and 'player_h' in m]
        if valid_height_frames:
            start_wrist_y = valid_height_frames[0]['wrist_y']
            # Y轴向下为正，因此最小的 Y 才是物理上的最高点
            min_wrist_y = min(m['wrist_y'] for m in valid_height_frames)
            player_h = valid_height_frames[0]['player_h']
            rel_height = (start_wrist_y - min_wrist_y) / player_h if player_h > 0 else 0.0
        else:
            rel_height = 0.0

        out1_path, out2_path = None, None

        # 第二遍：渲染画面并分离视频
        if save_visuals and out_dir:
            os.makedirs(out_dir, exist_ok=True)

            for file_name in os.listdir(out_dir):
                if file_name.endswith('.jpg') or file_name.endswith('.mp4'):
                    try:
                        os.remove(os.path.join(out_dir, file_name))
                    except Exception:
                        pass

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            slow_fps = fps * 0.4              # 参数：用于实现慢动作回放
            out1_path = os.path.join(out_dir, "clip1_squat.mp4")
            out2_path = os.path.join(out_dir, "clip2_release.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out1 = cv2.VideoWriter(out1_path, fourcc, slow_fps, (width, height))
            out2 = cv2.VideoWriter(out2_path, fourcc, slow_fps, (width, height))

            skeleton_connections = [
                (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
            ]

            curr_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break

                if curr_idx < len(frame_metrics):
                    data = frame_metrics[curr_idx]

                    for bx in data.get('ball_boxes', []):
                        bx1, by1, bx2, by2 = bx
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
                        cv2.putText(frame, "Ball", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                    if data['player_box'] is not None:
                        px1, py1, px2, py2 = data['player_box']
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 144, 30), 2)
                        cv2.putText(frame, "Player", (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 144, 30), 2)

                    if data['kpts'] is not None:
                        cx1, cy1 = data['cx1'], data['cy1']
                        for p1_idx, p2_idx in skeleton_connections:
                            pt1, pt2 = data['kpts'][p1_idx], data['kpts'][p2_idx]
                            if pt1[0] == 0 or pt2[0] == 0: continue
                            cv2.line(frame, (int(pt1[0] + cx1), int(pt1[1] + cy1)),
                                     (int(pt2[0] + cx1), int(pt2[1] + cy1)), (220, 110, 0), 2)
                        for i, pt in enumerate(data['kpts']):
                            if pt[0] == 0: continue
                            color, r = ((0, 255, 255), 3) if i <= 4 else ((50, 255, 50), 5)
                            cv2.circle(frame, (int(pt[0] + cx1), int(pt[1] + cy1)), r, color, -1)

                    if data['angles'] is not None:
                        shoulder, elbow, hip, knee = data['angles']
                        texts = [
                            f"Phase: {'P1: Squat' if curr_idx <= idx_squat else 'P2: Release'}",
                            f"Side: {data.get('side_str', 'Unknown')}", f"Shoulder: {shoulder:.1f}",
                            f"Elbow: {elbow:.1f}", f"Hip: {hip:.1f}", f"Knee: {knee:.1f}"
                        ]
                        for i, txt in enumerate(texts):
                            cv2.putText(frame, txt, (20, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                if data['angles'] is not None:
                    shoulder, elbow, hip, knee = data['angles']
                    texts = [
                        f"Phase: {'P1: Squat' if curr_idx <= idx_squat else 'P2: Release'}",
                        f"Side: {data.get('side_str', 'Unknown')}", f"Shoulder: {shoulder:.1f}",
                        f"Elbow: {elbow:.1f}", f"Hip: {hip:.1f}", f"Knee: {knee:.1f}"
                    ]
                    for i, txt in enumerate(texts):
                        cv2.putText(frame, txt, (20, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                    # ==================== 右下角火柴人(骨架)实时演示 ====================
                if data['kpts'] is not None:
                    # 1. 设定右下角画板的大小和位置 (宽占20%，高占35%)
                    sm_w, sm_h = int(width * 0.20), int(height * 0.35)
                    sm_x1, sm_y1 = width - sm_w - 20, height - sm_h - 20

                    # 2. 绘制半透明黑色背景板，避免背景杂乱干扰视觉
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (sm_x1, sm_y1), (sm_x1 + sm_w, sm_y1 + sm_h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, "Pose", (sm_x1 + 10, sm_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200),
                                2)

                    # 3. 提取有效关键点并计算其实际人体边界，用于后续动态缩放
                    valid_pts = [pt for pt in data['kpts'] if pt[0] > 0]
                    if len(valid_pts) > 0:
                        min_x = min(pt[0] for pt in valid_pts)
                        max_x = max(pt[0] for pt in valid_pts)
                        min_y = min(pt[1] for pt in valid_pts)
                        max_y = max(pt[1] for pt in valid_pts)

                        skel_w = max_x - min_x + 1e-5
                        skel_h = max_y - min_y + 1e-5

                        # 计算缩放比例（保留一定内边距，保持宽高比）
                        padding = 25                # 参数：避免火柴人骨架贴着边缘
                        scale = min((sm_w - 2 * padding) / skel_w, (sm_h - 2 * padding) / skel_h)

                        def get_sm_pt(pt):
                            """闭包函数：将原图绝对坐标转化为右下角小黑板的相对缩放坐标"""
                            if pt[0] == 0: return None
                            # 利用中心点对齐进行计算，让火柴人始终居中于黑板
                            nx = sm_x1 + sm_w / 2 + (pt[0] - (min_x + skel_w / 2)) * scale
                            ny = sm_y1 + sm_h / 2 + (pt[1] - (min_y + skel_h / 2)) * scale
                            return (int(nx), int(ny))

                        # 4. 绘制火柴人骨架连接线 (白色)
                        for p1_idx, p2_idx in skeleton_connections:
                            pt1, pt2 = data['kpts'][p1_idx], data['kpts'][p2_idx]
                            sm_pt1 = get_sm_pt(pt1)
                            sm_pt2 = get_sm_pt(pt2)
                            if sm_pt1 and sm_pt2:
                                cv2.line(frame, sm_pt1, sm_pt2, (255, 255, 255), 2)

                                # 5. 绘制火柴人关节节点 (黄色)
                        for pt in data['kpts']:
                            sm_pt = get_sm_pt(pt)
                            if sm_pt:
                                cv2.circle(frame, sm_pt, 3, (0, 255, 255), -1)

                if curr_idx <= idx_squat:
                    out1.write(frame)
                else:
                    out2.write(frame)

                cv2.imwrite(os.path.join(out_dir, f"frame_{curr_idx:04d}.jpg"), frame)
                curr_idx += 1

            out1.release()
            out2.release()

        cap.release()
        s1 = np.array(seq1)
        s2 = np.array(seq2)
        return s1, s2, out1_path, out2_path, rel_height, frame_metrics

    def run(self):
        try:
            self.progress_updated.emit(5)
            self.det_model = YOLO("runs/detect/train/weights/best.pt")          # 参数：YOLO模型路径
            self.pose_model = YOLO("yolov8n-pose.pt")                           # 参数：姿态模型
            self.progress_updated.emit(15)

            std_seqs1, std_seqs2, std_heights = [], [], []
            for i, path in enumerate(self.standard_videos):
                s1, s2, _, _, rel_h, _ = self.process_video(path.strip())
                if s1 is not None and len(s1) > 2: std_seqs1.append(s1)
                if s2 is not None and len(s2) > 2: std_seqs2.append(s2)
                if rel_h > 0: std_heights.append(rel_h)
                self.progress_updated.emit(15 + int(35 * (i / len(self.standard_videos))))

            # 计算标准视频的平均相对高度
            avg_std_height = sum(std_heights) / len(std_heights) if std_heights else 0.5

            if not std_seqs1 or not std_seqs2:
                raise ValueError("标准视频库解析失败，无法提取两段动作特征！")

            min_dist1, champ1 = float('inf'), std_seqs1[0]
            for i, sq_a in enumerate(std_seqs1):
                tot = sum(fastdtw(sq_a, sq_b, dist=euclidean)[0] / max(len(sq_a), len(sq_b)) for j, sq_b in
                          enumerate(std_seqs1) if i != j)
                if len(std_seqs1) > 1 and tot / (len(std_seqs1) - 1) < min_dist1:
                    min_dist1, champ1 = tot / (len(std_seqs1) - 1), sq_a

            min_dist2, champ2 = float('inf'), std_seqs2[0]
            for i, sq_a in enumerate(std_seqs2):
                tot = sum(fastdtw(sq_a, sq_b, dist=euclidean)[0] / max(len(sq_a), len(sq_b)) for j, sq_b in
                          enumerate(std_seqs2) if i != j)
                if len(std_seqs2) > 1 and tot / (len(std_seqs2) - 1) < min_dist2:
                    min_dist2, champ2 = tot / (len(std_seqs2) - 1), sq_a

            self.progress_updated.emit(60)

            out_folder = "chuxuezhe_clip_result"
            test_s1, test_s2, out_v1, out_v2, test_rel_h, test_metrics = self.process_video(
                self.test_video_path, save_visuals=True, out_dir=out_folder)

            if test_s1 is None or test_s2 is None:
                raise ValueError("测试视频未检测到完整的下蹲和出手动作！")

                # 计算高度差异得分
            height_diff = abs(test_rel_h - avg_std_height)
            height_score = max(0.0, min(100.0, 100.0 - height_diff * 150.0))            #参数：150.0是一个惩罚放大系数：数值越大，对出手高度不足的扣分越严重

            # 计算两段动作的 DTW 得分
            d1, p1 = fastdtw(champ1, test_s1, dist=euclidean)
            deg1 = (d1 / len(p1)) / np.sqrt(4)
            score1 = self.compute_dtw_score(deg1)

            d2, p2 = fastdtw(champ2, test_s2, dist=euclidean)
            deg2 = (d2 / len(p2)) / np.sqrt(4)
            score2 = self.compute_dtw_score(deg2)

            coord_score, coord_report = self.compute_coordination(test_metrics)

            knee_score, knee_report = self.compute_knee_power(test_metrics, fps=30.0)

            release_score, release_report = self.compute_release_angle(test_metrics)
            # 正确：所有变量都计算完毕后，统一发送一次带有 10 个参数的完整信号
            self.progress_updated.emit(100)
            self.finished.emit(score1, score2, deg1, deg2, out_folder, out_v1, out_v2,
                               height_score, test_rel_h, avg_std_height,
                               coord_score, coord_report, knee_score, knee_report,
                               release_score, release_report)

        except Exception as e:
            self.error.emit(str(e))


# ==========================================
# 3. PyQt6 界面主程序
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能体育评测系统 - 双阶段增强版")
        self.resize(1100, 900)
        self.video_path = None
        self.image_files_paths = []

        raw_str = "./left_shot_clips/shot_2.mp4,./left_shot_clips/shot_3.mp4,./left_shot_clips/shot_4.mp4,./left_shot_clips/shot_5.mp4,./left_shot_clips/shot_6.mp4"
        # 参数：标准动作参考视频库
        self.standard_videos = raw_str.split(",")

        # 视频播放状态控制
        self.is_playing1 = False
        self.is_playing2 = False

        self.setup_ui()
        self.setup_style()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.lbl_title = QLabel("🏀 投篮动作智能评分系统 (二段式评测)")
        self.lbl_title.setObjectName("title")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_title)

        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("📁 选择测试视频")
        self.btn_select.clicked.connect(self.select_video)
        self.btn_start = QPushButton("🚀 开始分段智能评分")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setEnabled(False)
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_start)
        layout.addLayout(btn_layout)

        self.lbl_path = QLabel("尚未选择视频")
        self.lbl_path.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_path.setStyleSheet("color: #aaa;")
        layout.addWidget(self.lbl_path)

        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)

        # 视频标题
        self.lbl_video_title = QLabel("▶️ 分段姿态跟踪 (准备-下蹲 vs 蹬伸-出手)")
        self.lbl_video_title.setObjectName("subtitle")
        self.lbl_video_title.hide()
        self.scroll_layout.addWidget(self.lbl_video_title)

        # ================= 视频画面、控制及进度条展示 =================
        video_layout = QHBoxLayout()

        # --- 视频1 容器 ---
        v1_container = QWidget()
        v1_layout = QVBoxLayout(v1_container)
        v1_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_video1 = QLabel()
        self.lbl_video1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video1.setStyleSheet("background-color: #000; border-radius: 10px;")
        self.lbl_video1.setMinimumSize(480, 360)

        # 进度条1
        self.slider1 = QSlider(Qt.Orientation.Horizontal)
        self.slider1.setObjectName("video_slider")
        self.slider1.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider1.sliderMoved.connect(self.on_slider1_moved)

        self.btn_play1 = QPushButton("▶️")
        self.btn_play1.setObjectName("icon_btn")
        self.btn_play1.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_play1.clicked.connect(self.toggle_video1)

        v1_ctrl_layout = QHBoxLayout()
        v1_ctrl_layout.addWidget(self.btn_play1)
        v1_ctrl_layout.addWidget(self.slider1)

        v1_layout.addWidget(self.lbl_video1)
        v1_layout.addLayout(v1_ctrl_layout)

        # --- 视频2 容器 ---
        v2_container = QWidget()
        v2_layout = QVBoxLayout(v2_container)
        v2_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_video2 = QLabel()
        self.lbl_video2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video2.setStyleSheet("background-color: #000; border-radius: 10px;")
        self.lbl_video2.setMinimumSize(480, 360)

        # 进度条2
        self.slider2 = QSlider(Qt.Orientation.Horizontal)
        self.slider2.setObjectName("video_slider")
        self.slider2.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider2.sliderMoved.connect(self.on_slider2_moved)

        self.btn_play2 = QPushButton("▶️")
        self.btn_play2.setObjectName("icon_btn")
        self.btn_play2.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_play2.clicked.connect(self.toggle_video2)

        v2_ctrl_layout = QHBoxLayout()
        v2_ctrl_layout.addWidget(self.btn_play2)
        v2_ctrl_layout.addWidget(self.slider2)

        v2_layout.addWidget(self.lbl_video2)
        v2_layout.addLayout(v2_ctrl_layout)

        video_layout.addWidget(v1_container)
        video_layout.addWidget(v2_container)

        self.video_container = QWidget()
        self.video_container.setLayout(video_layout)
        self.video_container.hide()
        self.scroll_layout.addWidget(self.video_container)

        # 帧图
        self.lbl_frames_title = QLabel("🎞️ 逐帧动作分析 (👉 点击图片可放大并交互)")
        self.lbl_frames_title.setObjectName("subtitle")
        self.lbl_frames_title.hide()
        self.scroll_layout.addWidget(self.lbl_frames_title)

        self.frames_scroll = QScrollArea()
        self.frames_scroll.setFixedHeight(220)
        self.frames_scroll.setWidgetResizable(True)
        self.frames_widget = QWidget()
        self.frames_layout = QHBoxLayout(self.frames_widget)
        self.frames_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.frames_scroll.setWidget(self.frames_widget)
        self.frames_scroll.hide()
        self.scroll_layout.addWidget(self.frames_scroll)

        # 评分
        self.lbl_score_title = QLabel("🏆 双阶段综合评测报告")
        self.lbl_score_title.setObjectName("subtitle")
        self.lbl_score_title.hide()
        self.scroll_layout.addWidget(self.lbl_score_title)

        self.lbl_score_result = QLabel("")
        self.lbl_score_result.setObjectName("score_box")
        self.lbl_score_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_score_result.hide()
        self.scroll_layout.addWidget(self.lbl_score_result)

        self.lbl_height_score = QLabel("")
        self.lbl_height_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_height_score.hide()
        self.scroll_layout.addWidget(self.lbl_height_score)

        self.lbl_coord_score = QLabel("")
        self.lbl_coord_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_coord_score.hide()
        self.scroll_layout.addWidget(self.lbl_coord_score)

        self.scroll_layout.addStretch()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_videos)
        self.cap1 = None
        self.cap2 = None

        self.lbl_knee_score = QLabel("")
        self.lbl_knee_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_knee_score.hide()
        self.scroll_layout.addWidget(self.lbl_knee_score)

        self.lbl_release_score = QLabel("")
        self.lbl_release_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_release_score.hide()
        self.scroll_layout.addWidget(self.lbl_release_score)

        self.scroll_layout.addStretch()

    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E2E; }
            QWidget { background-color: #1E1E2E; color: #CDD6F4; font-family: "Microsoft YaHei", sans-serif; }
            QLabel#title { font-size: 32px; font-weight: bold; color: #A6E3A1; margin-bottom: 10px; }
            QLabel#subtitle { font-size: 18px; font-weight: bold; color: #89B4FA; margin-top: 20px; }
            QLabel#score_box { background-color: #313244; padding: 20px; border-radius: 10px; }
            QPushButton { background-color: #89B4FA; color: #1E1E2E; font-weight: bold; border-radius: 8px; padding: 10px; font-size: 15px; }
            QPushButton:hover { background-color: #74C7EC; }
            QPushButton:disabled { background-color: #45475A; color: #6C7086; }
            QPushButton#icon_btn { background-color: #313244; color: #A6E3A1; border-radius: 20px; font-size: 18px; min-width: 40px; min-height: 40px; max-width: 40px; max-height: 40px; padding: 0px; margin-top: 5px; }
            QPushButton#icon_btn:hover { background-color: #45475A; }
            QProgressBar { border: 2px solid #45475A; border-radius: 5px; text-align: center; color: white; font-weight: bold; }
            QProgressBar::chunk { background-color: #A6E3A1; }
            QScrollArea { border: none; background-color: transparent; }

            /* 视频拖拽进度条样式 */
            QSlider#video_slider::groove:horizontal {
                border-radius: 4px;
                height: 8px;
                margin: 0px;
                background-color: #313244;
            }
            QSlider#video_slider::handle:horizontal {
                background-color: #A6E3A1;
                border: none;
                height: 16px;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider#video_slider::handle:horizontal:hover {
                background-color: #89B4FA;
            }
        """)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择测试视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            self.lbl_path.setText(f"已选择: {path}")
            self.btn_start.setEnabled(True)

    def start_processing(self):
        self.btn_start.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        self.lbl_video_title.hide()
        self.video_container.hide()
        self.lbl_frames_title.hide()
        self.frames_scroll.hide()
        self.lbl_score_title.hide()
        self.lbl_score_result.hide()
        self.lbl_height_score.hide()

        for i in reversed(range(self.frames_layout.count())):
            self.frames_layout.itemAt(i).widget().setParent(None)

        if self.cap1: self.cap1.release()
        if self.cap2: self.cap2.release()
        self.timer.stop()

        self.thread = ProcessingThread(self.video_path, self.standard_videos)
        self.thread.progress_updated.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.error.connect(self.on_processing_error)
        self.thread.start()

    def show_image_dialog(self, index):
        """点击缩略图放大查看的弹窗函数"""
        dialog = ImageViewerDialog(self.image_files_paths, index, self)
        dialog.exec()

    def on_processing_finished(self, score1, score2, deg1, deg2, frames_dir, out_vid1, out_vid2,
                               height_score, test_rel_h, avg_std_height,
                               coord_score, coord_report, knee_score, knee_report,
                               release_score, release_report):
        self.progress_bar.hide()
        self.btn_start.setEnabled(True)
        self.btn_select.setEnabled(True)

        self.lbl_video_title.show()
        self.video_container.show()

        self.cap1 = cv2.VideoCapture(out_vid1)
        self.cap2 = cv2.VideoCapture(out_vid2)

        # 设置视频进度条最大值
        if self.cap1.isOpened():
            total_frames1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider1.setRange(0, max(0, total_frames1 - 1))
        if self.cap2.isOpened():
            total_frames2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider2.setRange(0, max(0, total_frames2 - 1))

        self.is_playing1 = False
        self.is_playing2 = False
        self.btn_play1.setText("▶️")
        self.btn_play2.setText("▶️")

        self.read_first_frames()

        self.lbl_frames_title.show()
        self.frames_scroll.show()

        # 缓存所有逐帧图路径用于画廊式查看
        image_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        self.image_files_paths = [os.path.join(frames_dir, f) for f in image_files]

        for idx, img_path in enumerate(self.image_files_paths):
            pixmap = QPixmap(img_path).scaledToHeight(180, Qt.TransformationMode.SmoothTransformation)
            img_label = ClickableLabel(idx)  # 这里传入索引而非路径
            img_label.setPixmap(pixmap)
            img_label.setStyleSheet(
                "QLabel { border: 2px solid #45475A; border-radius: 5px; } QLabel:hover { border: 2px solid #A6E3A1; }")
            img_label.clicked.connect(self.show_image_dialog)
            self.frames_layout.addWidget(img_label)

        self.lbl_score_title.show()
        self.lbl_score_result.show()
        self.lbl_height_score.show()

        final_score = (score1 + score2) / 2
        color = "#A6E3A1" if final_score > 80 else ("#F9E2AF" if final_score > 60 else "#F38BA8")

        self.lbl_score_result.setStyleSheet(
            f"background-color: #313244; padding: 20px; font-size: 20px; border-radius: 10px; border: 2px solid {color};")
        self.lbl_score_result.setText(
            f"<div align='center'>🎯 <b>综合总得分：<font color='{color}' size='6'>{final_score:.1f} / 100</font></b><br><br></div>"
            f"<table width='100%' style='color:#A6ADC8;'>"
            f"<tr><td align='center'><b>阶段1：准备-下蹲</b></td><td align='center'><b>阶段2：蹬伸-出手</b></td></tr>"
            f"<tr><td align='center'>得分：{score1:.1f}</td><td align='center'>得分：{score2:.1f}</td></tr>"
            f"</table>"
        )

        h_color = "#A6E3A1" if height_score > 80 else ("#F9E2AF" if height_score > 60 else "#F38BA8")
        self.lbl_height_score.setStyleSheet(
            f"background-color: #1E1E2E; padding: 15px; font-size: 16px; border-radius: 10px; border: 1px dashed {h_color}; margin-top: 15px;"
        )
        self.lbl_height_score.setText(
            f"<div align='center'>📏 <b>出手高度专项评估：<font color='{h_color}' size='5'>{height_score:.1f} / 100</font></b><br><br></div>"
            f"<table width='100%' style='color:#A6ADC8;'>"
            f"<tr><td align='center'>测试者相对出手高度：<b>{test_rel_h:.2f}</b></td>"
            f"<td align='center'>标准参考相对高度：<b>{avg_std_height:.2f}</b></td></tr>"
            f"</table>"
            f"<div align='center' style='margin-top:10px; font-size:13px; color:#89B4FA;'>"
            f"<i>* 相对值 = (手腕最高点位移) / 身体像素身高。<br>此项为独立指标补充参考，不计入总分。</i></div>"
        )

        self.lbl_coord_score.show()

        c_color = "#A6E3A1" if coord_score > 80 else ("#F9E2AF" if coord_score > 60 else "#F38BA8")
        self.lbl_coord_score.setStyleSheet(
            f"background-color: #313244; padding: 15px; border-radius: 10px; border: 2px solid {c_color}; margin-top: 15px;"
        )
        self.lbl_coord_score.setText(
            f"<div align='center'>🔗 <b>动力链协同与发力节奏：<font color='{c_color}' size='5'>{coord_score:.1f} / 100</font></b><br><br></div>"
            f"{coord_report}"
            f"<div align='center' style='margin-top:10px; font-size:13px; color:#89B4FA;'>"
            f"<i>* 评估标准：能量应从下肢平顺传导至末端。发力节点若出现明显倒置或断档将被扣分。</i></div>"
        )

        self.lbl_knee_score.show()
        k_color = "#A6E3A1" if knee_score > 80 else ("#F9E2AF" if knee_score > 60 else "#F38BA8")

        self.lbl_knee_score.setStyleSheet(
            f"background-color: #313244; padding: 15px; border-radius: 10px; border: 2px solid {k_color}; margin-top: 15px;"
        )
        self.lbl_knee_score.setText(
            f"<div align='center'>🦵 <b>屈膝发力与爆发性：<font color='{k_color}' size='5'>{knee_score:.1f} / 100</font></b><br><br></div>"
            f"{knee_report}"
            f"<div align='center' style='margin-top:10px; font-size:13px; color:#89B4FA;'>"
            f"<i>* 评估标准：合理的下蹲幅度与快速的蹬伸角速度是提供投篮爆发力的核心保障。</i></div>"
        )

        self.lbl_release_score.show()
        r_color = "#A6E3A1" if release_score > 80 else ("#F9E2AF" if release_score > 60 else "#F38BA8")

        self.lbl_release_score.setStyleSheet(
            f"background-color: #313244; padding: 15px; border-radius: 10px; border: 2px solid {r_color}; margin-top: 15px;"
        )
        self.lbl_release_score.setText(
            f"<div align='center'>📐 <b>出手角度评估：<font color='{r_color}' size='5'>{release_score:.1f} / 100</font></b><br><br></div>"
            f"{release_report}"
            f"<div align='center' style='margin-top:10px; font-size:13px; color:#89B4FA;'>"
            f"<i>* 评估标准：计算出手瞬间小臂相对于地面的夹角。理想的投篮弧线通常需要 45° - 55° 之间的出手角度。</i></div>"
        )

    # =============== 独立播放器控制逻辑 ===============
    def read_first_frames(self):
        if self.cap1 and self.cap1.isOpened():
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = self.cap1.read()
            if ret1: self.lbl_video1.setPixmap(self.cv2_to_qpixmap(frame1, self.lbl_video1))
            self.slider1.setValue(0)

        if self.cap2 and self.cap2.isOpened():
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = self.cap2.read()
            if ret2: self.lbl_video2.setPixmap(self.cv2_to_qpixmap(frame2, self.lbl_video2))
            self.slider2.setValue(0)

    def toggle_video1(self):
        if not self.cap1: return
        if self.is_playing1:
            self.is_playing1 = False
            self.btn_play1.setText("▶️")
        else:
            if self.cap1.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap1.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_playing1 = True
            self.btn_play1.setText("⏸️")
            if not self.timer.isActive():
                self.timer.start(60)                  # 参数：每60ms刷新一帧

    def toggle_video2(self):
        if not self.cap2: return
        if self.is_playing2:
            self.is_playing2 = False
            self.btn_play2.setText("▶️")
        else:
            if self.cap2.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap2.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_playing2 = True
            self.btn_play2.setText("⏸️")
            if not self.timer.isActive():
                self.timer.start(60)                  # 参数：每60s刷新一帧

    def on_slider1_moved(self, position):
        """用户拖动进度条1时更新画面"""
        if self.cap1:
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = self.cap1.read()
            if ret:
                self.lbl_video1.setPixmap(self.cv2_to_qpixmap(frame, self.lbl_video1))

    def on_slider2_moved(self, position):
        """用户拖动进度条2时更新画面"""
        if self.cap2:
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = self.cap2.read()
            if ret:
                self.lbl_video2.setPixmap(self.cv2_to_qpixmap(frame, self.lbl_video2))

    def update_videos(self):
        if self.is_playing1 and self.cap1:
            ret1, frame1 = self.cap1.read()
            if ret1:
                self.lbl_video1.setPixmap(self.cv2_to_qpixmap(frame1, self.lbl_video1))
                # 更新滑块位置 (屏蔽信号避免陷入死循环)
                self.slider1.blockSignals(True)
                self.slider1.setValue(int(self.cap1.get(cv2.CAP_PROP_POS_FRAMES)))
                self.slider1.blockSignals(False)
            else:
                self.is_playing1 = False
                self.btn_play1.setText("▶️")
                self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.slider1.setValue(0)

        if self.is_playing2 and self.cap2:
            ret2, frame2 = self.cap2.read()
            if ret2:
                self.lbl_video2.setPixmap(self.cv2_to_qpixmap(frame2, self.lbl_video2))
                self.slider2.blockSignals(True)
                self.slider2.setValue(int(self.cap2.get(cv2.CAP_PROP_POS_FRAMES)))
                self.slider2.blockSignals(False)
            else:
                self.is_playing2 = False
                self.btn_play2.setText("▶️")
                self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.slider2.setValue(0)

        if not self.is_playing1 and not self.is_playing2:
            self.timer.stop()

    def cv2_to_qpixmap(self, frame, label):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qt_img).scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def on_processing_error(self, err_msg):
        self.progress_bar.hide()
        self.btn_start.setEnabled(True)
        self.btn_select.setEnabled(True)
        QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{err_msg}")

    def closeEvent(self, event):
        if self.cap1: self.cap1.release()
        if self.cap2: self.cap2.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())