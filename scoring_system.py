import os
import cv2
import numpy as np
import sys
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from ultralytics import YOLO
import requests  # 用于发起 豆包 API 请求
import json      # 用于处理 JSON 数据

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

        # Ctrl+滚轮 缩放
        self.view.wheelEvent = self.zoom_event

        self.load_current_image()

    def load_current_image(self):
        if 0 <= self.current_index < len(self.image_paths):
            pixmap = QPixmap(self.image_paths[self.current_index])
            self.pixmap_item.setPixmap(pixmap)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
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
            zoom_in_factor = 1.15  # 参数：Ctrl+滚轮放大的倍率
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
    finished = pyqtSignal(float, float, float, float, str, str, str, float, float, float, float, str, float, str, float,
                          str, float, str, str)
    error = pyqtSignal(str)

    def compute_completeness(self, frame_metrics, fps=30.0):
        """
        动作完成度专项评估模块
        结合髋关节纵向坐标的[显著下压]判定下蹲蓄力、[反向回弹]判定蹬伸发力
        """
        total_frames = len(frame_metrics)
        duration = total_frames / fps if fps > 0 else 0.0

        # 1. 提取纵向重心特征 (髋关节中心 Y 坐标) 及 球员像素高度
        hip_ys = [m['hip_y'] for m in frame_metrics if m.get('hip_y') is not None]
        player_heights = [m['player_h'] for m in frame_metrics if m.get('player_h') is not None]
        avg_h = np.mean(player_heights) if player_heights else 300.0

        # 2. 提取髋关节和膝关节角度序列
        hip_angles = [m['angles'][2] for m in frame_metrics if m.get('angles') is not None]
        knee_angles = [m['angles'][3] for m in frame_metrics if m.get('angles') is not None]

        has_squat = False
        has_extension = False
        has_release = False

        # 内部一维滑动平均平滑函数，滤除追踪抖动
        def smooth(data, window=5):
            if len(data) < window: return np.array(data)
            return np.convolve(data, np.ones(window) / window, mode='same')

        # 执行复合形态学判定
        if len(hip_ys) > 5:
            smoothed_hips = smooth(hip_ys)

            # 找到全视频中髋关节纵向的最低点（即图像坐标系中 Y 的最大值）
            max_y_idx = np.argmax(smoothed_hips)
            max_y_val = smoothed_hips[max_y_idx]

            # 计算从视频开始到最低点期间，身体曾达到的最高位置（Y 的最小值，通常是准备站立姿态）
            min_y_before = np.min(smoothed_hips[:max_y_idx + 1]) if max_y_idx > 0 else smoothed_hips[0]

            # 计算从最低点到视频结束期间，身体向上的反向腾跃/展直高度（Y 的最小值）
            min_y_after = np.min(smoothed_hips[max_y_idx:]) if max_y_idx < len(smoothed_hips) - 1 else max_y_val

            # 归一化：计算重心下蹲和向上反弹占总身高的比例
            drop_ratio = (max_y_val - min_y_before) / avg_h
            rise_ratio = (max_y_val - min_y_after) / avg_h

            # 获取全过程中的关节极限弯曲角度
            min_hip_angle = np.min(hip_angles) if hip_angles else 180.0
            min_knee_angle = np.min(knee_angles) if knee_angles else 180.0

            # -- 🎯 核心判别条件 1：下蹲蓄力环节 --
            is_coordinate_dropped = drop_ratio > 0.04
            is_direction_changed = (drop_ratio > 0.02) and (rise_ratio > 0.02)
            is_angle_flexed = (min_hip_angle < 160.0) or (min_knee_angle < 152.0)

            if is_coordinate_dropped or is_direction_changed or is_angle_flexed:
                has_squat = True

            # -- 🎯 核心判别条件 2：蹬伸发力环节 --
            post_min_knee = knee_angles[max_y_idx:] if knee_angles and max_y_idx < len(knee_angles) else []
            if post_min_knee and (np.max(post_min_knee) - np.min(knee_angles) > 15) and np.max(post_min_knee) > 155:
                has_extension = True
            elif rise_ratio > 0.05:
                has_extension = True
        else:
            if knee_angles and np.min(knee_angles) < 145: has_squat = True
            if knee_angles and np.max(knee_angles) - np.min(knee_angles) > 20: has_extension = True

        # -- 🎯 核心判别条件 3：出手释放环节 --
        for m in frame_metrics:
            if m.get('angles') is not None and m.get('wrist_y') is not None and m.get('shoulder_y') is not None:
                if m['wrist_y'] < m['shoulder_y'] and m['angles'][1] > 140:
                    has_release = True
                    break

        # 3. 生成可读性报告
        stages_status = []
        missing_details = []

        if has_squat:
            stages_status.append("<font color='#A6E3A1'><b>[已完成] 下蹲蓄力环节</b></font>")
        else:
            stages_status.append("<font color='#F38BA8'><b>[未检测到] 下蹲蓄力环节</b></font>")
            missing_details.append("❌ 缺乏有效的下蹲蓄力（髋关节重心无明显下压，或缺乏‘下蹲-蹬伸’的方向相反坐标衔接）")

        if has_extension:
            stages_status.append("<font color='#A6E3A1'><b>[已完成] 蹬伸发力环节</b></font>")
        else:
            stages_status.append("<font color='#F38BA8'><b>[未检测到] 蹬伸发力环节</b></font>")
            missing_details.append("❌ 缺乏蹬伸环节（重心最低点后未见身体及膝、髋关节有效向上延展）")

        if has_release:
            stages_status.append("<font color='#A6E3A1'><b>[已完成] 出手释放环节</b></font>")
        else:
            stages_status.append("<font color='#F38BA8'><b>[未检测到] 出手释放环节</b></font>")
            missing_details.append("❌ 缺乏出手环节（未见手腕举起过肩或肘关节未能有效伸直推球）")

        completed_count = sum([has_squat, has_extension, has_release])
        score = (completed_count / 3.0) * 100.0
        conclusion = "<font color='#A6E3A1'>🎉 恭喜！投篮核心技术环节完整，动作链衔接良好。</font>" if score == 100.0 else "<br>".join(
            missing_details)

        html_report = f"""
        <table width='100%' style='color:#A6ADC8; font-size: 14px;'>
            <tr style='color:#89B4FA;'>
                <th align='center' width='40%'><b>指标项</b></th>
                <th align='center' width='60%'><b>数据与判别结果</b></th>
            </tr>
            <tr>
                <td align='center'>⏱️ 动作完成时间</td>
                <td align='center'><b>{duration:.2f} 秒</b> (共计 {total_frames} 帧)</td>
            </tr>
            <tr>
                <td align='center'>🔍 核心技术环节检测</td>
                <td align='left' style='line-height:22px;'>
                    · {stages_status[0]}<br>
                    · {stages_status[1]}<br>
                    · {stages_status[2]}
                </td>
            </tr>
            <tr>
                <td align='center'>💡 针对性改进意见</td>
                <td align='left' style='color:#F9E2AF; line-height:20px;'>{conclusion}</td>
            </tr>
        </table>
        """
        return score, html_report

    def compute_knee_power(self, frame_metrics, fps=30.0):
        # --屈膝发力模块 结合髋关节纵向坐标的[显著下压]判定下蹲蓄力、[反向回弹]判定蹬伸发力--
        knee_angles = [m['angles'][3] for m in frame_metrics if m.get('angles') is not None]
        if len(knee_angles) < 5:
            return 0.0, "<div align='center'>膝关节数据不足，无法评估屈膝发力</div>"

        def smooth(data, window=5):
            if len(data) < window: return np.array(data)
            return np.convolve(data, np.ones(window) / window, mode='same')

        smoothed_knee = smooth(knee_angles)
        min_knee = np.min(smoothed_knee)
        max_knee = np.max(smoothed_knee)
        amplitude = max_knee - min_knee

        velocities = np.diff(smoothed_knee) * fps
        max_velocity = np.max(velocities) if len(velocities) > 0 else 0

        amp_score = 100.0 - abs(amplitude - 75.0) * 1.5
        amp_score = max(0.0, min(100.0, amp_score))

        vel_score = (max_velocity / 350.0) * 100.0
        vel_score = max(0.0, min(100.0, vel_score))

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
        # --动力链协同模块--
        if len(frame_metrics) < 10:
            return 0.0, "数据量过少，无法分析动力链"

        hip_angles = [m['angles'][2] if m['angles'] else 180 for m in frame_metrics]
        knee_angles = [m['angles'][3] if m['angles'] else 180 for m in frame_metrics]
        shoulder_angles = [m['angles'][0] if m['angles'] else 180 for m in frame_metrics]
        elbow_angles = [m['angles'][1] if m['angles'] else 180 for m in frame_metrics]

        wrist_ys = []
        for m in frame_metrics:
            w_y = m.get('wrist_y', None)
            if w_y is not None:
                wrist_ys.append(w_y)
            else:
                wrist_ys.append(wrist_ys[-1] if wrist_ys else 9999)

        def smooth(data, window=5):
            if len(data) < window: return np.array(data)
            return np.convolve(data, np.ones(window) / window, mode='same')

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

        score = max(0.0, min(100.0, score))

        t_start = min(peaks.values())
        t_end = max(peaks.values())
        total_time = t_end - t_start if t_end > t_start else 1

        html_rows = ""
        sorted_peaks = sorted(peaks.items(), key=lambda x: x[1])
        for name, t in sorted_peaks:
            rel_pct = ((t - t_start) / total_time) * 100
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
        self.current_fps = 30.0

    def calculate_angle(self, a, b, c):
        # --计算角度公式--
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def compute_release_angle(self, frame_metrics):
        # --出手角度--
        valid_frames = [m for m in frame_metrics if m.get('wrist_y') is not None and m.get('kpts') is not None]
        if not valid_frames:
            return 0.0, "<div align='center'>缺失手腕或关键点数据，无法评估出手角度</div>"

        release_frame = min(valid_frames, key=lambda x: x['wrist_y'])
        kpts = release_frame['kpts']
        side = release_frame.get('side_str', 'Right')

        if side == 'Right':
            e_idx, w_idx = 8, 10
        else:
            e_idx, w_idx = 7, 9

        elbow = kpts[e_idx]
        wrist = kpts[w_idx]

        if elbow[0] == 0 or wrist[0] == 0:
            return 0.0, "<div align='center'>出手瞬间手臂关键点被遮挡，无法计算角度</div>"

        dx = abs(wrist[0] - elbow[0])
        dy = elbow[1] - wrist[1]

        if dx == 0 and dy == 0:
            angle = 0.0
        else:
            angle = float(np.degrees(np.arctan2(dy, dx)))

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

    def compute_dtw_score(self, true_avg_degree):
        # --计算DTW分数--
        if true_avg_degree <= 10.0:
            return 100.0
        elif true_avg_degree >= 55.0:
            return 20.0
        else:
            return 100 - (true_avg_degree - 10) * (80 / 45)

    def process_video(self, video_path, save_visuals=False, out_dir=None):
        # 1. 读取视频，逐帧循环
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None, None, None, None, None, None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_fps = fps

        frame_metrics = []
        frame_idx = 0

        # 2.目标检测+跟踪（找球员、找篮球）+姿态
        while True:
            success, frame = cap.read()
            if not success: break

            det_results = self.det_model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False, conf=0.45)
            pose_results = self.pose_model.predict(frame, verbose=False)

            current_data = {
                'idx': frame_idx, 'hip_y': None, 'angles': None, 'kpts': None,
                'cx1': 0, 'cy1': 0, 'player_box': None, 'ball_boxes': []
            }

            main_kpts = None
            if pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
                pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                max_area = 0
                best_idx = -1

                # 3.筛选主球员
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

            if det_results[0].boxes is not None and len(det_results[0].boxes) > 0:
                boxes = det_results[0].boxes.xyxy.cpu().numpy()
                clses = det_results[0].boxes.cls.cpu().numpy()
                confs = det_results[0].boxes.conf.cpu().numpy()

                valid_balls = []
                for box, cls, conf in zip(boxes, clses, confs):
                    if int(cls) == 1:
                        bx1, by1, bx2, by2 = map(int, box)
                        w, h = bx2 - bx1, by2 - by1
                        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

                        if aspect_ratio < 2.5 and w < width * 0.3:
                            if current_data['player_box'] is not None:
                                px1, py1, px2, py2 = current_data['player_box']
                                ball_cx = (bx1 + bx2) / 2
                                ball_cy = (by1 + by2) / 2
                                player_w = px2 - px1

                                if ball_cy > py2 + 10: continue

                            valid_balls.append((conf, (bx1, by1, bx2, by2)))

                if valid_balls:
                    valid_balls.sort(key=lambda x: x[0], reverse=True)
                    best_ball_box = valid_balls[0][1]
                    current_data['ball_boxes'].append(best_ball_box)

            # 4.区分左右侧
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

        # 5.判断中间阶段
        idx_squat = frame_idx // 2

        for m in frame_metrics:
            if m.get('shoulder_y') is not None and len(m.get('ball_boxes', [])) > 0:
                bx1, by1, bx2, by2 = m['ball_boxes'][0]
                shoulder_y = m['shoulder_y']
                if by2 < shoulder_y:
                    idx_squat = m['idx']
                    break

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
            min_wrist_y = min(m['wrist_y'] for m in valid_height_frames)
            player_h = valid_height_frames[0]['player_h']
            rel_height = (start_wrist_y - min_wrist_y) / player_h if player_h > 0 else 0.0
        else:
            rel_height = 0.0

        out1_path, out2_path = None, None

        if save_visuals and out_dir:
            os.makedirs(out_dir, exist_ok=True)

            for file_name in os.listdir(out_dir):
                if file_name.endswith('.jpg') or file_name.endswith('.mp4'):
                    try:
                        os.remove(os.path.join(out_dir, file_name))
                    except Exception:
                        pass

            # 6.可视化输出
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            slow_fps = fps * 0.4
            out1_path = os.path.join(out_dir, "clip1_squat.mp4")
            out2_path = os.path.join(out_dir, "clip2_release.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out1 = cv2.VideoWriter(out1_path, fourcc, slow_fps, (480, 640))            # 输出分段视频的尺寸
            out2 = cv2.VideoWriter(out2_path, fourcc, slow_fps, (480, 640))

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

                if data['kpts'] is not None:
                    sm_w, sm_h = int(width * 0.20), int(height * 0.35)
                    sm_x1, sm_y1 = width - sm_w - 20, height - sm_h - 20

                    overlay = frame.copy()
                    cv2.rectangle(overlay, (sm_x1, sm_y1), (sm_x1 + sm_w, sm_y1 + sm_h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, "Pose", (sm_x1 + 10, sm_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200),
                                2)

                    valid_pts = [pt for pt in data['kpts'] if pt[0] > 0]
                    if len(valid_pts) > 0:
                        min_x = min(pt[0] for pt in valid_pts)
                        max_x = max(pt[0] for pt in valid_pts)
                        min_y = min(pt[1] for pt in valid_pts)
                        max_y = max(pt[1] for pt in valid_pts)

                        skel_w = max_x - min_x + 1e-5
                        skel_h = max_y - min_y + 1e-5

                        padding = 25
                        scale = min((sm_w - 2 * padding) / skel_w, (sm_h - 2 * padding) / skel_h)

                        def get_sm_pt(pt):
                            if pt[0] == 0: return None
                            nx = sm_x1 + sm_w / 2 + (pt[0] - (min_x + skel_w / 2)) * scale
                            ny = sm_y1 + sm_h / 2 + (pt[1] - (min_y + skel_h / 2)) * scale
                            return (int(nx), int(ny))

                        for p1_idx, p2_idx in skeleton_connections:
                            pt1, pt2 = data['kpts'][p1_idx], data['kpts'][p2_idx]
                            sm_pt1 = get_sm_pt(pt1)
                            sm_pt2 = get_sm_pt(pt2)
                            if sm_pt1 and sm_pt2:
                                cv2.line(frame, sm_pt1, sm_pt2, (255, 255, 255), 2)

                        for pt in data['kpts']:
                            sm_pt = get_sm_pt(pt)
                            if sm_pt:
                                cv2.circle(frame, sm_pt, 3, (0, 255, 255), -1)

                frame = cv2.resize(frame, (480, 640))
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
            self.det_model = YOLO("runs/detect/train/weights/best.pt")
            self.pose_model = YOLO("yolov8n-pose.pt")
            self.progress_updated.emit(15)

            std_seqs1, std_seqs2, std_heights = [], [], []
            for i, path in enumerate(self.standard_videos):
                s1, s2, _, _, rel_h, _ = self.process_video(path.strip())
                if s1 is not None and len(s1) > 2: std_seqs1.append(s1)
                if s2 is not None and len(s2) > 2: std_seqs2.append(s2)
                if rel_h > 0: std_heights.append(rel_h)
                self.progress_updated.emit(15 + int(35 * (i / len(self.standard_videos))))

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

            height_diff = abs(test_rel_h - avg_std_height)
            height_score = max(0.0, min(100.0, 100.0 - height_diff * 150.0))

            d1, p1 = fastdtw(champ1, test_s1, dist=euclidean)
            deg1 = (d1 / len(p1)) / np.sqrt(4)
            score1 = self.compute_dtw_score(deg1)

            d2, p2 = fastdtw(champ2, test_s2, dist=euclidean)
            deg2 = (d2 / len(p2)) / np.sqrt(4)
            score2 = self.compute_dtw_score(deg2)

            coord_score, coord_report = self.compute_coordination(test_metrics)
            video_fps = getattr(self, 'current_fps', 30.0)
            knee_score, knee_report = self.compute_knee_power(test_metrics, fps=video_fps)
            release_score, release_report = self.compute_release_angle(test_metrics)
            completeness_score, completeness_report = self.compute_completeness(test_metrics, fps=video_fps)

            # ================= 调用豆包 API 生成 AI 教练报告 =================
            self.progress_updated.emit(90)  # 提示用户正在生成 AI 报告

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

            ai_report = "AI 教练开小差了，未能生成报告。"
            try:
                api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "xxx"
                }
                payload = {
                    "model": "ep-m-20260720143111-jtzg2",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": ai_prompt, "type": "text"}]
                        }
                    ]
                }
                response = requests.post(api_url, headers=headers, json=payload, timeout=360)
                if response.status_code == 200:
                    resp_json = response.json()
                    ai_report = resp_json['choices'][0]['message']['content']
            except Exception as e:
                ai_report = f"AI 请求失败: {str(e)}"

            # 发送包含了 ai_report 的完整信号
            self.progress_updated.emit(100)
            self.finished.emit(score1, score2, deg1, deg2, out_folder, out_v1, out_v2,
                               height_score, test_rel_h, avg_std_height,
                               coord_score, coord_report, knee_score, knee_report,
                               release_score, release_report, completeness_score, completeness_report,
                               ai_report)

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

        raw_str = "./left_shot_clips/shot_2.mp4,./left_shot_clips/shot_3.mp4,./left_shot_clips/shot_4.mp4,./left_shot_clips/shot_5.mp4,./left_shot_clips/shot_6.mp4,./left_shot_clips/shot_7.mp4,./left_shot_clips/shot_8.mp4,./left_shot_clips/shot_9.mp4,./left_shot_clips/shot_10.mp4,./left_shot_clips/shot_11.mp4,./left_shot_clips/shot_11.mp4,./left_shot_clips/shot_12.mp4,./left_shot_clips/shot_13.mp4,./left_shot_clips/shot_14.mp4,./left_shot_clips/shot_15.mp4,./left_shot_clips/shot_16.mp4,./left_shot_clips/shot_17.mp4,./left_shot_clips/shot_18.mp4,./left_shot_clips/shot_19.mp4,./left_shot_clips/shot_20.mp4,./left_shot_clips/shot_21.mp4,./left_shot_clips/shot_22.mp4,./left_shot_clips/shot_23.mp4,./left_shot_clips/shot_24.mp4,./left_shot_clips/shot_25.mp4,./left_shot_clips/shot_26.mp4,./left_shot_clips/shot_27.mp4,./left_shot_clips/shot_28.mp4,./left_shot_clips/shot_29.mp4"
        self.standard_videos = raw_str.split(",")

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

        video_layout = QHBoxLayout()

        # --- 视频1 容器 ---
        v1_container = QWidget()
        v1_layout = QVBoxLayout(v1_container)
        v1_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_video1 = QLabel()
        self.lbl_video1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video1.setStyleSheet("background-color: #000; border-radius: 10px;")
        self.lbl_video1.setMinimumSize(480, 360)

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

        # 评分标题
        self.lbl_score_title = QLabel("🏆 双阶段综合评测报告")
        self.lbl_score_title.setObjectName("subtitle")
        self.lbl_score_title.hide()
        self.scroll_layout.addWidget(self.lbl_score_title)

        self.lbl_score_result = QLabel("")
        self.lbl_score_result.setObjectName("score_box")
        self.lbl_score_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_score_result.hide()
        self.scroll_layout.addWidget(self.lbl_score_result)

        self.lbl_completeness_score = QLabel("")
        self.lbl_completeness_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_completeness_score.hide()
        self.scroll_layout.addWidget(self.lbl_completeness_score)

        self.lbl_height_score = QLabel("")
        self.lbl_height_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_height_score.hide()
        self.scroll_layout.addWidget(self.lbl_height_score)

        self.lbl_coord_score = QLabel("")
        self.lbl_coord_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_coord_score.hide()
        self.scroll_layout.addWidget(self.lbl_coord_score)

        self.lbl_knee_score = QLabel("")
        self.lbl_knee_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_knee_score.hide()
        self.scroll_layout.addWidget(self.lbl_knee_score)

        self.lbl_release_score = QLabel("")
        self.lbl_release_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_release_score.hide()
        self.scroll_layout.addWidget(self.lbl_release_score)

        # AI 豆包教练点评组件
        self.lbl_ai_report = QLabel("")
        self.lbl_ai_report.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.lbl_ai_report.setWordWrap(True)
        self.lbl_ai_report.hide()
        self.scroll_layout.addWidget(self.lbl_ai_report)

        self.scroll_layout.addStretch()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_videos)
        self.cap1 = None
        self.cap2 = None

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
        self.lbl_ai_report.hide()  # 隐藏 AI 报告组件
        self.lbl_completeness_score.hide()
        self.lbl_height_score.hide()
        self.lbl_coord_score.hide()
        self.lbl_knee_score.hide()
        self.lbl_release_score.hide()

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
                               release_score, release_report, completeness_score, completeness_report,
                               ai_report):
        self.progress_bar.hide()
        self.btn_start.setEnabled(True)
        self.btn_select.setEnabled(True)

        self.lbl_video_title.show()
        self.video_container.show()

        self.cap1 = cv2.VideoCapture(out_vid1)
        self.cap2 = cv2.VideoCapture(out_vid2)

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

        image_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        self.image_files_paths = [os.path.join(frames_dir, f) for f in image_files]

        for idx, img_path in enumerate(self.image_files_paths):
            pixmap = QPixmap(img_path).scaledToHeight(180, Qt.TransformationMode.SmoothTransformation)
            img_label = ClickableLabel(idx)
            img_label.setPixmap(pixmap)
            img_label.setStyleSheet(
                "QLabel { border: 2px solid #45475A; border-radius: 5px; } QLabel:hover { border: 2px solid #A6E3A1; }")
            img_label.clicked.connect(self.show_image_dialog)
            self.frames_layout.addWidget(img_label)

        self.lbl_score_title.show()
        self.lbl_score_result.show()

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

        # 展示“动作完成度模块”
        self.lbl_completeness_score.show()
        comp_color = "#A6E3A1" if completeness_score == 100.0 else (
            "#F9E2AF" if completeness_score > 40.0 else "#F38BA8")
        self.lbl_completeness_score.setStyleSheet(
            f"background-color: #313244; padding: 15px; border-radius: 10px; border: 2px solid {comp_color}; margin-top: 15px;"
        )
        self.lbl_completeness_score.setText(
            f"<div align='center'>📋 <b>核心环节技术完整度：<font color='{comp_color}' size='5'>{completeness_score:.1f}%</font></b><br><br></div>"
            f"{completeness_report}"
        )

        self.lbl_height_score.show()
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

        # ======= 展示 AI 豆包教练点评 =======
        self.lbl_ai_report.show()
        self.lbl_ai_report.setStyleSheet(
            "background-color: #313244; padding: 18px; border-radius: 10px; border: 2px solid #89B4FA; margin-top: 15px;"
        )
        formatted_ai_text = ai_report.replace('\n', '<br>')
        self.lbl_ai_report.setText(
            f"<div align='center'>🤖 <b>AI 豆包大模型 - 智能教练指导意见：</b><br></div>"
            f"<div style='line-height:22px; margin-top:10px; color:#CDD6F4; font-size:14px;'>{formatted_ai_text}</div>"
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
                self.timer.start(60)

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
                self.timer.start(60)

    def on_slider1_moved(self, position):
        if self.cap1:
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = self.cap1.read()
            if ret:
                self.lbl_video1.setPixmap(self.cv2_to_qpixmap(frame, self.lbl_video1))

    def on_slider2_moved(self, position):
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