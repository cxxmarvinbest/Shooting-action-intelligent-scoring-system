import os
import cv2
import numpy as np
from fastdtw import fastdtw                           # 动态时间规整算法
from scipy.spatial.distance import euclidean          # 欧氏距离
from ultralytics import YOLO
import requests                                       # 用于发起HTTP请求
import json
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)                                 # 初始化Flask应用

# 配置文件夹
UPLOAD_FOLDER = 'uploads3'
CLIPS_FOLDER = 'shot_clips3'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)             # 如果目录不存在就创建
os.makedirs(CLIPS_FOLDER, exist_ok=True)

# 初始化模型 (全局加载，避免每次请求重复加载)
try:
    det_model = YOLO("runs/detect/train/weights/best.pt")
    pose_model = YOLO("yolov8n-pose.pt")
except Exception as e:
    print(f"模型加载失败: {e}")
    det_model, pose_model = None, None

class BasketballAnalyzer:
    def __init__(self, test_video_path):
        self.test_video_path = test_video_path
        raw_str = "./left_shot_clips/shot_2.mp4,./left_shot_clips/shot_3.mp4,./left_shot_clips/shot_4.mp4,./left_shot_clips/shot_5.mp4,./left_shot_clips/shot_6.mp4"
        self.standard_videos = raw_str.split(",")
        self.current_fps = 30.0                       #默认视频帧率

    def calculate_angle(self, a, b, c):                # 计算关节角度
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def compute_completeness(self, frame_metrics, fps=30.0):
        total_frames = len(frame_metrics)
        duration = total_frames / fps if fps > 0 else 0.0               # 计算动作总耗时
        
        hip_ys = [m['hip_y'] for m in frame_metrics if m.get('hip_y') is not None]
        player_heights = [m['player_h'] for m in frame_metrics if m.get('player_h') is not None]
        avg_h = np.mean(player_heights) if player_heights else 300.0
        hip_angles = [m['angles'][2] for m in frame_metrics if m.get('angles') is not None]
        knee_angles = [m['angles'][3] for m in frame_metrics if m.get('angles') is not None]

        has_squat, has_extension, has_release = False, False, False

        def smooth(data, window=5):                                     # 滑动平局滤波函数，用于平滑轨迹，减少关键点抖动带来的误差
            if len(data) < window: return np.array(data)
            return np.convolve(data, np.ones(window) / window, mode='same')

        if len(hip_ys) > 5:
            smoothed_hips = smooth(hip_ys)
            max_y_idx = np.argmax(smoothed_hips)
            max_y_val = smoothed_hips[max_y_idx]                        # 图像y轴向下，Y值最大代表重心最低

            min_y_before = np.min(smoothed_hips[:max_y_idx + 1]) if max_y_idx > 0 else smoothed_hips[0]            # 寻找下蹲前重心最高点
            min_y_after = np.min(smoothed_hips[max_y_idx:]) if max_y_idx < len(smoothed_hips) - 1 else max_y_val   # 寻找起跳后重心最高点

            drop_ratio = (max_y_val - min_y_before) / avg_h
            rise_ratio = (max_y_val - min_y_after) / avg_h

            min_hip_angle = np.min(hip_angles) if hip_angles else 180.0
            min_knee_angle = np.min(knee_angles) if knee_angles else 180.0

            # 判别1：是否下蹲（三选一条件：重心下降比例，明显的一上一下，角度）
            is_coordinate_dropped = drop_ratio > 0.04
            is_direction_changed = (drop_ratio > 0.02) and (rise_ratio > 0.02)
            is_angle_flexed = (min_hip_angle < 160.0) or (min_knee_angle < 152.0)
            if is_coordinate_dropped or is_direction_changed or is_angle_flexed:
                has_squat = True

            # 判别2：是否蹬伸
            post_min_knee = knee_angles[max_y_idx:] if knee_angles and max_y_idx < len(knee_angles) else []
            if post_min_knee and (np.max(post_min_knee) - np.min(knee_angles) > 15) and np.max(post_min_knee) > 155:
                has_extension = True
            elif rise_ratio > 0.05:
                has_extension = True
        else:
            if knee_angles and np.min(knee_angles) < 145: has_squat = True
            if knee_angles and np.max(knee_angles) - np.min(knee_angles) > 20: has_extension = True

        for m in frame_metrics:
            if m.get('angles') is not None and m.get('wrist_y') is not None and m.get('shoulder_y') is not None:
                if m['wrist_y'] < m['shoulder_y'] and m['angles'][1] > 140:
                    has_release = True
                    break

        stages_status, missing_details = [], []

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
        conclusion = "<font color='#A6E3A1'>🎉 恭喜！投篮核心技术环节完整，动作链衔接良好。</font>" if score == 100.0 else "<br>".join(missing_details)

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

    def compute_release_angle(self, frame_metrics):
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
        if true_avg_degree <= 10.0: return 100.0
        elif true_avg_degree >= 55.0: return 20.0
        else: return 100 - (true_avg_degree - 10) * (80 / 45)

    def process_video(self, video_path, save_visuals=False, out_dir=None):
        global det_model, pose_model
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened() or det_model is None or pose_model is None:
            return None, None, None, None, None, None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_fps = fps

        frame_metrics = []
        frame_idx = 0

        while True:
            success, frame = cap.read()
            if not success: break

            det_results = det_model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False, conf=0.45)
            pose_results = pose_model.predict(frame, verbose=False)

            current_data = {
                'idx': frame_idx, 'hip_y': None, 'angles': None, 'kpts': None,
                'cx1': 0, 'cy1': 0, 'player_box': None, 'ball_boxes': []
            }

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
                                if ball_cx < px1 - player_w * 1.5 or ball_cx > px2 + player_w * 1.5: continue

                            valid_balls.append((conf, (bx1, by1, bx2, by2)))

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

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            slow_fps = fps * 0.4
            out1_path = os.path.join(out_dir, "clip1_squat.mp4")
            out2_path = os.path.join(out_dir, "clip2_release.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out1 = cv2.VideoWriter(out1_path, fourcc, slow_fps, (480, 640))
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

    def run_analysis(self):
        # 1. 解析标准库 
        std_seqs1, std_seqs2, std_heights = [], [], []
        # 注意：此处确保 standard_videos 路径有效，或者自行加上跳过错误逻辑
        for i, path in enumerate(self.standard_videos):
            if not os.path.exists(path.strip()):
                continue # 忽略找不到的视频
            s1, s2, _, _, rel_h, _ = self.process_video(path.strip())
            if s1 is not None and len(s1) > 2: std_seqs1.append(s1)
            if s2 is not None and len(s2) > 2: std_seqs2.append(s2)
            if rel_h > 0: std_heights.append(rel_h)

        if not std_seqs1 or not std_seqs2:
            raise ValueError("标准视频库解析失败，无法提取两段动作特征（请确认文件是否存在）")

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

        # 2. 解析测试视频
        out_folder = os.path.join(CLIPS_FOLDER, str(uuid.uuid4()))
        os.makedirs(out_folder, exist_ok=True)
        test_s1, test_s2, out_v1, out_v2, test_rel_h, test_metrics = self.process_video(
            self.test_video_path, save_visuals=True, out_dir=out_folder)

        if test_s1 is None or test_s2 is None:
            raise ValueError("测试视频未检测到完整的下蹲和出手动作！")

        # 3. 算分
        d1, p1 = fastdtw(champ1, test_s1, dist=euclidean)
        deg1 = (d1 / len(p1)) / np.sqrt(4)
        score1 = self.compute_dtw_score(deg1)

        d2, p2 = fastdtw(champ2, test_s2, dist=euclidean)
        deg2 = (d2 / len(p2)) / np.sqrt(4)
        score2 = self.compute_dtw_score(deg2)

        coord_score, coord_report = self.compute_coordination(test_metrics)
        knee_score, knee_report = self.compute_knee_power(test_metrics, fps=self.current_fps)
        release_score, release_report = self.compute_release_angle(test_metrics)
        completeness_score, completeness_report = self.compute_completeness(test_metrics, fps=self.current_fps)

        # 4. 请求豆包 API
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
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                resp_json = response.json()
                ai_report = resp_json['choices'][0]['message']['content']
        except Exception as e:
            ai_report = f"AI 请求失败: {str(e)}"

        # 5. 组织前端返回数据
        v1_url = f"/clips/{os.path.basename(out_folder)}/clip1_squat.mp4" if out_v1 else ""
        v2_url = f"/clips/{os.path.basename(out_folder)}/clip2_release.mp4" if out_v2 else ""
        frames = []
        if os.path.exists(out_folder):
            frames = [f"/clips/{os.path.basename(out_folder)}/{f}" for f in sorted(os.listdir(out_folder)) if f.endswith('.jpg')]

        return {
            "score1": score1, 
            "score2": score2, 
            "total_score": (score1 + score2) / 2,
            "v1_url": v1_url, 
            "v2_url": v2_url, 
            "frames": frames,
            "completeness": {"score": completeness_score, "report": completeness_report},
            "coordination": {"score": coord_score, "report": coord_report},
            "knee": {"score": knee_score, "report": knee_report},
            "release": {"score": release_score, "report": release_report},
            "ai_report": ai_report
        }

# ---------------- Flask 路由设定 ----------------

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "未上传视频文件"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
    file.save(filepath)

    try:
        analyzer = BasketballAnalyzer(filepath)
        results = analyzer.run_analysis()
        return jsonify(results)
    except Exception as e:
        import traceback
        traceback.print_exc() # 在控制台打印详细错误方便排查
        return jsonify({"error": str(e)}), 500

@app.route('/clips/<path:filename>')
def serve_clips(filename):
    return send_from_directory(CLIPS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)