import sys
import os
import cv2
import platform
import subprocess
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QListWidget,
                             QMessageBox, QListWidgetItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from ultralytics import YOLO

class VideoProcessorThread(QThread):
    # 定义信号，用于子线程与主线程的 UI 交互
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            self.progress_signal.emit("正在加载 YOLO 模型...")

            # 1. 加载目标检测和姿态估计模型
            det_model = YOLO("D:/develop/lanqiu/ultralytics-main/runs/detect/train/weights/best.pt")
            pose_model = YOLO("./yolov8n-pose.pt")

            # 2. 视频路径设置
            input_path = self.video_path
            output_path = "./main_pose_result.mp4"

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.error_signal.emit("无法打开视频，请检查视频文件是否损坏。")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 自动切分视频的初始化设置
            output_dir = "shot_clips"
            os.makedirs(output_dir, exist_ok=True)

            # 清理目录中旧的切片文件
            for f in os.listdir(output_dir):
                if f.endswith(".mp4"):
                    os.remove(os.path.join(output_dir, f))

            is_shooting = False
            shot_writer = None
            shot_count = 0
            buffer_frames = 0
            MAX_BUFFER = 45  # 缓冲期：球离手后继续录制 45 帧（约 1.5 秒）

            # 3. 定义人体骨架和连接线
            skeleton_connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12),
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]

            self.progress_signal.emit("开始处理视频，这可能需要一段时间...")

            frame_idx = 0
            clip_paths = []

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_idx += 1
                if frame_idx % 15 == 0:  # 每处理15帧更新一次 UI 提示
                    self.progress_signal.emit(f"正在解析进度: {frame_idx} / {total_frames} 帧")

                det_results = det_model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)

                # 临时存放当前帧中所有的球员框和篮球框，用于后续相交判断
                current_player_boxes = []
                current_ball_boxes = []

                if det_results[0].boxes is not None and len(det_results[0].boxes) > 0:
                    boxes = det_results[0].boxes.xyxy.cpu().numpy()
                    clses = det_results[0].boxes.cls.cpu().numpy()
                    scores = det_results[0].boxes.conf.cpu().numpy()

                    for box, cls, score in zip(boxes, clses, scores):
                        x1, y1, x2, y2 = map(int, box)
                        cls_id = int(cls)

                        if cls_id == 0:
                            current_player_boxes.append((x1, y1, x2, y2))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 110, 0), 2)
                            display_text = f"player {score:.2f}"
                            cv2.putText(frame, display_text, (x1, max(10, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 110, 0), 2)

                            pad_w, pad_h = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
                            cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                            cx2, cy2 = min(width, x2 + pad_w), min(height, y2 + pad_h)

                            player_crop = frame[cy1:cy2, cx1:cx2]
                            if player_crop.size == 0: continue

                            pose_results = pose_model.predict(player_crop, verbose=False)

                            if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
                                kpts = pose_results[0].keypoints.xy[0].cpu().numpy()

                                for p1_idx, p2_idx in skeleton_connections:
                                    pt1 = kpts[p1_idx]
                                    pt2 = kpts[p2_idx]
                                    if pt1[0] == 0 or pt2[0] == 0: continue
                                    pt1_mapped = (int(pt1[0] + cx1), int(pt1[1] + cy1))
                                    pt2_mapped = (int(pt2[0] + cx1), int(pt2[1] + cy1))
                                    cv2.line(frame, pt1_mapped, pt2_mapped, (255, 144, 30), 2)

                                for i, pt in enumerate(kpts):
                                    if pt[0] == 0: continue
                                    pt_mapped = (int(pt[0] + cx1), int(pt[1] + cy1))
                                    if i <= 4:
                                        cv2.circle(frame, pt_mapped, 2, (0, 255, 255), -1)
                                    else:
                                        cv2.circle(frame, pt_mapped, 4, (0, 255, 0), -1)

                        else:
                            current_ball_boxes.append((x1, y1, x2, y2))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                            display_text = f"basketball {score:.2f}"
                            cv2.putText(frame, display_text, (x1, max(10, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # 持球相交判断逻辑
                player_has_ball = False
                for p_box in current_player_boxes:
                    px1, py1, px2, py2 = p_box
                    for b_box in current_ball_boxes:
                        bx1, by1, bx2, by2 = b_box
                        if not (bx2 < px1 or bx1 > px2 or by2 < py1 or by1 > py2):
                            player_has_ball = True
                            break
                    if player_has_ball:
                        break

                is_trigger_active = player_has_ball

                if not is_shooting and is_trigger_active:
                    is_shooting = True
                    shot_count += 1
                    buffer_frames = 0
                    clip_name = os.path.join(output_dir, f"shot_{shot_count}.mp4")
                    shot_writer = cv2.VideoWriter(clip_name, fourcc, fps, (width, height))
                    clip_paths.append(os.path.abspath(clip_name))
                    self.progress_signal.emit(f"检测到持球，正在录制片段 {shot_count}...")

                if is_shooting:
                    shot_writer.write(frame)
                    if not is_trigger_active:
                        buffer_frames += 1
                        if buffer_frames >= MAX_BUFFER:
                            is_shooting = False
                            shot_writer.release()
                            shot_writer = None
                    else:
                        buffer_frames = 0

                out.write(frame)

            cap.release()
            out.release()
            if shot_writer is not None:
                shot_writer.release()

            self.progress_signal.emit(f"处理完成！主视频已保存为：{output_path}")
            self.finished_signal.emit(clip_paths)

        except Exception as e:
            self.error_signal.emit(f"发生错误: {str(e)}")


class BasketballPoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("篮球视频分析系统")
        self.resize(800, 600)

        main_layout = QVBoxLayout()

        # 1. 大标题 (正上方中心) - PyQt6 中对齐方式需写完整：Qt.AlignmentFlag.AlignCenter
        self.title_label = QLabel("篮球姿态识别解析", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: #003366; background-color: transparent;")

        # 字体样式
        font = QFont("Microsoft YaHei", 24)
        font.setBold(True)
        self.setStyleSheet("""
                    QWidget {
                        background-color: #F3F3F6;
                        color: black;
                    }
                    QMessageBox {
                        background-color: white; /* 让弹窗背景变成纯白 */
                    }
                    QMessageBox QLabel {
                        color: black; /* 确保弹窗里的提示文字是黑色的 */
                    }
                """)
        self.title_label.setFont(font)
        main_layout.addWidget(self.title_label)

        main_layout.addSpacing(20)

        # 2. 按钮区域 (左边选择，右边解析)
        btn_layout = QHBoxLayout()

        self.btn_select = QPushButton("选择视频", self)
        self.btn_select.setMinimumHeight(40)
        self.btn_select.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50; /* 常态背景色：绿色 */
                        color: white;              /* 字体颜色：白色 */
                        border-radius: 5px;        /* 圆角 */
                        font-weight: bold;         /* 字体加粗 */
                    }
                    QPushButton:hover {
                        background-color: #45a049; /* 鼠标悬停变深绿 */
                    }
                    QPushButton:disabled {
                        background-color: #cccccc; /* 禁用状态变为灰色 */
                        color: #666666;
                    }
                """)
        self.btn_select.clicked.connect(self.select_video)

        self.lbl_file_path = QLabel("尚未选择视频文件")
        self.lbl_file_path.setStyleSheet("color: black;")

        self.btn_analyze = QPushButton("开始解析", self)
        self.btn_analyze.setMinimumHeight(40)
        self.btn_analyze.setStyleSheet("""
                            QPushButton {
                                background-color: #4CAF50; /* 常态背景色：绿色 */
                                color: white;              /* 字体颜色：白色 */
                                border-radius: 5px;        /* 圆角 */
                                font-weight: bold;         /* 字体加粗 */
                            }
                            QPushButton:hover {
                                background-color: #45a049; /* 鼠标悬停变深绿 */
                            }
                            QPushButton:disabled {
                                background-color: #cccccc; /* 禁用状态变为灰色 */
                                color: #666666;
                            }
                        """)
        self.btn_analyze.setEnabled(False)  # 未选择文件前不可点击
        self.btn_analyze.clicked.connect(self.start_analysis)

        btn_layout.addWidget(self.btn_select, 1)
        btn_layout.addWidget(self.lbl_file_path, 3)
        btn_layout.addWidget(self.btn_analyze, 1)
        main_layout.addLayout(btn_layout)

        main_layout.addSpacing(20)

        # 3. 状态提示
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color: black;")
        main_layout.addWidget(self.lbl_status)

        # 4. 输出切分小视频的列表区
        self.lbl_list_title = QLabel("切分动作片段列表 (双击列表项即可播放):")
        self.lbl_list_title.setStyleSheet("color: black;")
        main_layout.addWidget(self.lbl_list_title)

        self.clip_list = QListWidget(self)
        self.clip_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                color: black;
                border: 1px solid #cccccc;
            }
        """)
        self.clip_list.itemDoubleClicked.connect(self.play_video)
        main_layout.addWidget(self.clip_list)

        self.setLayout(main_layout)

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择篮球视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_path = file_name
            self.lbl_file_path.setText(self.video_path)
            self.btn_analyze.setEnabled(True)
            self.clip_list.clear()
            self.lbl_status.setText("视频已选择，准备解析。")

    def start_analysis(self):
        if not self.video_path:
            return

        self.btn_select.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.clip_list.clear()

        # 启动处理线程
        self.thread = VideoProcessorThread(self.video_path)
        self.thread.progress_signal.connect(self.update_status)
        self.thread.error_signal.connect(self.show_error)
        self.thread.finished_signal.connect(self.analysis_finished)
        self.thread.start()

    def update_status(self, msg):
        self.lbl_status.setText(msg)

    def show_error(self, err_msg):
        self.lbl_status.setText("处理失败！")
        QMessageBox.critical(self, "错误", err_msg)
        self.btn_select.setEnabled(True)
        self.btn_analyze.setEnabled(True)

    def analysis_finished(self, clip_paths):
        self.btn_select.setEnabled(True)
        self.btn_analyze.setEnabled(True)

        if not clip_paths:
            self.clip_list.addItem("未检测到有效投篮/持球动作片段。")
        else:
            for path in clip_paths:
                item = QListWidgetItem(f"{os.path.basename(path)}")
                # PyQt6 中 DataRole 需要写完整：Qt.ItemDataRole.UserRole
                item.setData(Qt.ItemDataRole.UserRole, path)
                self.clip_list.addItem(item)

        QMessageBox.information(self, "完成", "视频解析与切分已完成！")

    def play_video(self, item):
        video_path = item.data(Qt.ItemDataRole.UserRole)
        if video_path and os.path.exists(video_path):
            # 根据不同的操作系统调用默认播放器打开视频
            if platform.system() == "Windows":
                os.startfile(video_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(["open", video_path])
            else:  # Linux
                subprocess.call(["xdg-open", video_path])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BasketballPoseApp()
    ex.show()
    sys.exit(app.exec())