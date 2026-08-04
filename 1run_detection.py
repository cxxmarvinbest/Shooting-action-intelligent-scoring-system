import cv2
from ultralytics import YOLO

# 1. 导入已经训练好的模型
model = YOLO("runs/detect/train/weights/best.pt")

# 2. 执行预测任务
results = model.predict(
    source="./1right_side_basketball.mp4",      # 源视频路径
    show=True,                                  # 在屏幕上实时弹出窗口显示检测过程
    save=True,                                  # 核心参数：将带有检测框的视频保存下来
    project="runs/detect",                      # 指定保存的主目录
    name="right_basketball_detection_result",   # 指定保存的子文件夹名称
    conf=0.5                                    # 置信度阈值，过滤掉低于50%的预测框，让结果更干净
)

print("视频检测完成！请前往 runs/detect/basketball_result 文件夹查看结果视频。")