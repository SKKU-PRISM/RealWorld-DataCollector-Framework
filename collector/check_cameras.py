"""
Camera Check - recording_config.yaml에 정의된 모든 카메라의
라이브 스트리밍 및 실시간 FPS를 시각화합니다.
Press 'q' to quit.
"""

import sys
import time
import collections
from pathlib import Path

import cv2
import numpy as np

# project root를 path에 추가
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from record_dataset.config import load_cameras_from_yaml

# ============================================================
# Constants
# ============================================================
FPS_WINDOW = 60       # rolling window for FPS calculation
HISTORY_LEN = 300     # number of FPS samples to plot
GRAPH_W, GRAPH_H = 500, 280
MAX_FPS_DISPLAY = 40

# 카메라별 색상 팔레트
COLORS = [
    (0, 255, 0),     # green
    (0, 128, 255),   # orange
    (255, 100, 100), # blue-ish
    (255, 255, 0),   # cyan
    (180, 0, 255),   # magenta
    (0, 255, 255),   # yellow
]


def open_opencv(device_path, width, height, fps, fourcc="MJPG"):
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def open_realsense(serial, width=640, height=480, fps=30):
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


def draw_graph(cam_data_list):
    graph = np.zeros((GRAPH_H, GRAPH_W, 3), dtype=np.uint8)

    # Grid lines
    for y_val in [10, 20, 30]:
        y_px = GRAPH_H - int(y_val * GRAPH_H / MAX_FPS_DISPLAY)
        cv2.line(graph, (0, y_px), (GRAPH_W, y_px), (40, 40, 40), 1)
        cv2.putText(graph, f"{y_val}", (5, y_px - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    # 30fps target line
    y30 = GRAPH_H - int(30 * GRAPH_H / MAX_FPS_DISPLAY)
    cv2.line(graph, (0, y30), (GRAPH_W, y30), (0, 100, 100), 1)
    cv2.putText(graph, "30fps target", (GRAPH_W - 120, y30 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 100), 1)

    # Plot each camera
    for i, cam in enumerate(cam_data_list):
        pts = list(cam["fps_history"])
        if len(pts) < 2:
            continue
        step = GRAPH_W / HISTORY_LEN
        for j in range(1, len(pts)):
            x1 = int((j - 1) * step)
            x2 = int(j * step)
            y1 = GRAPH_H - int(min(pts[j - 1], MAX_FPS_DISPLAY) * GRAPH_H / MAX_FPS_DISPLAY)
            y2 = GRAPH_H - int(min(pts[j], MAX_FPS_DISPLAY) * GRAPH_H / MAX_FPS_DISPLAY)
            cv2.line(graph, (x1, y1), (x2, y2), cam["color"], 2)

        cv2.putText(
            graph,
            f'{cam["display_name"]}: {cam["current_fps"]:.1f} fps',
            (10, 20 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, cam["color"], 1,
        )

    return graph


def main():
    yaml_path = ROOT / "pipeline_config" / "recording_config.yaml"
    camera_configs = load_cameras_from_yaml(str(yaml_path))

    cam_data_list = []

    for idx, cam_cfg in enumerate(camera_configs):
        if not cam_cfg.enabled:
            print(f"[SKIP] {cam_cfg.feature_name} (disabled)")
            continue

        color = COLORS[idx % len(COLORS)]

        if cam_cfg.type == "realsense":
            serial = cam_cfg.serial_number or ""
            display_name = f"{cam_cfg.feature_name} (realsense serial={serial})"
            try:
                pipeline = open_realsense(
                    serial, cam_cfg.width, cam_cfg.height, cam_cfg.fps)
                cam_data_list.append({
                    "display_name": display_name,
                    "type": "realsense",
                    "color": color,
                    "pipeline": pipeline,
                    "timestamps": collections.deque(maxlen=FPS_WINDOW),
                    "fps_history": collections.deque(maxlen=HISTORY_LEN),
                    "current_fps": 0.0,
                })
                print(f"[OK] {display_name}")
            except Exception as e:
                print(f"[FAIL] {display_name}: {e}")

        elif cam_cfg.type == "opencv":
            device = cam_cfg.get_device_path() or "/dev/video0"
            display_name = f"{cam_cfg.feature_name} (opencv {device})"
            cap = open_opencv(
                device, cam_cfg.width, cam_cfg.height, cam_cfg.fps, cam_cfg.fourcc)
            if not cap.isOpened():
                print(f"[FAIL] {display_name}: cannot open")
                continue
            cam_data_list.append({
                "display_name": display_name,
                "type": "opencv",
                "color": color,
                "cap": cap,
                "timestamps": collections.deque(maxlen=FPS_WINDOW),
                "fps_history": collections.deque(maxlen=HISTORY_LEN),
                "current_fps": 0.0,
            })
            print(f"[OK] {display_name}")

    if not cam_data_list:
        print("\nNo cameras opened. Check recording_config.yaml and device connections.")
        return

    print(f"\nOpened {len(cam_data_list)} camera(s). Press q to quit.\n")
    frame_count = 0

    try:
        while True:
            for cam in cam_data_list:
                frame = None

                if cam["type"] == "opencv":
                    ret, frame = cam["cap"].read()
                    if not ret:
                        continue
                elif cam["type"] == "realsense":
                    try:
                        frames = cam["pipeline"].wait_for_frames(timeout_ms=1000)
                    except RuntimeError:
                        continue
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    frame = np.asanyarray(color_frame.get_data())

                cam["timestamps"].append(time.time())
                ts = cam["timestamps"]
                cam["current_fps"] = (
                    len(ts) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
                )
                cam["fps_history"].append(cam["current_fps"])

                cv2.putText(frame, cam["display_name"], (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cam["color"], 2)
                cv2.putText(frame, f'FPS: {cam["current_fps"]:.1f}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, cam["color"], 2)
                cv2.imshow(cam["display_name"], frame)

            graph = draw_graph(cam_data_list)
            cv2.imshow("FPS Monitor", graph)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_count += 1

    finally:
        for cam in cam_data_list:
            if cam["type"] == "opencv":
                cam["cap"].release()
            elif cam["type"] == "realsense":
                cam["pipeline"].stop()
        cv2.destroyAllWindows()

    print(f"\nStopped after {frame_count} frames.")
    for cam in cam_data_list:
        print(f'  {cam["display_name"]}: {cam["current_fps"]:.1f} fps')


if __name__ == "__main__":
    main()
