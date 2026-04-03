"""
Camera FPS Monitor - Live streaming with real-time FPS visualization.
Supports OpenCV (Innomaker) and RealSense cameras.
Press 'q' to quit.
"""

import cv2
import time
import collections
import numpy as np

# ============================================================
# Camera Configuration
# ============================================================
OPENCV_CAMERAS = [
    {"name": "Innomaker #1", "dev": "/dev/video6", "color": (0, 255, 0)},
    {"name": "Innomaker #2", "dev": "/dev/video8", "color": (0, 128, 255)},
]

REALSENSE_CAMERAS = [
    {"name": "RealSense D435", "serial": "335622072328", "color": (255, 100, 100)},
]

FPS_WINDOW = 60       # rolling window for FPS calculation
HISTORY_LEN = 300     # number of FPS samples to plot
GRAPH_W, GRAPH_H = 500, 280
MAX_FPS_DISPLAY = 40


def open_mjpg(dev):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def open_realsense(serial, width=640, height=480, fps=30):
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


def draw_graph(cam_data):
    graph = np.zeros((GRAPH_H, GRAPH_W, 3), dtype=np.uint8)

    # Grid lines
    for y_val in [10, 20, 30]:
        y_px = GRAPH_H - int(y_val * GRAPH_H / MAX_FPS_DISPLAY)
        cv2.line(graph, (0, y_px), (GRAPH_W, y_px), (40, 40, 40), 1)
        cv2.putText(graph, f"{y_val}", (5, y_px - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    # 30fps target line
    y30 = GRAPH_H - int(30 * GRAPH_H / MAX_FPS_DISPLAY)
    cv2.line(graph, (0, y30), (GRAPH_W, y30), (0, 100, 100), 1)
    cv2.putText(graph, "30fps target", (GRAPH_W - 120, y30 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 100), 1)

    # Plot each camera's FPS history
    for i, cam in enumerate(cam_data):
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

        # Legend
        cv2.putText(
            graph,
            f'{cam["name"]}: {cam["current_fps"]:.1f} fps',
            (10, 20 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            cam["color"],
            1,
        )

    return graph


def main():
    cam_data = []

    # Open OpenCV cameras (Innomaker)
    for cam in OPENCV_CAMERAS:
        cap = open_mjpg(cam["dev"])
        if not cap.isOpened():
            print(f'[WARN] Cannot open {cam["dev"]}')
            continue
        cam_data.append({
            "name": cam["name"],
            "type": "opencv",
            "color": cam["color"],
            "dev": cam["dev"],
            "cap": cap,
            "timestamps": collections.deque(maxlen=FPS_WINDOW),
            "fps_history": collections.deque(maxlen=HISTORY_LEN),
            "current_fps": 0.0,
        })
        print(f'[OK] {cam["name"]} ({cam["dev"]})')

    # Open RealSense cameras
    for cam in REALSENSE_CAMERAS:
        try:
            pipeline = open_realsense(cam["serial"])
            cam_data.append({
                "name": cam["name"],
                "type": "realsense",
                "color": cam["color"],
                "serial": cam["serial"],
                "pipeline": pipeline,
                "timestamps": collections.deque(maxlen=FPS_WINDOW),
                "fps_history": collections.deque(maxlen=HISTORY_LEN),
                "current_fps": 0.0,
            })
            print(f'[OK] {cam["name"]} (serial={cam["serial"]})')
        except Exception as e:
            print(f'[WARN] Cannot open RealSense {cam["serial"]}: {e}')

    if not cam_data:
        print("No cameras opened. Exiting.")
        return

    print(f"\nOpened {len(cam_data)} cameras. Press q to quit.")
    frame_count = 0

    try:
        while True:
            for cam in cam_data:
                frame = None

                if cam["type"] == "opencv":
                    ret, frame = cam["cap"].read()
                    if not ret:
                        continue
                elif cam["type"] == "realsense":
                    frames = cam["pipeline"].wait_for_frames(timeout_ms=100)
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    frame = np.asanyarray(color_frame.get_data())

                cam["timestamps"].append(time.time())
                ts = cam["timestamps"]
                cam["current_fps"] = len(ts) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
                cam["fps_history"].append(cam["current_fps"])

                label = cam["name"]
                if cam["type"] == "opencv":
                    label += f' ({cam["dev"]})'
                elif cam["type"] == "realsense":
                    label += f' (serial={cam["serial"]})'
                cv2.putText(frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cam["color"], 2)
                cv2.putText(frame, f'FPS: {cam["current_fps"]:.1f}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, cam["color"], 2)
                cv2.imshow(label, frame)

            graph = draw_graph(cam_data)
            cv2.imshow("FPS Monitor", graph)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_count += 1

    finally:
        for cam in cam_data:
            if cam["type"] == "opencv":
                cam["cap"].release()
            elif cam["type"] == "realsense":
                cam["pipeline"].stop()
        cv2.destroyAllWindows()

    print(f"\nStopped after {frame_count} frames.")
    for cam in cam_data:
        print(f'  {cam["name"]}: {cam["current_fps"]:.1f} fps')


if __name__ == "__main__":
    main()
