#!/bin/bash
# Check Cameras - recording_config.yaml에 정의된 카메라들의
# 라이브 스트리밍 및 FPS를 실시간으로 확인합니다.
# Press 'q' to quit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Camera Check (recording_config.yaml)"
echo "========================================"
echo "Config: pipeline_config/recording_config.yaml"
echo "Press 'q' on any camera window to quit."
echo "========================================"
echo ""

python3 check_cameras.py
