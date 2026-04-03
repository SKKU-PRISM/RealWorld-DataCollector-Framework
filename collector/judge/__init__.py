"""
Judge Module for LeRobot Task Evaluation

VLM을 활용하여 로봇 태스크 완료 여부를 자동으로 판단합니다.

폴더 구조:
    judge/
    ├── __init__.py (이 파일)
    ├── vlm.py (공통 - VLM API 호출)
    ├── image_capture.py (공통 - 이미지 캡처)
    ├── visualize.py (공통 - 결과 시각화)
    ├── forward_execution/
    │   ├── judge.py (Forward Task Judge)
    │   └── prompt.py (Forward Judge Prompt)
    └── reset_execution/
        ├── judge.py (Reset Task Judge)
        └── prompt.py (Reset Judge Prompt)

Usage:
    # Forward Task Judge
    from judge import TaskJudge, capture_frame

    judge = TaskJudge()
    result = judge.judge(
        instruction="pick up the red cup",
        initial_image=initial_img,
        final_image=final_img,
        object_positions={"red cup": [0.15, 0.05, 0.02]},
        executed_code=code_string,
    )
    print(result['prediction'])  # TRUE / FALSE / UNCERTAIN
    print(result['reasoning'])   # 판단 근거

    # Reset Task Judge
    from judge import ResetJudge

    reset_judge = ResetJudge()
    result = reset_judge.judge(
        reset_mode="original",  # or "random"
        current_positions=current_pos,
        target_positions=target_pos,
        initial_image=before_reset_img,
        final_image=after_reset_img,
        executed_code=reset_code,
        original_instruction="pick up the red cup",
    )
"""

# Forward Execution Judge
from .forward_execution.judge import TaskJudge
from .forward_execution.prompt import build_judge_prompt

# Reset Execution Judge
from .reset_execution.judge import ResetJudge
from .reset_execution.prompt import build_reset_judge_prompt

# Common utilities
from .image_capture import capture_frame, ImageCapture
from .visualize import (
    show_judge_result,
    save_judge_log,
    create_result_image,
    show_reset_judge_result,
    create_reset_result_image,
)

__all__ = [
    # Forward Judge
    "TaskJudge",
    "build_judge_prompt",
    # Reset Judge
    "ResetJudge",
    "build_reset_judge_prompt",
    # Common
    "capture_frame",
    "ImageCapture",
    "show_judge_result",
    "save_judge_log",
    "create_result_image",
    "show_reset_judge_result",
    "create_reset_result_image",
]
