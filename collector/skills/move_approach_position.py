"""
move_approach_position — 물체 위 접근 이동 스킬

물체의 [x, y] 좌표 위, approach_height 높이로 EE를 이동하는 스킬.
move_to_position과 달리 물체 위치 기반으로 접근 높이를 자동 적용.

차이점:
    move_to_position: position=[x, y, z] 그대로 이동
    move_approach_position: position=[obj_x, obj_y, obj_z]에서 z를 approach_height로 대체

용도:
    물체 상호작용 전 안전한 접근 위치로 이동.
    LLM 생성 코드에서 pick/place/press/insert 등 전에 호출.

Usage:
    from skills.move_approach_position import move_approach_position

    # 물체 위 10cm 접근
    move_approach_position(
        skills,
        object_position=[0.20, 0.05, 0.03],
        approach_height=0.10,
        object_name="red block",
    )
"""

from typing import List, Optional, Union

import numpy as np


def move_approach_position(
    skills,
    object_position: Union[List[float], np.ndarray],
    approach_height: float = 0.10,
    duration: Optional[float] = None,
    object_name: Optional[str] = None,
    skill_description: Optional[str] = None,
) -> bool:
    """
    물체 위 접근 높이로 이동.

    물체의 xy 좌표 위, approach_height 높이로 EE를 이동.
    물체의 z(높이)는 무시하고 approach_height를 사용.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        object_position: 물체 위치 [x, y, z] (단위: meters)
                         - x, y: 물체의 수평 좌표 (이동 목표)
                         - z: 물체 높이 (접근 이동에서는 무시됨)

        approach_height: 접근 높이 (단위: meters). default=0.10 (10cm)
                         - 물체 위에서 유지할 안전 높이
                         - EE가 이동할 실제 z 좌표
                         - 0.10: 일반 접근 (pick/place)
                         - 0.15~0.20: 큰 물체가 있을 때

        duration: 이동 시간 (단위: 초). default=None (기본값 사용)

        object_name: 접근 대상 물체 이름 (선택). default=None
                     - 레코딩 시 subgoal 라벨에 사용

        skill_description: 스킬 동작 설명 (선택). default=None
                           - None이면 자동 생성: "approach above {object_name}"

    Returns:
        bool: True면 접근 성공
    """
    pos = np.array(object_position, dtype=float)

    desc = skill_description or f"approach above {object_name or 'target'}"

    skills._log(f"\n[move_approach_position] {desc}")
    skills._log(f"  Object: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    skills._log(f"  Approach: [{pos[0]:.3f}, {pos[1]:.3f}, {approach_height:.3f}]")

    return skills.move_to_position(
        position=[pos[0], pos[1], approach_height],
        duration=duration,
        target_name=object_name,
        skill_description=desc,
    )
