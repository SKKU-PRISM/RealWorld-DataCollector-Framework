"""
Reset Workspace - Reset 태스크용 추가 제약

BaseWorkspace를 상속하고 더 좁은 범위로 제한.

주요 기능:
1. Grippable 객체 분류 (그리퍼로 잡을 수 있는지 판단)
2. 랜덤 타겟 위치 생성 (충돌 회피, 초기 위치와 다른 위치)
3. 워크스페이스 자동 경계 계산 (IK 그리드 샘플링)
4. 이미지 위 워크스페이스 시각화
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot_cap.workspace import BaseWorkspace

if TYPE_CHECKING:
    from lerobot_cap.kinematics.engine import KinematicsEngine


# SO-101 그리퍼 사양 (픽셀 기준)
GRIPPER_MAX_OPEN_PX = 80  # 그리퍼 최대 열림 폭 (pixels, ~10cm 상당)


# ============================================================
# Reset Quadrant Definitions (오버헤드 카메라 pixel 기준, 640x480)
# ============================================================
QUADRANT_DEFINITIONS = {
    "all": None,  # 제약 없음
    "top-left":     {"u_range": (0, 320), "v_range": (0, 240)},
    "top-right":    {"u_range": (320, 640), "v_range": (0, 240)},
    "bottom-left":  {"u_range": (0, 320), "v_range": (240, 480)},
    "bottom-right": {"u_range": (320, 640), "v_range": (240, 480)},
}

VALID_RESETSPACE_TYPES = list(QUADRANT_DEFINITIONS.keys())


def is_in_quadrant(u: int, v: int, quadrant: str) -> bool:
    """pixel 좌표 (u, v)가 지정된 quadrant 안에 있는지 체크."""
    if quadrant == "all" or quadrant is None:
        return True
    bounds = QUADRANT_DEFINITIONS.get(quadrant)
    if bounds is None:
        return True
    u_min, u_max = bounds["u_range"]
    v_min, v_max = bounds["v_range"]
    return u_min <= u < u_max and v_min <= v < v_max


class ResetWorkspace(BaseWorkspace):
    """
    Reset 태스크 전용 Workspace.

    BaseWorkspace를 상속하고 추가 제약:
    - Reach 기반 도달 가능 범위 (base_link frame)
    - IK 검증 (approach + pick 높이)
    - Z는 테이블 표면에 고정
    - 객체 간 최소 거리
    - 초기 위치에서 최소 이동 거리
    """

    def __init__(
        self,
        kinematics_engine: Optional["KinematicsEngine"] = None,
        z_fixed: float = 0.01,
    ):
        """
        Initialize ResetWorkspace.

        Args:
            kinematics_engine: KinematicsEngine 인스턴스
            z_fixed: 고정 Z 높이 (테이블 표면, meters)
        """
        super().__init__(kinematics_engine)

        self.z_fixed = z_fixed

    def is_valid(self, position: np.ndarray) -> bool:
        """
        Reset용 유효성 검사 (reach 기반).

        Args:
            position: [x, y, z] 위치 (base_link frame)

        Returns:
            True if valid for reset
        """
        return self.is_reachable(position)

    def _check_ik_feasible(self, position: np.ndarray) -> bool:
        """
        해당 위치에서 approach(z=0.20) + pick(z=0.025) IK 기본 검증.
        상세 검증은 dry_run_code()에서 실제 생성 코드로 수행.
        """
        if self._kinematics is None:
            return True

        # approach 높이
        pos_approach = np.array([position[0], position[1], 0.20])
        _, success = self._kinematics.inverse_kinematics_position_only(pos_approach, max_iterations=50)
        if not success:
            return False

        # pick 높이
        pos_pick = np.array([position[0], position[1], 0.025])
        _, success = self._kinematics.inverse_kinematics_position_only(pos_pick, max_iterations=50)
        return success

    @staticmethod
    def _compute_iou(cx1, cy1, w1, h1, cx2, cy2, w2, h2) -> float:
        """두 bbox의 IoU 계산 (center + size 형식)."""
        # AABB 좌표 변환
        ax1, ay1 = cx1 - w1 / 2, cy1 - h1 / 2
        ax2, ay2 = cx1 + w1 / 2, cy1 + h1 / 2
        bx1, by1 = cx2 - w2 / 2, cy2 - h2 / 2
        bx2, by2 = cx2 + w2 / 2, cy2 + h2 / 2

        # 교집합
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        if inter == 0:
            return 0.0

        # 합집합
        area_a = w1 * h1
        area_b = w2 * h2
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def generate_random_position(
        self,
        obstacles: List[dict],
        obj_bbox_px: Optional[Tuple[int, int]] = None,
        pix2robot=None,
        max_attempts: int = 500,
        max_iou: float = 0.3,
        exclusion_zones: Optional[List[dict]] = None,
        resetspace: Optional[str] = None,
    ) -> Optional[List[float]]:
        """
        단일 객체용 랜덤 위치 생성 (reach + FOV + IoU 기반 충돌 검증).

        Args:
            obstacles: 피해야 할 장애물들
                [{"center_px": (u,v), "bbox_w": int, "bbox_h": int, "allow_overlap": bool}, ...]
                allow_overlap=True: IoU ≤ max_iou 허용 (grippable)
                allow_overlap=False: 겹침 불허 (non-grippable, margin 포함)
            obj_bbox_px: 이 객체의 bbox 픽셀 크기 (w_px, h_px). None이면 (30, 30) 사용.
            pix2robot: Pix2RobotCalibrator 인스턴스 (robot↔pixel 변환)
            max_attempts: 최대 시도 횟수
            max_iou: grippable 장애물과 허용 최대 IoU (default: 0.5)
            exclusion_zones: 제외 영역 리스트 (robot base_link frame).
                       [{"center": [x, y], "radius": float}, ...]
                       예: free state EE 주변 8cm 제외.
            resetspace: reset quadrant 제약 ("all", "top-left", "top-right",
                       "bottom-left", "bottom-right"). None이면 제약 없음.

        Returns:
            [x, y, z] 또는 None (실패 시)
        """
        margin = 0.01
        r_min = self.min_reach + margin
        r_max = self.max_reach - margin

        if obj_bbox_px is None:
            obj_bbox_px = (30, 30)
        obj_w, obj_h = obj_bbox_px

        for _ in range(max_attempts):
            # 전방 도넛 영역 내에서 직접 샘플링
            angle = np.random.uniform(-np.pi / 2, np.pi / 2)
            r = np.sqrt(np.random.uniform(r_min**2, r_max**2))
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = self.z_fixed

            candidate = [x, y, z]

            # 조건 0: exclusion_zones 제한 (free state EE 주변 등)
            if exclusion_zones:
                in_exclusion = False
                for zone in exclusion_zones:
                    zx, zy = zone["center"]
                    zr = zone["radius"]
                    dist = np.sqrt((x - zx) ** 2 + (y - zy) ** 2)
                    if dist < zr:
                        in_exclusion = True
                        break
                if in_exclusion:
                    continue

            # 조건 1: pixel 변환 + quadrant 체크 + FOV 체크 (가벼운 연산 먼저)
            if pix2robot is not None:
                try:
                    cu, cv = pix2robot.robot_to_pixel(x, y)
                except Exception:
                    continue

                # Quadrant 체크 (resetspace 제약)
                if not is_in_quadrant(cu, cv, resetspace):
                    continue

                # FOV + 가장자리 마진
                img_w, img_h = 640, 480
                edge_margin = 30
                hw, hh = obj_w // 2, obj_h // 2
                if (cu - hw < edge_margin or cu + hw >= img_w - edge_margin or
                    cv - hh < edge_margin or cv + hh >= img_h - edge_margin):
                    continue
            elif resetspace is not None and resetspace != "all":
                # pix2robot 없으면 quadrant 체크 불가 → 스킵
                continue

            # 조건 2: 기본 IK 검증 (무거운 연산)
            if not self._check_ik_feasible(np.array(candidate)):
                continue

            # 조건 3: 충돌 검사 (pix2robot 필요)
            if pix2robot is not None:
                # 충돌 검사: IoU 기반
                collision = False
                for occ in obstacles:
                    ou, ov = occ["center_px"]
                    ow, oh = occ["bbox_w"], occ["bbox_h"]

                    if occ.get("allow_overlap", False):
                        # grippable: IoU ≤ max_iou 허용
                        iou = self._compute_iou(cu, cv, obj_w, obj_h, ou, ov, ow, oh)
                        if iou > max_iou:
                            collision = True
                            break
                    else:
                        # non-grippable: 겹침 불허 (AABB, margin 포함)
                        if (abs(cu - ou) < (obj_w + ow) / 2 and
                            abs(cv - ov) < (obj_h + oh) / 2):
                            collision = True
                            break

                if collision:
                    continue

            return candidate

        return None

    def __repr__(self) -> str:
        return (f"ResetWorkspace(reach=[{self.min_reach:.2f}, {self.max_reach:.2f}], "
                f"z_fixed={self.z_fixed:.2f})")


# ============================================================
# Helper Functions
# ============================================================

# bbox가 커도 가장자리를 잡을 수 있는 deformable 물체 키워드
DEFORMABLE_KEYWORDS = ("towel", "cloth", "fabric", "napkin", "sheet")


def is_grippable(
    bbox_px: Tuple[int, int],
    gripper_max_px: int = GRIPPER_MAX_OPEN_PX,
) -> bool:
    """
    그리퍼로 잡을 수 있는 물체인지 판단 (픽셀 bbox 기반).

    Args:
        bbox_px: (width_px, height_px) 픽셀 크기
        gripper_max_px: 그리퍼 최대 열림 폭 (pixels)

    Returns:
        True if object can be gripped
    """
    if bbox_px is None:
        return True
    return min(bbox_px) < gripper_max_px


def classify_objects(
    detections: Dict[str, dict],
    gripper_max_px: int = GRIPPER_MAX_OPEN_PX,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    객체를 grippable / non-grippable(obstacle)로 분류 (픽셀 bbox 기반).

    Deformable 물체 (이름에 towel, cloth 등 포함)는 bbox가 커도 grippable로 분류.

    Args:
        detections: {name: {"position": [...], "bbox_px": (w,h), ...}}
        gripper_max_px: 그리퍼 최대 열림 폭 (pixels)

    Returns:
        (grippable_objects, obstacle_objects)
    """
    grippable = {}
    obstacles = {}

    for name, info in detections.items():
        if info is None:
            continue
        # Deformable 물체는 bbox 크기와 무관하게 grippable
        if any(kw in name.lower() for kw in DEFORMABLE_KEYWORDS):
            grippable[name] = info
            continue
        bbox_px = info.get("bbox_px")
        if is_grippable(bbox_px, gripper_max_px):
            grippable[name] = info
        else:
            obstacles[name] = info

    return grippable, obstacles


def _get_bbox_px(info: dict, default: Tuple[int, int] = (30, 30)) -> Tuple[int, int]:
    """객체 info에서 bbox 픽셀 크기(w, h) 추출."""
    bbox = info.get("bbox_px")
    if bbox is not None:
        return tuple(bbox)
    # box_2d [ymin, xmin, ymax, xmax] (0-1000 스케일) → 픽셀 크기 추정
    box = info.get("box_2d")
    if box is not None and len(box) == 4:
        ymin, xmin, ymax, xmax = box
        # 0-1000 스케일 → 대략 640x480 이미지 기준
        w = int((xmax - xmin) * 640 / 1000)
        h = int((ymax - ymin) * 480 / 1000)
        return (max(w, 10), max(h, 10))
    return default


def _get_center_px(info: dict, pix2robot) -> Optional[Tuple[int, int]]:
    """객체 info에서 center 픽셀 좌표 추출. pixel이 있으면 사용, 없으면 robot→pixel 변환."""
    px = info.get("pixel")
    if px is not None:
        return (int(px[0]), int(px[1]))
    pos = info.get("position")
    if pos is not None and pix2robot is not None:
        try:
            return pix2robot.robot_to_pixel(pos[0], pos[1])
        except Exception:
            pass
    return None


def generate_random_positions(
    grippable_objects: Dict[str, dict],
    obstacle_objects: Dict[str, dict],
    initial_positions: Dict[str, List[float]],
    workspace: ResetWorkspace = None,
    pix2robot=None,
    seed: int = None,
    max_attempts: int = 500,
    bbox_margin_px: int = 10,
    current_positions: Dict[str, dict] = None,
    current_positions_margin_px: int = 15,
    exclusion_zones: Optional[List[dict]] = None,
    resetspace: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    랜덤 위치 생성 (IoU 기반 충돌 검증).

    조건:
    1. workspace 범위 내 (reach + IK 검증)
    2. non-grippable: 겹침 불허 + margin
    3. 이미 배치된 다른 grippable 객체와 bbox 겹치지 않음
    4. 초기 위치에서 min_displacement_from_initial 이상 떨어져야 함

    Args:
        grippable_objects: 이동할 객체들 {name: {"position": [...], "bbox_px": (w,h), ...}}
        obstacle_objects: 장애물 객체들 (위치 고정)
        initial_positions: 객체들의 초기 위치
        workspace: ResetWorkspace 인스턴스
        pix2robot: Pix2RobotCalibrator 인스턴스 (robot↔pixel 변환)
        seed: 재현성을 위한 random seed
        max_attempts: 위치당 최대 시도 횟수
        bbox_margin_px: bbox 충돌 마진 (pixels)

    Returns:
        {object_name: [x, y, z]} 랜덤 타겟 위치
    """
    if workspace is None:
        workspace = ResetWorkspace()

    if seed is not None:
        np.random.seed(seed)

    # pix2robot 자동 로드
    if pix2robot is None:
        try:
            from pix2robot_calibrator import Pix2RobotCalibrator
            from pathlib import Path
            for rid in [2, 3]:
                p = Path(__file__).parent.parent.parent / "robot_configs" / "pix2robot_matrices" / f"robot{rid}_pix2robot_data.npz"
                if p.exists():
                    pix2robot = Pix2RobotCalibrator(robot_id=rid)
                    if not pix2robot.load(str(p)):
                        pix2robot = None
                    else:
                        break
        except Exception:
            pass

    # 장애물 리스트 (픽셀 bbox 기반)
    # allow_overlap: True → IoU ≤ 0.5 허용 (grippable), False → 겹침 불허 + margin (non-grippable)
    occupied = []

    # 1) Non-grippable 객체 (고정 장애물, 겹침 불허 + margin)
    for name, info in obstacle_objects.items():
        if info is None:
            continue
        center_px = _get_center_px(info, pix2robot)
        if center_px is None:
            continue
        bbox_px = _get_bbox_px(info)
        occupied.append({
            "name": name,
            "center_px": center_px,
            "bbox_w": bbox_px[0] + bbox_margin_px * 2,  # 원본 bbox + margin 양쪽
            "bbox_h": bbox_px[1] + bbox_margin_px * 2,
            "allow_overlap": False,  # 겹침 불허
        })

    # 2) Grippable 객체의 현재 위치 (IoU ≤ 0.5 허용)
    for name, info in grippable_objects.items():
        if info is None:
            continue
        center_px = _get_center_px(info, pix2robot)
        if center_px is None:
            continue
        bbox_px = _get_bbox_px(info)
        occupied.append({
            "name": name,
            "center_px": center_px,
            "bbox_w": bbox_px[0],
            "bbox_h": bbox_px[1],
            "allow_overlap": True,  # IoU ≤ 0.5 허용
        })

    # 3) 초기 위치 + 과거 시드 위치 (IoU ≤ 0.5 허용)
    for name, info in initial_positions.items():
        if info is None:
            continue
        init_info = info if isinstance(info, dict) else {"position": info}
        center_px = _get_center_px(init_info, pix2robot)
        if center_px is None:
            continue
        bbox_px = _get_bbox_px(init_info)
        occupied.append({
            "name": f"{name}_initial",
            "center_px": center_px,
            "bbox_w": bbox_px[0],
            "bbox_h": bbox_px[1],
            "allow_overlap": True,  # IoU ≤ 0.5 허용
        })

    # 4) 현재 위치의 물체 (고정 장애물, 겹침 불허 + margin)
    if current_positions is not None:
        for name, info in current_positions.items():
            if info is None or name.startswith("_"):
                continue
            cur_info = info if isinstance(info, dict) else {"position": info}
            center_px = _get_center_px(cur_info, pix2robot)
            if center_px is None:
                continue
            bbox_px = _get_bbox_px(cur_info)
            occupied.append({
                "name": f"{name}_current",
                "center_px": center_px,
                "bbox_w": bbox_px[0] + current_positions_margin_px * 2,
                "bbox_h": bbox_px[1] + current_positions_margin_px * 2,
                "allow_overlap": False,  # 겹침 불허 + margin
            })

    # 결과 저장
    target_positions = {}

    for obj_name, obj_info in grippable_objects.items():
        if obj_info is None:
            continue

        # 이 객체의 bbox 픽셀 크기
        obj_bbox_px = _get_bbox_px(obj_info if isinstance(obj_info, dict) else {})

        # 장애물 필터링:
        # - 자기 자신의 현재 위치 제거 (이동할 거니까)
        # - 과거 seed 위치(_pseed): 같은 종류만 유지, 다른 종류는 제거
        #   (chocolate_pie_1과 chocolate_pie_2는 같은 종류 → 둘 다 비교)
        # - 그 외 (obstacle, 현재 seed 내 확정 위치): 전부 유지
        import re
        obj_type = re.sub(r'_?\d+$', '', obj_name)  # chocolate_pie_1 → chocolate_pie
        def _pseed_type(n):
            return re.sub(r'_?\d+$', '', n.split('_pseed')[0])
        obstacles_for_this = [
            occ for occ in occupied
            if occ["name"] != obj_name and (
                "_pseed" not in occ["name"] or _pseed_type(occ["name"]) == obj_type
            )
        ]

        # 랜덤 위치 생성 (IK + FOV + IoU 기반 충돌 검증)
        position = workspace.generate_random_position(
            obstacles=obstacles_for_this,
            obj_bbox_px=obj_bbox_px,
            pix2robot=pix2robot,
            max_attempts=max_attempts,
            exclusion_zones=exclusion_zones,
            resetspace=resetspace,
        )

        if position is not None:
            target_positions[obj_name] = position
            # 새 위치로 장애물 업데이트 (같은 시드 내 물체끼리는 겹침 불허)
            occupied = [occ for occ in occupied if occ["name"] != obj_name]
            if pix2robot is not None:
                try:
                    INTRA_SEED_MARGIN_PX = 15  # 같은 시드 내 물체 간 최소 간격 (pixels)
                    new_px = pix2robot.robot_to_pixel(position[0], position[1])
                    occupied.append({
                        "name": obj_name,
                        "center_px": new_px,
                        "bbox_w": obj_bbox_px[0] + INTRA_SEED_MARGIN_PX * 2,
                        "bbox_h": obj_bbox_px[1] + INTRA_SEED_MARGIN_PX * 2,
                        "allow_overlap": False,  # 같은 시드 내 물체끼리 겹침 불허 + margin
                    })
                except Exception:
                    pass
        else:
            # 유효 위치를 찾지 못함 → 빈 dict 반환 (workspace 포화)
            print(f"[Warning] Could not find valid position for '{obj_name}' — workspace saturated")
            return {}

    return target_positions


# ============================================================
# Workspace Bounds & Visualization
# ============================================================

def compute_workspace_bounds(
    workspace: BaseWorkspace = None,
    z_table: float = 0.01,
    step: float = 0.01,
    x_scan: Tuple[float, float] = (0.05, 0.45),
    y_scan: Tuple[float, float] = (-0.35, 0.20),
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    IK 그리드 샘플링으로 로봇별 도달 가능 workspace 경계를 자동 계산.

    Returns:
        ((x_min, x_max), (y_min, y_max)) in meters (base_link frame)
    """
    if workspace is None:
        workspace = BaseWorkspace()

    valid_x, valid_y = [], []
    for x in np.arange(x_scan[0], x_scan[1], step):
        for y in np.arange(y_scan[0], y_scan[1], step):
            if workspace.is_reachable(np.array([x, y, z_table])):
                valid_x.append(x)
                valid_y.append(y)

    if not valid_x:
        return (0.12, 0.40), (-0.30, 0.15)

    return (min(valid_x), max(valid_x)), (min(valid_y), max(valid_y))


def draw_workspace_on_image(
    image: np.ndarray,
    robot_id: int = 3,
    pix2robot_calibrator=None,
    workspace_bounds=None,
    coord_transformer=None,
    resetspace: Optional[str] = None,
) -> np.ndarray:
    """
    로봇 워크스페이스를 이미지에 시각화 (pix2robot 직접 매핑 기반).

    시각적 요소:
    - 도달 가능 영역 밝게 / 불가 영역 어둡게 (convex hull 마스킹)
    - CYAN 점선: min_reach / max_reach 원호 (로봇 base 중심)
    - resetspace: 지정된 quadrant 외 영역을 어둡게 처리 + 경계선 + 라벨

    Args:
        image: BGR 이미지 (numpy array)
        robot_id: 로봇 ID
        pix2robot_calibrator: Pix2RobotCalibrator 인스턴스 (우선 사용)
        workspace_bounds: (legacy, 미사용) 호환성 유지
        coord_transformer: (legacy, 미사용) 호환성 유지
        resetspace: reset quadrant 제약. 지정 시 해당 quadrant 외 영역 어둡게.

    Returns:
        시각화된 이미지 (numpy array, copy)
    """
    import cv2

    img_h, img_w = image.shape[:2]
    result = image.copy()

    # ── Pix2Robot 로드 (없으면 자동 로드 시도) ──
    p2r = pix2robot_calibrator
    if p2r is None:
        try:
            from pix2robot_calibrator import Pix2RobotCalibrator
            calib_path = (
                Path(__file__).parent.parent.parent
                / "robot_configs" / "pix2robot_matrices" / f"robot{robot_id}_pix2robot_data.npz"
            )
            if calib_path.exists():
                p2r = Pix2RobotCalibrator(robot_id=robot_id)
                if not p2r.load(str(calib_path)):
                    p2r = None
        except Exception:
            pass

    if p2r is None:
        return result

    def robot_to_px(x, y):
        """로봇 좌표 → 픽셀 좌표 (범위 밖이면 None)"""
        try:
            u, v = p2r.robot_to_pixel(x, y)
            if 0 <= u < img_w and 0 <= v < img_h:
                return (u, v)
        except Exception:
            pass
        return None

    # ── Workspace 파라미터 ──
    ws = BaseWorkspace()
    min_reach = ws.min_reach
    max_reach = ws.max_reach
    margin = 0.01

    # ── 도달 가능 영역 마스크 생성 (픽셀 순회 방식) ──
    # 모든 픽셀 → pixel_to_robot → 거리 계산 → 도넛 범위 내인지 판정
    ws_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    step_px = 2  # 2px 간격으로 샘플링 (속도 vs 정밀도)

    for v in range(0, img_h, step_px):
        for u in range(0, img_w, step_px):
            try:
                rx, ry, _ = p2r.pixel_to_robot(u, v)
                dist = np.sqrt(rx * rx + ry * ry)
                if (min_reach + margin) <= dist <= (max_reach - margin):
                    # step_px 크기의 사각형으로 채움
                    ws_mask[v:v+step_px, u:u+step_px] = 255
            except Exception:
                continue

    # ── 도달 불가 영역 검정 마스킹 ──
    if np.any(ws_mask):
        result[ws_mask == 0] = (result[ws_mask == 0] * 0.4).astype(np.uint8)

    # ── Resetspace quadrant 마스킹 ──
    if resetspace is not None and resetspace != "all":
        bounds = QUADRANT_DEFINITIONS.get(resetspace)
        if bounds is not None:
            u_min, u_max = bounds["u_range"]
            v_min, v_max = bounds["v_range"]
            for v in range(0, img_h, step_px):
                for u in range(0, img_w, step_px):
                    if ws_mask[v, u] == 0:
                        continue  # 이미 도달 불가로 어두워진 영역은 스킵
                    if not (u_min <= u < u_max and v_min <= v < v_max):
                        result[v:v+step_px, u:u+step_px] = (
                            image[v:v+step_px, u:u+step_px] * 0.4
                        ).astype(np.uint8)
                        ws_mask[v:v+step_px, u:u+step_px] = 0

    # ── Free state EE exclusion zone (반경 8cm) 마스킹 ──
    FREE_STATE_EXCLUSION_RADIUS = 0.08
    free_ee = None
    try:
        from lerobot_cap.kinematics import load_calibration_limits as _load_cl
        free_state_path = (
            Path(__file__).parent.parent.parent
            / "robot_configs" / "free_state" / f"robot{robot_id}_free_state.json"
        )
        calib_path_ee = (
            Path(__file__).parent.parent.parent
            / "robot_configs" / "motor_calibration" / "so101" / f"robot{robot_id}_calibration.json"
        )
        urdf_path_ee = (
            Path(__file__).parent.parent.parent
            / "assets" / "urdf" / f"so101_robot{robot_id}.urdf"
        )
        if free_state_path.exists() and calib_path_ee.exists() and urdf_path_ee.exists():
            import json as _json
            with open(free_state_path) as f:
                free_norm = np.array(_json.load(f)["initial_state_normalized"])
            _cl = _load_cl(
                str(calib_path_ee),
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            )
            from lerobot_cap.kinematics import KinematicsEngine as _KE
            _kin = _KE(str(urdf_path_ee), end_effector_frame="gripper_frame_link",
                       joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
            free_rad = _cl.normalized_to_radians(free_norm)
            free_ee = _kin.get_ee_position(free_rad)
    except Exception:
        pass

    if free_ee is not None:
        for v in range(0, img_h, step_px):
            for u in range(0, img_w, step_px):
                if ws_mask[v, u] == 0:
                    continue
                try:
                    rx, ry, _ = p2r.pixel_to_robot(u, v)
                    dist_ee = np.sqrt((rx - free_ee[0])**2 + (ry - free_ee[1])**2)
                    if dist_ee < FREE_STATE_EXCLUSION_RADIUS:
                        result[v:v+step_px, u:u+step_px] = (
                            image[v:v+step_px, u:u+step_px] * 0.4
                        ).astype(np.uint8)
                        ws_mask[v:v+step_px, u:u+step_px] = 0  # 유효 영역에서도 제거
                except Exception:
                    continue

    COLOR_CYAN = (255, 255, 0)

    # ── Min reach 원호 (Cyan 점선) — exclusion zone 겹침 제거 ──
    for angle in range(0, 360, 3):
        rad = np.radians(angle)
        x = (min_reach + margin) * np.cos(rad)
        y = (min_reach + margin) * np.sin(rad)
        # exclusion zone과 겹치면 스킵
        if free_ee is not None:
            dist_ee = np.sqrt((x - free_ee[0])**2 + (y - free_ee[1])**2)
            if dist_ee < FREE_STATE_EXCLUSION_RADIUS + 0.01:
                continue
        px = robot_to_px(x, y)
        if px is not None:
            cv2.circle(result, px, 2, COLOR_CYAN, -1)

    # ── Max reach 원호 (Cyan 점선) ──
    for angle in range(0, 360, 2):
        rad = np.radians(angle)
        x = (max_reach - margin) * np.cos(rad)
        y = (max_reach - margin) * np.sin(rad)
        px = robot_to_px(x, y)
        if px is not None:
            cv2.circle(result, px, 2, COLOR_CYAN, -1)

    # ── Exclusion zone 경계 (Cyan 점선) — reach 안쪽만 ──
    if free_ee is not None:
        for angle in range(0, 360, 6):
            rad = np.radians(angle)
            x = free_ee[0] + FREE_STATE_EXCLUSION_RADIUS * np.cos(rad)
            y = free_ee[1] + FREE_STATE_EXCLUSION_RADIUS * np.sin(rad)
            dist_origin = np.sqrt(x*x + y*y)
            if (min_reach + margin) <= dist_origin <= (max_reach - margin):
                px = robot_to_px(x, y)
                if px is not None:
                    cv2.circle(result, px, 2, COLOR_CYAN, -1)

    # ── 가장자리 마진 사각형 (Green) ──
    edge_margin = 30
    COLOR_GREEN = (0, 255, 0)
    cv2.rectangle(result,
                  (edge_margin, edge_margin),
                  (img_w - edge_margin - 1, img_h - edge_margin - 1),
                  COLOR_GREEN, 1)

    # ── Resetspace quadrant 경계선 (Yellow 점선) ──
    COLOR_YELLOW = (0, 255, 255)
    if resetspace is not None and resetspace != "all":
        bounds = QUADRANT_DEFINITIONS.get(resetspace)
        if bounds is not None:
            u_min, u_max = bounds["u_range"]
            v_min, v_max = bounds["v_range"]
            # 점선으로 경계 그리기
            dash_len = 8
            # 수평선 (v_min, v_max)
            for line_v in [v_min, v_max - 1]:
                if 0 < line_v < img_h:
                    for u_start in range(u_min, u_max, dash_len * 2):
                        u_end = min(u_start + dash_len, u_max)
                        cv2.line(result, (u_start, line_v), (u_end, line_v), COLOR_YELLOW, 1)
            # 수직선 (u_min, u_max)
            for line_u in [u_min, u_max - 1]:
                if 0 < line_u < img_w:
                    for v_start in range(v_min, v_max, dash_len * 2):
                        v_end = min(v_start + dash_len, v_max)
                        cv2.line(result, (line_u, v_start), (line_u, v_end), COLOR_YELLOW, 1)

    # ── 라벨 텍스트 ──
    reach_label = f"Reach: [{min_reach:.2f}, {max_reach:.2f}]m"
    cv2.putText(result, reach_label, (10, img_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN, 1, cv2.LINE_AA)
    margin_label = f"Edge margin: {edge_margin}px"
    cv2.putText(result, margin_label, (10, img_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1, cv2.LINE_AA)
    if resetspace is not None and resetspace != "all":
        rs_label = f"Reset: {resetspace}"
        cv2.putText(result, rs_label, (10, img_h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YELLOW, 1, cv2.LINE_AA)

    return result


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("ResetWorkspace Test")
    print("=" * 60)

    # Create workspace
    ws = ResetWorkspace()
    print(f"Workspace: {ws}")
    print(f"Base reach: {ws.get_reach_limits()}")
    print()

    # Test positions
    test_positions = [
        [0.15, 0.0, 0.01],   # Valid
        [0.10, 0.0, 0.01],   # Out of x_range
        [0.17, 0.10, 0.01],  # Out of y_range
        [0.50, 0.0, 0.01],   # Out of reach
    ]

    for pos in test_positions:
        reachable = ws.is_reachable(np.array(pos))
        valid = ws.is_valid(np.array(pos))
        print(f"  {pos} -> reachable={reachable}, valid={valid}")


# =============================================================================
# Multi-Robot Seed Position Generation
# =============================================================================

def generate_multi_robot_seed_positions(
    grippable_objects: Dict[str, dict],
    obstacle_objects: Dict[str, dict],
    initial_positions: Dict[str, List[float]],
    workspaces: Dict[int, "ResetWorkspace"],
    pix2robots: Dict[int, object] = None,
    seed: int = None,
    max_attempts: int = 500,
    bbox_margin_px: int = 10,
    current_positions: Dict[str, dict] = None,
    current_positions_margin_px: int = 15,
    exclusion_zones: Optional[List[dict]] = None,
    previous_seed_positions: Optional[List[Dict]] = None,
    resetspace: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Generate random seed positions for multi-robot setup.

    The workspace is the UNION of all robots' workspaces. A position is valid
    if at least one robot can reach it. This ensures objects are always
    manipulable by at least one arm.

    Args:
        grippable_objects: Objects to move {name: {"position": [...], "bbox_px": (w,h)}}.
        obstacle_objects: Stationary obstacle objects.
        initial_positions: Objects' initial positions.
        workspaces: Per-robot ResetWorkspace instances {robot_id: ResetWorkspace}.
        pix2robots: Per-robot Pix2RobotCalibrator instances.
        seed: Random seed for reproducibility.
        max_attempts: Max attempts per object.
        bbox_margin_px: Margin for non-grippable object collision check.
        current_positions: Current object positions for collision avoidance.
        current_positions_margin_px: Margin for current position collision.
        exclusion_zones: Zones to avoid (e.g., free state EE positions).
        previous_seed_positions: Previous seed positions to avoid collision.

    Returns:
        {object_name: [x, y, z]} generated positions.
    """
    robot_ids = list(workspaces.keys())

    if not robot_ids:
        # Fallback to single-robot generate_random_positions
        return generate_random_positions(
            grippable_objects=grippable_objects,
            obstacle_objects=obstacle_objects,
            initial_positions=initial_positions,
            workspace=None,
            seed=seed,
            max_attempts=max_attempts,
            bbox_margin_px=bbox_margin_px,
            current_positions=current_positions,
            current_positions_margin_px=current_positions_margin_px,
            exclusion_zones=exclusion_zones,
            resetspace=resetspace,
        )

    # Use first robot's pix2robot for pixel↔position conversion
    # (shared camera means same pixel space)
    primary_pix2robot = None
    if pix2robots:
        for rid in robot_ids:
            if rid in pix2robots and pix2robots[rid] is not None:
                primary_pix2robot = pix2robots[rid]
                break

    # Create a union workspace that accepts if ANY robot can reach
    class UnionWorkspace(ResetWorkspace):
        """Workspace that is valid if any robot's workspace accepts."""
        def __init__(self, ws_list):
            # Initialize with first workspace's parameters
            first_ws = ws_list[0] if ws_list else None
            super().__init__(
                kinematics_engine=first_ws._kinematics if first_ws else None,
                frame_transformer=first_ws._frame_transformer if first_ws else None,
            )
            self._workspaces = ws_list

        def is_valid(self, position: np.ndarray) -> bool:
            return any(ws.is_valid(position) for ws in self._workspaces)

        def is_reachable(self, position: np.ndarray) -> bool:
            return any(ws.is_reachable(position) for ws in self._workspaces)

    union_ws = UnionWorkspace(list(workspaces.values()))

    # Merge exclusion zones from all robots' free states
    all_exclusion_zones = list(exclusion_zones or [])

    return generate_random_positions(
        grippable_objects=grippable_objects,
        obstacle_objects=obstacle_objects,
        initial_positions=initial_positions,
        workspace=union_ws,
        pix2robot=primary_pix2robot,
        seed=seed,
        max_attempts=max_attempts,
        bbox_margin_px=bbox_margin_px,
        current_positions=current_positions,
        current_positions_margin_px=current_positions_margin_px,
        exclusion_zones=all_exclusion_zones,
        resetspace=resetspace,
    )

    print()

    # Test random position generation
    grippable = {
        "green block": {"position": [0.15, 0.05, 0.01], "bbox_size_m": [0.03, 0.03]},
        "red cube": {"position": [0.18, -0.02, 0.01], "bbox_size_m": [0.025, 0.025]},
    }

    obstacles = {
        "blue dish": {"position": [0.20, -0.03, 0.01], "bbox_size_m": [0.12, 0.10]},
    }

    initial_positions = {
        "green block": [0.15, 0.05, 0.01],
        "red cube": [0.18, -0.02, 0.01],
    }

    print(f"Grippable: {list(grippable.keys())}")
    print(f"Obstacles: {list(obstacles.keys())}")

    random_targets = generate_random_positions(
        grippable_objects=grippable,
        obstacle_objects=obstacles,
        initial_positions=initial_positions,
        workspace=ws,
        seed=42,
    )

    print("\nGenerated positions:")
    for name, pos in random_targets.items():
        initial = initial_positions.get(name, [0, 0, 0])
        displacement = np.sqrt((pos[0] - initial[0])**2 + (pos[1] - initial[1])**2)
        print(f"  {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] (disp: {displacement*100:.1f}cm)")
