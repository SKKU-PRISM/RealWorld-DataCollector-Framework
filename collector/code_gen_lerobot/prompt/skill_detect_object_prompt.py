"""
detect_objects 스킬 전용 프롬프트

코드 실행 중 실시간 객체 re-detection을 위한 간결한 프롬프트.
코드 생성 파이프라인의 Turn 1/2와 동일한 역할이지만,
독립 호출(stateless)에 최적화되어 불필요한 컨텍스트를 제거.
"""


def detect_t1_prompt(object_names: list) -> str:
    """Turn 1: bbox detection 프롬프트.

    Args:
        object_names: 검출할 객체 이름 리스트

    Returns:
        프롬프트 문자열
    """
    labels_str = ", ".join(f'"{q}"' for q in object_names)
    return f"""Detect the following objects in the overhead camera image: [{labels_str}]

Return a JSON array with bounding boxes in normalized 0-1000 coordinates:
```json
[{{"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"}}]
```
Only include clearly visible objects. Use exact label names."""


def detect_t2_prompt(object_label: str, scene_summary: str = "", point_labels: list = None) -> str:
    """Turn 2: crop 이미지에서 critical point detection 프롬프트.

    Args:
        object_label: 객체 이름
        scene_summary: Turn 0 장면 분석 요약 (있으면 컨텍스트로 prepend)
        point_labels: 검출할 포인트 라벨 리스트 (Turn 2에서 사용된 라벨)
                     None이면 기본 "grasp center" 1개만 요청

    Returns:
        프롬프트 문자열
    """
    context = f"Scene context:\n{scene_summary}\n\n" if scene_summary else ""

    if point_labels and len(point_labels) > 0:
        # 기존 Turn 2 라벨을 사용하여 동일한 포인트들을 요청
        points_example = ", ".join(
            f'{{"point_2d": [y, x], "label": "{lbl}", "role": "{"grasp" if i == 0 else "interaction"}"}}'
            for i, lbl in enumerate(point_labels)
        )
        labels_str = ", ".join(f'"{lbl}"' for lbl in point_labels)
        return f"""{context}This is a cropped close-up of "{object_label}" from the overhead camera.

Identify the following critical points on the object: [{labels_str}]
The robot gripper approaches from directly above.
- For grasp points: find the center of the object's **top surface** as seen from above.
- For other points: find the specified surface/feature point as seen from above.

Return a JSON block:
```json
{{"critical_points": [{points_example}]}}
```
**CRITICAL**: You MUST use exactly these label names: [{labels_str}]. Do NOT rename or paraphrase.
Coordinates are [y, x] normalized to 0-1000 relative to this cropped image."""
    else:
        return f"""{context}This is a cropped close-up of "{object_label}" from the overhead camera.

Identify the **grasp point** — the center of the object's **top surface** as seen from above.
The robot gripper approaches from directly above, so the grasp point must be on the widest visible top face.
Do NOT pick the volumetric center of the object — pick the center of the top surface visible in this image.

Return a JSON block:
```json
{{"critical_points": [{{"point_2d": [y, x], "label": "grasp center", "role": "grasp"}}]}}
```
Coordinates are [y, x] normalized to 0-1000 relative to this cropped image."""
