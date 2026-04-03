"""
save_logs — 파이프라인 실행 결과 로그/시각화 저장.

싱글/멀티암 공용. 각 함수는 stateless (self 불필요).
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional


# ─────────────────────────────────────────────
# Turn text logs
# ─────────────────────────────────────────────

def save_turn_logs(save_dir: str, mt_info: Dict) -> None:
    """Turn 0~3 텍스트 로그 저장."""
    fwd = Path(save_dir)

    # Turn 0
    turn0_raw = mt_info.get("turn0_response", "")
    if turn0_raw:
        lines = ["=" * 60, "Turn 0: Scene Understanding", "=" * 60, "", turn0_raw]
        (fwd / "turn0_log.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"  Turn 0 log saved: {fwd / 'turn0_log.txt'}")

    # Turn 1
    turn1_raw = mt_info.get("turn1_response", "")
    if turn1_raw:
        lines = ["=" * 60, "Turn 1: Bounding Box Detection", "=" * 60, "", turn1_raw]
        t1_parsed = mt_info.get("turn1_parsed")
        if t1_parsed:
            lines.append("\n[Parsed Summary]")
            obj_list = (
                t1_parsed if isinstance(t1_parsed, list)
                else t1_parsed.get("objects", []) if isinstance(t1_parsed, dict)
                else []
            )
            for obj in obj_list:
                name = obj.get("label") or obj.get("name", "?")
                box = obj.get("box_2d") or obj.get("bbox_pixel", "N/A")
                lines.append(f"  - {name}: bbox={box}")
        (fwd / "turn1_log.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"  Turn 1 log saved: {fwd / 'turn1_log.txt'}")

    # Turn 2+
    all_points = mt_info.get("all_points", [])
    crop_responses = mt_info.get("crop_responses", [])
    if all_points or crop_responses:
        lines = ["=" * 60, "Turn 2+: Crop-then-Point", "=" * 60, ""]
        for cr in crop_responses:
            lines.append(f"--- Crop: {cr.get('label', '?')} ---")
            lines.append(cr.get("response", ""))
            lines.append("")
        lines.append("[Parsed Points Summary]")
        for pt in all_points:
            obj = pt.get("object_label", "?")
            label = pt.get("label", "?")
            role = pt.get("role", "?")
            px, py = pt.get("px", 0), pt.get("py", 0)
            reasoning = pt.get("reasoning", "")
            lines.append(f"  - {obj}: {label} ({role}) pixel=({px},{py})")
            if reasoning:
                lines.append(f"    reasoning: {reasoning}")
        (fwd / "turn2_log.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"  Turn 2+ log saved: {fwd / 'turn2_log.txt'}")

    # Turn 3
    turn3_raw = mt_info.get("turn3_response", "")
    if turn3_raw:
        lines = ["=" * 60, "Turn 3: Code Generation", "=" * 60, "", turn3_raw]
        (fwd / "turn3_log.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"  Turn 3 log saved: {fwd / 'turn3_log.txt'}")


# ─────────────────────────────────────────────
# Multi-turn info + LLM cost + crop images
# ─────────────────────────────────────────────

def save_multi_turn_info(save_dir: str, mt_info: Dict, phase: str = "forward") -> None:
    """multi_turn_info.json + llm_cost.json + crop 이미지 + turn 로그 저장.

    Args:
        save_dir: 저장 디렉토리 (forward/ 또는 reset/)
        mt_info: multi_turn_info dict
        phase: "forward" 또는 "reset"
    """
    if not mt_info:
        return

    fwd = Path(save_dir)

    # 1. multi_turn_info.json
    mt_save = {k: v for k, v in mt_info.items() if k != "crop_dir"}
    with open(fwd / "multi_turn_info.json", 'w', encoding='utf-8') as f:
        json.dump(mt_save, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Multi-turn info saved: {fwd / 'multi_turn_info.json'}")

    # 2. LLM cost
    llm_cost = mt_info.get("llm_cost")
    if llm_cost:
        with open(fwd / "llm_cost.json", 'w') as f:
            json.dump({"phase": phase, **llm_cost}, f, indent=2)
        print(f"  LLM cost saved: {fwd / 'llm_cost.json'}")

    # 3. Crop images
    crop_dir = mt_info.get("crop_dir")
    if crop_dir and os.path.isdir(crop_dir):
        for fname in sorted(os.listdir(crop_dir)):
            if fname.endswith(('.jpg', '.png')):
                shutil.copy2(os.path.join(crop_dir, fname), os.path.join(save_dir, fname))
        print(f"  Crop images saved to: {save_dir}")

    # 4. Turn text logs
    save_turn_logs(save_dir, mt_info)


# ─────────────────────────────────────────────
# Turn visualizations (bbox, grasp points, waypoints)
# ─────────────────────────────────────────────

def save_turn_visualizations(
    save_dir: str,
    mt_info: Dict,
    base_image,
    phase: str = "forward",
) -> None:
    """turn1_detection.jpg, turn2_grasp_points.jpg, turn_test 시각화 저장.

    Args:
        save_dir: 저장 디렉토리
        mt_info: multi_turn_info dict
        base_image: 시각화 기준 이미지 (numpy BGR)
        phase: "forward" 또는 "reset"
    """
    if base_image is None or not mt_info:
        return

    fwd = Path(save_dir)
    t1_parsed = mt_info.get("turn1_parsed")
    all_pts = mt_info.get("all_points", [])

    try:
        from execution_forward_and_reset import ForwardAndResetPipeline
        dummy = object.__new__(ForwardAndResetPipeline)

        # Turn 1 bbox visualization
        if t1_parsed:
            kwargs = {"turn1_raw": mt_info.get("turn0_response", "")} if phase == "forward" else {}
            dummy._visualize_turn1(
                base_image.copy(), t1_parsed,
                str(fwd / "turn1_detection.jpg"),
                **kwargs,
            )

        if phase == "forward":
            # Forward: turn2_parsed 직접 사용
            t2_parsed = mt_info.get("turn2_parsed")
            if t2_parsed:
                dummy._visualize_turn2(
                    base_image.copy(), t1_parsed, t2_parsed,
                    str(fwd / "turn2_grasp_points.jpg"),
                )

            # Waypoint visualization
            oh_waypoints = mt_info.get("turn_test_overhead_waypoints")
            if oh_waypoints:
                dummy._visualize_turn_test(
                    base_image.copy(), t2_parsed, oh_waypoints,
                    str(fwd / "turn_test_overhead_waypoints.jpg"),
                )
        else:
            # Reset: all_points → t2_compat 변환
            if t1_parsed and all_pts:
                r_img_h, r_img_w = base_image.shape[:2]
                t2_compat = {"grasp_points": [
                    {"object_name": p["object_label"], "label": p["label"],
                     "role": p["role"],
                     "point_pixel": [
                         int(p["py"] * 1000 / r_img_h),
                         int(p["px"] * 1000 / r_img_w),
                     ]}
                    for p in all_pts
                ]}
                dummy._visualize_turn2(
                    base_image.copy(), t1_parsed, t2_compat,
                    str(fwd / "turn2_grasp_points.jpg"),
                )

    except Exception as e:
        print(f"  [Warning] Turn visualization failed: {e}")


# ─────────────────────────────────────────────
# Batch info
# ─────────────────────────────────────────────

def save_batch_info(episode_dir: str, batch_index: int, slot: int, judge_pred: str) -> None:
    """batch_info.json 저장."""
    bi = {"batch_seed_index": batch_index + 1, "slot": slot, "judge": judge_pred}
    bi_path = Path(episode_dir) / "batch_info.json"
    bi_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bi_path, 'w') as f:
        json.dump(bi, f, indent=2)


# ─────────────────────────────────────────────
# Execution context
# ─────────────────────────────────────────────

def save_execution_context(
    save_dir: str,
    instruction: str,
    positions: Dict,
    code: str,
    success: bool,
    **extra,
) -> None:
    """execution_context.json 저장."""
    ctx = {
        "instruction": instruction,
        "object_positions": positions,
        "generated_code": code,
        "execution_success": success,
        **extra,
    }
    ctx_path = Path(save_dir) / "execution_context.json"
    with open(ctx_path, 'w') as f:
        json.dump(ctx, f, indent=2, default=str)
    print(f"  Execution context saved: {ctx_path}")


# ─────────────────────────────────────────────
# Judge result
# ─────────────────────────────────────────────

def save_judge_result(
    save_dir: str,
    judge_result: Dict,
    initial_image=None,
    final_image=None,
    instruction: str = "",
) -> None:
    """judge_result.json + judge_result.jpg 시각화 저장."""
    fwd = Path(save_dir)

    # JSON
    with open(fwd / "judge_result.json", 'w') as f:
        json.dump(judge_result, f, indent=2, default=str)

    # Visualization image
    if initial_image is not None and final_image is not None:
        try:
            from judge.visualize import create_result_image
            import cv2
            judge_vis = create_result_image(
                initial_image, final_image,
                judge_result.get('prediction', 'UNCERTAIN'),
                judge_result.get('reasoning', ''),
                instruction,
            )
            if judge_vis is not None:
                cv2.imwrite(str(fwd / "judge_result.jpg"), judge_vis)
        except Exception:
            pass


# ─────────────────────────────────────────────
# LLM cost: detect_objects token merge
# ─────────────────────────────────────────────

def update_llm_cost_with_detect_usage(save_dir: str, skills) -> None:
    """detect_objects 토큰 사용량을 llm_cost.json에 합산.

    Args:
        save_dir: llm_cost.json이 있는 디렉토리
        skills: LeRobotSkills 또는 MultiArmSkills
    """
    # Collect usage from skills (single arm or both arms)
    detect_turns = []
    arms = []
    if hasattr(skills, 'left_arm') and hasattr(skills, 'right_arm'):
        # MultiArmSkills
        arms = [skills.left_arm, skills.right_arm]
    else:
        # Single LeRobotSkills
        arms = [skills]

    for arm in arms:
        usage = getattr(arm, '_detect_token_usage', [])
        if usage:
            detect_turns.extend(usage)
            arm._detect_token_usage = []

    if not detect_turns:
        return

    cost_path = Path(save_dir) / "llm_cost.json"
    try:
        if cost_path.exists():
            with open(cost_path) as f:
                llm_cost = json.load(f)
        else:
            llm_cost = {}

        detect_summary = {
            "model": detect_turns[0].get("model", "unknown") if detect_turns else "unknown",
            "inference_time_s": round(sum(t.get("inference_time_s", 0) for t in detect_turns), 2),
            "input_tokens": sum(t.get("input_tokens", 0) for t in detect_turns),
            "output_tokens": sum(t.get("output_tokens", 0) for t in detect_turns),
            "total_tokens": sum(t.get("total_tokens", 0) for t in detect_turns),
            "num_calls": len(detect_turns),
            "turns": detect_turns,
        }
        llm_cost["detect_objects"] = detect_summary

        # Update total and move to end
        old_total = llm_cost.pop("total", {})
        for key in ["input_tokens", "output_tokens", "total_tokens"]:
            old_total[key] = old_total.get(key, 0) + detect_summary.get(key, 0)
        old_total["inference_time_s"] = round(
            old_total.get("inference_time_s", 0) + detect_summary["inference_time_s"], 2)
        llm_cost["total"] = old_total

        with open(cost_path, 'w') as f:
            json.dump(llm_cost, f, indent=2)
        print(f"  detect_objects cost merged into: {cost_path}")
        print(f"    detect calls: {len(detect_turns)}, "
              f"tokens: in={detect_summary['input_tokens']}, out={detect_summary['output_tokens']}")

    except Exception as e:
        print(f"  [Warning] Failed to update llm_cost with detect usage: {e}")
