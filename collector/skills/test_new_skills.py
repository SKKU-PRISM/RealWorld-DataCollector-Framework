"""
모듈형 스킬 통합 테스트 (8개 스킬).
press, pull, insert, stir, shake, strike, push_object, wipe 순서로 실행.

모듈형 패턴: 외부(LLM 코드) approach/grasp → core skill → 외부 retreat
"""

import importlib.util
import pathlib
import sys
import types

# skills.py (레거시) 가 skills 패키지를 덮어쓰는 문제 회피
# 직접 파일 경로로 모듈 로드
_SKILLS_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SKILLS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(_SKILLS_DIR / filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# skills 패키지 네임스페이스를 수동 구성 (skills.py 충돌 회피)
_skills_pkg = types.ModuleType("skills")
_skills_pkg.__path__ = [str(_SKILLS_DIR)]
sys.modules["skills"] = _skills_pkg

# 의존성 순서대로 로드
_sl = _load("skills.skills_lerobot", "skills_lerobot.py")
_skills_pkg.skills_lerobot = _sl
LeRobotSkills = _sl.LeRobotSkills

_ml = _load("skills.move_linear", "move_linear.py")
_skills_pkg.move_linear = _ml

_map = _load("skills.move_approach_position", "move_approach_position.py")
_skills_pkg.move_approach_position = _map
move_approach_position = _map.move_approach_position

# 8개 스킬 로드
press = _load("skills.press", "press.py").press
pull = _load("skills.pull", "pull.py").pull
insert = _load("skills.insert", "insert.py").insert
stir = _load("skills.stir", "stir.py").stir
shake = _load("skills.shake", "shake.py").shake
strike = _load("skills.strike", "strike.py").strike
push_object = _load("skills.push_object", "push_object.py").push_object
wipe = _load("skills.wipe", "wipe.py").wipe


def main():
    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot3.yaml",
        frame="world",
    )
    if not skills.connect():
        print("Robot connection failed")
        return

    results = {}

    try:
        skills.move_to_initial_state()

        # ============================================================
        # TEST 1: press — 버튼 누르기 (토크 제한 400)
        # 외부: gripper_close + approach → core press → 외부: lift
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 1: press (1cm depth, 토크 제한 400)")
        print("=" * 60)
        # 외부: approach
        skills.gripper_close(skill_description="test press: close gripper")
        move_approach_position(
            skills, object_position=[0.20, 0.0, 0.03],
            approach_height=0.10, object_name="test button",
        )
        # Core press
        success = press(
            skills,
            position=[0.20, 0.0, 0.03],
            press_depth=0.01,
            contact_height=0.03,
            hold_time=0.3,
            max_press_torque=400,
            target_name="test button",
        )
        # 외부: retreat
        skills.move_to_position(
            [0.20, 0.0, 0.10],
            skill_description="test press: lift after press",
        )
        results["press"] = success
        print(f"  press result: {'SUCCESS' if success else 'FAILED'}")

        skills.move_to_initial_state()

        # ============================================================
        # TEST 2: pull — 로봇 쪽으로 5cm 당기기
        # 외부: approach + pick → core pull → 외부: release + lift
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 2: pull (로봇 쪽으로 5cm 당기기)")
        print("=" * 60)
        test_pos = [0.22, 0.0, 0.03]
        # 외부: approach & grasp
        skills.gripper_open()
        move_approach_position(
            skills, object_position=test_pos,
            approach_height=0.10, object_name="test object",
        )
        skills.move_to_position(test_pos, maintain_pitch=True)
        skills.gripper_close()
        # Core pull
        success = pull(
            skills,
            position=test_pos,
            pull_direction=[-1, 0],
            pull_distance=0.05,
            object_name="test object",
        )
        # 외부: release & retreat
        skills.gripper_open()
        skills.move_to_position(
            [test_pos[0], test_pos[1], 0.10],
            skill_description="test pull: lift after pull",
        )
        results["pull"] = success
        print(f"  pull result: {'SUCCESS' if success else 'FAILED'}")

        skills.move_to_initial_state()

        # ============================================================
        # TEST 3: insert — 2cm 깊이 삽입 (토크 제한 400)
        # 외부: gripper_close + approach → core insert → 외부: lift
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 3: insert (2cm 삽입, 토크 제한 400)")
        print("=" * 60)
        hole_pos = [0.20, 0.0, 0.03]
        # 외부: approach (물체 잡은 상태)
        skills.gripper_close()
        move_approach_position(
            skills, object_position=hole_pos,
            approach_height=0.10, object_name="test hole",
        )
        # Core insert
        success = insert(
            skills,
            position=hole_pos,
            insert_depth=0.02,
            align_height=0.04,
            max_insert_torque=400,
            target_name="test hole",
        )
        # 외부: retreat
        skills.move_to_position(
            [hole_pos[0], hole_pos[1], 0.10],
            skill_description="test insert: lift after insert",
        )
        results["insert"] = success
        print(f"  insert result: {'SUCCESS' if success else 'FAILED'}")

        skills.move_to_initial_state()

        # ============================================================
        # TEST 4: stir — 반지름 3cm, 2바퀴 원형 젓기
        # 외부: gripper_close + approach → core stir → 외부: lift
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 4: stir (반지름 3cm, 2바퀴)")
        print("=" * 60)
        cup_center = [0.20, 0.0, 0.03]
        # 외부: approach (숟가락 잡은 상태)
        skills.gripper_close()
        move_approach_position(
            skills, object_position=cup_center,
            approach_height=0.10, object_name="test cup",
        )
        # Core stir
        success = stir(
            skills,
            center=cup_center,
            stir_radius=0.03,
            num_rotations=2,
            stir_height=0.03,
            segments_per_rotation=8,
            object_name="test cup",
        )
        # 외부: lift
        skills.move_to_position(
            [cup_center[0], cup_center[1], 0.10],
            skill_description="test stir: lift after stir",
        )
        results["stir"] = success
        print(f"  stir result: {'SUCCESS' if success else 'FAILED'}")

        skills.move_to_initial_state()

        # ============================================================
        # TEST 5: shake — y축 방향 6회(3왕복) 흔들기
        # 외부: approach + pick → core shake → 외부: place + release
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 5: shake (y축 6회 흔들기)")
        print("=" * 60)
        bottle_pos = [0.20, 0.0, 0.03]
        # 외부: approach & grasp (물체 잡기)
        skills.gripper_open()
        move_approach_position(
            skills, object_position=bottle_pos,
            approach_height=0.10, object_name="test bottle",
        )
        skills.move_to_position(bottle_pos, maintain_pitch=True)
        skills.gripper_close()
        # Core shake (내부에서 lift → oscillation → center)
        success = shake(
            skills,
            position=bottle_pos,
            shake_axis="y",
            shake_amplitude=0.03,
            num_shakes=6,
            shake_height=0.10,
            object_name="test bottle",
        )
        # 외부: place & release
        skills.move_to_position(bottle_pos, maintain_pitch=True)
        skills.gripper_open()
        skills.move_to_position(
            [bottle_pos[0], bottle_pos[1], 0.10],
            skill_description="test shake: lift after shake",
        )
        results["shake"] = success
        print(f"  shake result: {'SUCCESS' if success else 'FAILED'}")

        skills.move_to_initial_state()

        # ============================================================
        # TEST 6: strike — 3회 반복 타격 (토크 제한 500)
        # 외부: gripper_close + approach → core strike → 외부: lift
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 6: strike (3회 타격, 토크 제한 500)")
        print("=" * 60)
        target_pos = [0.20, 0.0, 0.03]
        # 외부: approach (도구 잡은 상태)
        skills.gripper_close()
        move_approach_position(
            skills, object_position=target_pos,
            approach_height=0.10, object_name="test target",
        )
        # Core strike
        success = strike(
            skills,
            position=target_pos,
            strike_height=0.02,
            wind_up_height=0.06,
            num_strikes=3,
            max_strike_torque=500,
            target_name="test target",
        )
        # 외부: retreat
        skills.move_to_position(
            [target_pos[0], target_pos[1], 0.10],
            skill_description="test strike: lift after strike",
        )
        results["strike"] = success
        print(f"  strike result: {'SUCCESS' if success else 'FAILED'}")

        skills.move_to_initial_state()

        # ============================================================
        # TEST 7: push_object — y축 방향 10cm 밀기
        # 외부: gripper_close + approach → core push → 외부: lift
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 7: push_object (y축 10cm 밀기)")
        print("=" * 60)
        push_start = [0.20, -0.05, 0.03]
        push_end = [0.20, 0.05, 0.03]
        # 외부: approach
        skills.gripper_close()
        move_approach_position(
            skills, object_position=push_start,
            approach_height=0.10, object_name="test block",
        )
        # Core push
        success = push_object(
            skills,
            start_position=push_start,
            end_position=push_end,
            push_height=0.01,
            object_name="test block",
        )
        # 외부: retreat
        skills.move_to_position(
            [push_end[0], push_end[1], 0.10],
            skill_description="test push: lift after push",
        )
        results["push_object"] = success
        print(f"  push_object result: {'SUCCESS' if success else 'FAILED'}")

        skills.move_to_initial_state()

        # ============================================================
        # TEST 8: wipe — y축 방향 3회 왕복 닦기
        # 외부: gripper_close + approach → core wipe → 외부: lift
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 8: wipe (y축 3회 왕복)")
        print("=" * 60)
        wipe_start = [0.20, -0.05, 0.0]
        wipe_end = [0.20, 0.05, 0.0]
        # 외부: approach
        skills.gripper_close()
        move_approach_position(
            skills, object_position=wipe_start,
            approach_height=0.10, object_name="table surface",
        )
        # Core wipe
        success = wipe(
            skills,
            start_position=wipe_start,
            end_position=wipe_end,
            num_strokes=3,
            wipe_height=0.01,
            object_name="table surface",
        )
        # 외부: retreat
        skills.move_to_position(
            [wipe_end[0], wipe_end[1], 0.10],
            skill_description="test wipe: lift after wipe",
        )
        results["wipe"] = success
        print(f"  wipe result: {'SUCCESS' if success else 'FAILED'}")

        # ============================================================
        # 결과 요약
        # ============================================================
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETE")
        print("=" * 60)
        for name, result in results.items():
            print(f"  {name:15s}: {'SUCCESS' if result else 'FAILED'}")
        total = sum(results.values())
        print(f"\n  {total}/{len(results)} passed")

        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()


if __name__ == "__main__":
    main()
