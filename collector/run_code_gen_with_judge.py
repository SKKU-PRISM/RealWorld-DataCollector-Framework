#!/usr/bin/env python3
"""
LeRobot Code Generation with Judge Pipeline

Detection → Code Generation → Execution → Judge 전체 파이프라인을 통합한 스크립트

Usage:
    # 기본 실행
    ./run_code_gen_with_judge.py -i "Pick up the red cup" -o "red cup" "blue dish"

    # 시각화 및 Judge 포함
    ./run_code_gen_with_judge.py -i "Pick up the red cup" -o "red cup" --visualize-detection

    # 코드 생성만 (실행 없이)
    ./run_code_gen_with_judge.py -i "Pick up the red cup" -o "red cup" --dry-run

    # 이미지 저장
    ./run_code_gen_with_judge.py -i "Pick up the red cup" -o "red cup" --save-images /tmp/judge_images
"""

import argparse
import json
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "object_detection"))


class Lerobot_codegen_with_judge:
    """Detection → Code Gen → Execution → Judge 통합 파이프라인"""

    def __init__(
        self,
        robot_id: int = 3,
        llm_provider: str = "openai",
        judge_model: str = "gpt-4o",
        verbose: bool = True,
    ):
        """
        초기화

        Args:
            robot_id: 로봇 번호 (2 또는 3)
            llm_provider: LLM 제공자
            judge_model: Judge VLM 모델
            verbose: 상세 출력 여부
        """
        self.robot_id = robot_id
        self.llm_provider = llm_provider
        self.judge_model = judge_model
        self.verbose = verbose

        # 상태 저장
        self.camera = None
        self.initial_image: Optional[np.ndarray] = None
        self.final_image: Optional[np.ndarray] = None
        self.detection_image: Optional[np.ndarray] = None  # 검출 결과 이미지
        self.detected_positions: Dict = {}
        self.generated_code: str = ""
        self.instruction: str = ""

    def initialize_camera(self) -> bool:
        """카메라 초기화"""
        try:
            from object_detection.camera import RealSenseD435

            self.camera = RealSenseD435(width=640, height=480, fps=30)
            self.camera.start()
            if self.verbose:
                print("[Pipeline] Camera initialized")
            return True
        except Exception as e:
            print(f"[Pipeline] Camera initialization failed: {e}")
            return False

    def shutdown_camera(self) -> None:
        """카메라 종료"""
        if self.camera:
            self.camera.stop()
            if self.verbose:
                print("[Pipeline] Camera shutdown")

    def capture_frame(self) -> Optional[np.ndarray]:
        """현재 프레임 캡처"""
        if self.camera is None:
            return None
        try:
            color, _ = self.camera.get_frames()
            return color
        except Exception as e:
            print(f"[Pipeline] Frame capture failed: {e}")
            return None

    def run_detection(
        self,
        queries: list,
        timeout: float = 10.0,
        visualize: bool = False,
    ) -> Dict:
        """
        객체 검출 수행 (통합된 run_realtime_detection 사용)

        Args:
            queries: 검출할 객체 리스트
            timeout: 검출 타임아웃 (초)
            visualize: 시각화 여부

        Returns:
            {query: (x, y, z) or None} 딕셔너리
        """
        from run_detect import run_realtime_detection

        positions, last_frame, vis_image = run_realtime_detection(
            queries=queries,
            timeout=timeout,
            unit="m",
            return_last_frame=True,
            visualize=visualize,
        )

        if last_frame is not None:
            self.initial_image = last_frame
            if self.verbose:
                print("  Initial image captured from detection")

        if vis_image is not None:
            self.detection_image = vis_image
            if self.verbose:
                print("  Detection visualization image captured")

        return positions

    def generate_code(self, instruction: str, positions: Dict) -> str:
        """
        LLM을 통한 코드 생성

        Args:
            instruction: 자연어 명령어
            positions: 객체 위치 딕셔너리

        Returns:
            생성된 Python 코드
        """
        from code_gen_lerobot.code_gen_with_skill import lerobot_code_gen

        # 검출 결과 검증
        not_found = [name for name, pos in positions.items() if pos is None]
        if not_found:
            raise ValueError(f"Required objects not detected: {not_found}")

        # 코드 생성 (use_detection=False로 이미 검출된 위치 사용)
        code, _ = lerobot_code_gen(
            instruction=instruction,
            object_positions=positions,
            use_detection=False,
            llm_provider=self.llm_provider,
            robot_id=self.robot_id,
        )

        return code

    def execute_code(self, code: str, positions: Dict) -> bool:
        """
        생성된 코드 실행

        Args:
            code: 실행할 Python 코드
            positions: 객체 위치 딕셔너리 (exec 환경에 전달)

        Returns:
            실행 성공 여부
        """
        try:
            exec_globals = {
                "__name__": "__main__",
                "positions": positions,
            }
            exec(code, exec_globals)
            return True
        except Exception as e:
            print(f"[Pipeline] Code execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def capture_final_image(self) -> Optional[np.ndarray]:
        """최종 이미지 캡처"""
        if self.camera is None:
            if not self.initialize_camera():
                return None
            time.sleep(0.5)

        self.final_image = self.capture_frame()
        return self.final_image

    def run_judge(
        self,
        instruction: str,
        positions: Dict,
        executed_code: str,
    ) -> Dict:
        """
        Judge 실행

        Args:
            instruction: 목표 명령어
            positions: 객체 위치 딕셔너리
            executed_code: 실행된 Python 코드

        Returns:
            Judge 결과 딕셔너리
        """
        if self.initial_image is None or self.final_image is None:
            return {
                'prediction': 'UNCERTAIN',
                'reasoning': 'Initial or final image not captured',
                'success': False,
                'error': 'Missing images',
            }

        from judge import TaskJudge

        judge = TaskJudge(model=self.judge_model, verbose=self.verbose)
        return judge.judge(
            instruction=instruction,
            initial_image=self.initial_image,
            final_image=self.final_image,
            object_positions=positions,
            executed_code=executed_code,
        )

    def save_images(self, save_dir: str, prefix: str = "judge") -> Dict[str, str]:
        """
        이미지 저장

        Args:
            save_dir: 저장 디렉토리
            prefix: 파일명 접두사

        Returns:
            {'initial': path, 'final': path}
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        paths = {}

        if self.initial_image is not None:
            initial_path = Path(save_dir) / f"{prefix}_initial_{timestamp}.jpg"
            cv2.imwrite(str(initial_path), self.initial_image)
            paths['initial'] = str(initial_path)
            print(f"[Pipeline] Initial image saved: {initial_path}")

        if self.final_image is not None:
            final_path = Path(save_dir) / f"{prefix}_final_{timestamp}.jpg"
            cv2.imwrite(str(final_path), self.final_image)
            paths['final'] = str(final_path)
            print(f"[Pipeline] Final image saved: {final_path}")

        return paths

    def run(
        self,
        instruction: str,
        objects: list,
        detection_timeout: float = 10.0,
        visualize_detection: bool = False,
        execute: bool = True,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        전체 파이프라인 실행

        Args:
            instruction: 자연어 명령어
            objects: 검출할 객체 리스트
            detection_timeout: 검출 타임아웃
            visualize_detection: 검출 시각화 여부
            execute: 코드 실행 여부
            save_dir: 결과 저장 디렉토리 (None이면 저장 안함)

        Returns:
            {
                'positions': Dict,
                'code': str,
                'execution_success': bool,
                'judge_result': Dict,
                'saved_files': Dict,
            }
        """
        self.instruction = instruction
        result = {
            'positions': {},
            'code': '',
            'execution_success': False,
            'judge_result': {},
            'saved_files': {},
        }

        print("\n" + "=" * 60)
        print("LeRobot Code Generation with Judge Pipeline")
        print("=" * 60)

        try:
            # Step 1: 카메라 초기화 (시각화 모드가 아닐 때만)
            print("\n[Step 1/6] Initializing camera...")
            if visualize_detection:
                # 시각화 모드: run_realtime_detection이 자체 카메라 사용
                print("  (Deferred - visualization mode uses its own camera)")
            else:
                if not self.initialize_camera():
                    print("[Error] Camera initialization failed")
                    return result

            # Step 2: 객체 검출
            print(f"\n[Step 2/6] Detecting objects: {objects}")
            self.detected_positions = self.run_detection(
                queries=objects,
                timeout=detection_timeout,
                visualize=visualize_detection,
            )
            result['positions'] = self.detected_positions

            # 검출 결과 검증
            not_found = [k for k, v in self.detected_positions.items() if v is None]
            if not_found:
                print(f"[Error] Objects not detected: {not_found}")
                return result

            # 초기 이미지가 없으면 캡처
            if self.initial_image is None:
                print("  Capturing initial state image...")
                self.initial_image = self.capture_frame()

            # Step 3: 코드 생성
            print(f"\n[Step 3/6] Generating code via LLM ({self.llm_provider})...")
            self.generated_code = self.generate_code(instruction, self.detected_positions)
            result['code'] = self.generated_code

            print("\n" + "-" * 40)
            print("Generated Code:")
            print("-" * 40)
            print(self.generated_code)
            print("-" * 40)

            # Step 4: 코드 실행
            if execute:
                print(f"\n[Step 4/6] Executing generated code on Robot {self.robot_id}...")
                execution_success = self.execute_code(self.generated_code, self.detected_positions)
                result['execution_success'] = execution_success

                if not execution_success:
                    print("[Warning] Code execution failed")
            else:
                print("\n[Step 4/6] Skipped (dry-run mode)")
                result['execution_success'] = None

            # Step 5: 최종 이미지 캡처
            print("\n[Step 5/6] Capturing final state image...")
            time.sleep(1.0)  # 로봇 동작 완료 대기
            self.capture_final_image()

            # Step 6: Judge 실행
            print("\n[Step 6/6] Running Judge evaluation...")
            if execute and self.initial_image is not None and self.final_image is not None:
                judge_result = self.run_judge(
                    instruction=instruction,
                    positions=self.detected_positions,
                    executed_code=self.generated_code,
                )
                result['judge_result'] = judge_result
            else:
                print("  Skipped (no execution or missing images)")
                result['judge_result'] = {
                    'prediction': 'UNCERTAIN',
                    'reasoning': 'Judge skipped - no execution or missing images',
                }

            # 결과 출력
            self._print_summary(result)

            # Judge 결과 시각화 UI (항상 표시)
            if self.initial_image is not None and self.final_image is not None:
                from judge import show_judge_result, save_judge_log

                prediction = result['judge_result'].get('prediction', 'UNCERTAIN')
                reasoning = result['judge_result'].get('reasoning', '')

                result_image = show_judge_result(
                    initial_image=self.initial_image,
                    final_image=self.final_image,
                    instruction=instruction,
                    prediction=prediction,
                    reasoning=reasoning,
                    object_positions=self.detected_positions,
                    wait_key=True,
                )

                # 저장 (save_dir 지정 시)
                if save_dir:
                    result['saved_files'] = save_judge_log(
                        save_dir=save_dir,
                        initial_image=self.initial_image,
                        final_image=self.final_image,
                        instruction=instruction,
                        prediction=prediction,
                        reasoning=reasoning,
                        object_positions=self.detected_positions,
                        executed_code=self.generated_code,
                        result_image=result_image,
                        detection_image=self.detection_image,
                    )

            return result

        finally:
            self.shutdown_camera()

    def _print_summary(self, result: Dict) -> None:
        """결과 요약 출력"""
        print("\n" + "=" * 60)
        print("Pipeline Summary")
        print("=" * 60)

        print("\n[Detection Results]")
        for name, pos in result['positions'].items():
            if pos:
                print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
            else:
                print(f"  {name}: NOT FOUND")

        print(f"\n[Execution]")
        if result['execution_success'] is None:
            print("  Status: Skipped (dry-run)")
        elif result['execution_success']:
            print("  Status: SUCCESS")
        else:
            print("  Status: FAILED")

        print(f"\n[Judge Result]")
        judge = result.get('judge_result', {})
        print(f"  Prediction: {judge.get('prediction', 'N/A')}")
        reasoning = judge.get('reasoning', 'N/A')
        if len(reasoning) > 200:
            reasoning = reasoning[:200] + "..."
        print(f"  Reasoning: {reasoning}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot Code Generation with Judge Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 필수 인자
    parser.add_argument(
        "--instruction", "-i",
        type=str,
        required=True,
        help="Natural language instruction"
    )

    parser.add_argument(
        "--objects", "-o",
        type=str,
        nargs="+",
        required=True,
        help="Objects to detect"
    )

    # 선택 인자
    parser.add_argument(
        "--robot", "-r",
        type=int,
        default=3,
        help="Robot number (default: 3)"
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="openai",
        choices=["openai", "gemini", "llama"],
        help="LLM provider (default: openai)"
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-2.5-flash",
        help="Judge VLM model (default: gemini-2.5-flash)"
    )

    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=10.0,
        help="Detection timeout in seconds (default: 10)"
    )

    parser.add_argument(
        "--visualize-detection",
        action="store_true",
        help="Show real-time detection visualization"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate code only, don't execute"
    )

    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Directory to save all results (images, code, judge_log)"
    )

    args = parser.parse_args()

    # 파이프라인 실행
    pipeline = Lerobot_codegen_with_judge(
        robot_id=args.robot,
        llm_provider=args.llm,
        judge_model=args.judge_model,
        verbose=True,
    )

    result = pipeline.run(
        instruction=args.instruction,
        objects=args.objects,
        detection_timeout=args.timeout,
        visualize_detection=args.visualize_detection,
        execute=not args.dry_run,
        save_dir=args.save,
    )

    # 종료 코드 결정
    judge_prediction = result.get('judge_result', {}).get('prediction', 'UNCERTAIN')
    if judge_prediction == 'TRUE':
        sys.exit(0)
    elif judge_prediction == 'FALSE':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
