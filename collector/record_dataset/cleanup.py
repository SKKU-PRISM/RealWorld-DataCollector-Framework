"""
Dataset Cleanup for Resume Mode

resume 시 재취득 대상 에피소드를 데이터셋에서 제거하여
파이프라인 결과와 데이터셋의 1:1 정합성을 유지합니다.

데이터셋에는 judge=="TRUE"인 에피소드만 저장되므로,
성공 에피소드의 파이프라인 폴더가 삭제되어 재취득 대상이 되면
해당 데이터셋 에피소드도 함께 제거해야 합니다.
"""

import sys
import json
import shutil
from pathlib import Path
from typing import Optional

LEROBOT_PATH = Path(__file__).parent.parent / "lerobot" / "src"
if str(LEROBOT_PATH) not in sys.path:
    sys.path.insert(0, str(LEROBOT_PATH))


def cleanup_dataset_for_resume(
    session_dir: str,
    repo_id: str,
    root: Optional[str] = None,
) -> dict:
    """resume 전 데이터셋에서 재취득 대상 에피소드를 제거.

    매핑 로직:
    - 데이터셋에는 judge=="TRUE"인 에피소드만 순서대로 저장됨
    - 폴더가 삭제된 에피소드는 원래 TRUE/FALSE를 알 수 없으므로,
      데이터셋의 실제 에피소드 수를 기준으로 역추적

    순회 방식:
    1. ep1부터 순서대로 스캔하며 각 에피소드의 상태 파악
       - TRUE (유지) / FALSE (재취득, dataset 미기록) / 삭제됨 (재취득, 불명)
    2. dataset에 기록된 에피소드 수만큼 "TRUE 또는 삭제됨"에 dataset idx를 순서대로 할당
    3. 삭제된 에피소드에 할당된 dataset idx = 삭제 대상
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

    session_path = Path(session_dir)
    dataset_path = Path(root) if root else HF_LEROBOT_HOME / repo_id

    stats = {
        "pipeline_episodes_total": 0,
        "pipeline_episodes_success": 0,
        "pipeline_episodes_to_rerun": 0,
        "dataset_episodes_before": 0,
        "dataset_episodes_after": 0,
        "deleted_indices": [],
    }

    # 1. 데이터셋 존재 확인 및 열기
    if not dataset_path.exists():
        print("[Cleanup] No dataset found, skipping")
        return stats

    # 불완전 데이터셋 감지: meta/episodes/ parquet이 없으면 finalize() 미호출 상태
    # 전체 삭제 대신 finalize를 시도하여 기존 데이터 보존
    episodes_meta_dir = dataset_path / "meta" / "episodes"
    if dataset_path.exists() and not episodes_meta_dir.exists():
        print(f"[Cleanup] Incomplete dataset detected (no meta/episodes/) — attempting recovery...")
        try:
            dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
            dataset.meta._close_writer()  # flush metadata buffer → create meta/episodes/
            dataset._close_writer()
            print(f"[Cleanup] Recovery successful: {dataset.meta.total_episodes} episodes recovered")
        except Exception as e:
            print(f"[Cleanup] Recovery failed ({e}) — removing dataset")
            shutil.rmtree(dataset_path)
            return stats

    try:
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
        actual_dataset_episodes = dataset.meta.total_episodes
        stats["dataset_episodes_before"] = actual_dataset_episodes
    except Exception as e:
        print(f"[Cleanup] Warning: Cannot open dataset, removing: {e}")
        shutil.rmtree(dataset_path)
        return stats

    if actual_dataset_episodes == 0:
        print("[Cleanup] Dataset is empty, skipping")
        return stats

    # 2. 파이프라인 에피소드 스캔
    episode_dirs = sorted(session_path.glob("episode_*"))
    if not episode_dirs:
        print("[Cleanup] No episodes found in session, skipping")
        return stats

    max_episode = 0
    for ep_dir in episode_dirs:
        try:
            ep_num = int(ep_dir.name.split("_")[1])
            max_episode = max(max_episode, ep_num)
        except (ValueError, IndexError):
            continue

    # 각 에피소드 상태 파악:
    #   "kept_true"  = TRUE + 폴더 존재 → dataset에 기록됨, 유지
    #   "false"      = FALSE + 폴더 존재 → dataset에 미기록, 재취득
    #   "missing"    = 폴더 삭제됨 → dataset에 기록되었을 수도 있음, 재취득
    ep_states = []  # [(ep_num, state)]
    for ep_num in range(1, max_episode + 1):
        batch_info_path = session_path / f"episode_{ep_num:02d}" / "batch_info.json"

        if not batch_info_path.exists():
            ep_states.append((ep_num, "missing"))
            stats["pipeline_episodes_to_rerun"] += 1
        else:
            with open(batch_info_path) as f:
                bi = json.load(f)
            judge = bi.get("judge", "")
            if judge == "TRUE":
                ep_states.append((ep_num, "kept_true"))
                stats["pipeline_episodes_success"] += 1
            else:
                ep_states.append((ep_num, "false"))
                stats["pipeline_episodes_to_rerun"] += 1

    stats["pipeline_episodes_total"] = len(ep_states)

    # 3. dataset index 매핑
    #    dataset에는 TRUE 에피소드만 순서대로 저장됨.
    #    "kept_true"와 "missing" 중 원래 TRUE였던 것들이 dataset idx를 차지함.
    #    "false"는 discard되어 dataset에 없음.
    #
    #    방법: "false"가 아닌 에피소드(=TRUE였거나 삭제됨)에 순서대로
    #    dataset idx를 할당. 총 할당 수가 actual_dataset_episodes와 일치해야 함.
    could_be_in_dataset = [(ep_num, state) for ep_num, state in ep_states
                           if state != "false"]

    # 할당 가능한 수가 실제 dataset 에피소드 수보다 적으면 → 데이터 불일치
    if len(could_be_in_dataset) < actual_dataset_episodes:
        print(f"[Cleanup] Warning: dataset has {actual_dataset_episodes} episodes but "
              f"only {len(could_be_in_dataset)} non-FALSE pipeline episodes found")
        # 초과분은 뒤에서 자르기 (이전 불완전 resume 잔여물)
        excess_start = len(could_be_in_dataset)
        excess_indices = list(range(excess_start, actual_dataset_episodes))
        if excess_indices:
            print(f"[Cleanup] Trimming {len(excess_indices)} excess episodes from dataset tail")

    # 앞에서부터 actual_dataset_episodes개만 할당
    dataset_indices_to_delete = []
    assigned = 0
    for ep_num, state in could_be_in_dataset:
        if assigned >= actual_dataset_episodes:
            break
        if state == "missing":
            # 삭제된 에피소드 → 원래 TRUE였고 dataset에 기록됨 → 삭제 대상
            dataset_indices_to_delete.append(assigned)
            print(f"  ep{ep_num:02d} (deleted folder) → dataset idx {assigned} → DELETE")
        else:
            # kept_true → 유지
            print(f"  ep{ep_num:02d} (TRUE, kept)     → dataset idx {assigned} → keep")
        assigned += 1

    # 초과분 처리 (dataset에 pipeline보다 많은 에피소드가 있는 경우)
    if actual_dataset_episodes > assigned:
        excess_indices = list(range(assigned, actual_dataset_episodes))
        dataset_indices_to_delete.extend(excess_indices)
        print(f"[Cleanup] {len(excess_indices)} excess episodes at tail → DELETE (indices {excess_indices})")

    stats["deleted_indices"] = sorted(dataset_indices_to_delete)

    if not dataset_indices_to_delete:
        print(f"[Cleanup] Dataset is clean ({actual_dataset_episodes} episodes), no cleanup needed")
        stats["dataset_episodes_after"] = actual_dataset_episodes
        return stats

    print(f"[Cleanup] Deleting {len(dataset_indices_to_delete)} episodes from dataset: {dataset_indices_to_delete}")

    # 4. 전부 삭제해야 하는 경우 → 데이터셋 디렉토리 자체 제거
    if len(dataset_indices_to_delete) >= actual_dataset_episodes:
        print(f"[Cleanup] All episodes to be deleted, removing dataset entirely")
        del dataset
        shutil.rmtree(dataset_path)
        stats["dataset_episodes_after"] = 0
        return stats

    # 5. delete_episodes()로 정리된 데이터셋 생성
    from lerobot.datasets.dataset_tools import delete_episodes

    temp_dir = dataset_path.parent / f"{dataset_path.name}_cleanup_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    try:
        new_dataset = delete_episodes(
            dataset=dataset,
            episode_indices=dataset_indices_to_delete,
            output_dir=temp_dir,
            repo_id=repo_id,
        )
        stats["dataset_episodes_after"] = new_dataset.meta.total_episodes
        print(f"[Cleanup] New dataset: {new_dataset.meta.total_episodes} episodes")

        del dataset
        del new_dataset

        # 6. 안전하게 교체: old → backup, temp → original, backup 삭제
        backup_dir = dataset_path.parent / f"{dataset_path.name}_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        dataset_path.rename(backup_dir)
        temp_dir.rename(dataset_path)
        shutil.rmtree(backup_dir)

        print(f"[Cleanup] Dataset replaced successfully")

    except Exception as e:
        print(f"[Cleanup] Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    return stats
