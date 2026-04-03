"""
Skill Recording Visualization

에피소드 레코딩 후 skill.* 데이터를 시각화하여 PNG로 저장합니다.
"""

import textwrap
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# skill.type별 고정 색상 (원색 계열, 선명한 구분)
_TYPE_COLORS = {
    "move": "#2196F3",          # 파랑
    "move_initial": "#9E9E9E",  # 회색
    "move_free": "#607D8B",     # 청회색
    "gripper_open": "#4CAF50",  # 초록
    "gripper_close": "#F44336", # 빨강
    "rotate": "#FF9800",        # 주황
}


def _draw_colored_bar(ax, df, column, transitions, colors_map, fontsize=7,
                      type_column=None, fig_width_inch=16):
    """색상 영역 바 그리기.

    Args:
        type_column: 지정 시, 각 세그먼트의 해당 컬럼 값으로 colors_map(dict)에서 색상 조회.
                     skill.natural_language 바를 skill.type 색상으로 칠할 때 사용.
        fig_width_inch: 피규어 폭 (인치). 텍스트 줄바꿈 계산에 사용.
    """
    total_width = len(df)
    # 텍스트 줄바꿈 계산을 위한 상수
    # plot area ≈ fig_width * 0.9, at 120 dpi
    plot_px = fig_width_inch * 0.9 * 120
    char_width_px = fontsize * 0.8  # approximate character width in pixels (conservative)

    for i in range(len(transitions) - 1):
        start, end = transitions[i], transitions[i + 1]
        label = str(df[column].iloc[start])
        if type_column is not None and isinstance(colors_map, dict):
            lookup_key = df[type_column].iloc[start]
            color = colors_map.get(lookup_key, "#CCCCCC")
        elif isinstance(colors_map, dict):
            color = colors_map.get(label, "#CCCCCC")
        else:
            color = colors_map[i % len(colors_map)]
        ax.axvspan(start, end, alpha=0.9, facecolor=color, edgecolor="white", linewidth=1.5)

        # 세그먼트 폭에 맞춰 텍스트 줄바꿈
        seg_ratio = (end - start) / total_width
        available_px = seg_ratio * plot_px
        max_chars = max(int(available_px / char_width_px), 1)
        wrapped = textwrap.fill(label, width=max_chars)

        # 세그먼트 영역 내 clipping (인접 세그먼트로 텍스트 넘침 방지)
        from matplotlib.patches import Rectangle
        clip_rect = Rectangle(
            (start, 0), end - start, 1,
            transform=ax.transData, visible=False,
        )
        ax.add_patch(clip_rect)

        txt = ax.text(
            (start + end) / 2, 0.5, wrapped,
            ha="center", va="center", fontsize=fontsize,
            color="black", fontweight="bold",
        )
        txt.set_clip_path(clip_rect)
    ax.set_yticks([])
    ax.set_xlim(0, len(df))


def generate_skill_visualizations(
    parquet_path: Optional[str] = None,
    save_dir: str = ".",
    episode_index: Optional[int] = None,
    dataframe: Optional[pd.DataFrame] = None,
) -> List[str]:
    """
    Skill recording 데이터 시각화 생성 및 저장.

    Args:
        parquet_path: parquet 파일 경로 (dataframe이 없을 때 사용)
        save_dir: 시각화 저장 디렉토리
        episode_index: 특정 에피소드만 필터 (None이면 전체)
        dataframe: 직접 전달할 DataFrame (parquet 대신 사용)

    Returns:
        저장된 파일 경로 리스트
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("[Visualization] matplotlib not installed, skipping")
        return []

    # Load data
    if dataframe is not None:
        df = dataframe.copy()
    elif parquet_path is not None:
        df = pd.read_parquet(parquet_path)
    else:
        print("[Visualization] No data source provided")
        return []
    if episode_index is not None and "episode_index" in df.columns:
        df = df[df["episode_index"] == episode_index].reset_index(drop=True)

    if len(df) == 0:
        print("[Visualization] No data for visualization")
        return []

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    frames = np.arange(len(df))

    # === Skill transition boundaries ===
    # natural_language가 없으면 skill.type을 대신 사용
    has_nl = "skill.natural_language" in df.columns
    nl_col = "skill.natural_language" if has_nl else "skill.type"

    nl_transitions = [0]
    for i in range(1, len(df)):
        if df[nl_col].iloc[i] != df[nl_col].iloc[i - 1]:
            nl_transitions.append(i)
    nl_transitions.append(len(df))

    # === Type transition boundaries ===
    type_transitions = [0]
    for i in range(1, len(df)):
        if df["skill.type"].iloc[i] != df["skill.type"].iloc[i - 1]:
            type_transitions.append(i)
    type_transitions.append(len(df))

    # Pre-compute common arrays
    obs_state = np.stack(df["observation.state"].values)
    goal_joint = np.stack(df["skill.goal_position.joint"].values)

    # =============================================
    # Plot 1: Timeline + Progress + Goal vs State
    # =============================================
    fig = plt.figure(figsize=(16, 15))
    gs = GridSpec(5, 1, figure=fig, height_ratios=[2.0, 1.2, 1, 1.5, 1.5], hspace=0.3)
    fig.suptitle(f"Skill Recording Analysis (Episode {episode_index}, {len(df)} frames)", fontsize=14)

    # 1a: Natural language timeline (fallback to skill.type if not available)
    ax = fig.add_subplot(gs[0])
    _draw_colored_bar(ax, df, nl_col, nl_transitions, _TYPE_COLORS,
                      fontsize=6, type_column="skill.type")
    ax.set_title(nl_col, fontsize=10)

    # 1b: Skill type (colored regions)
    ax = fig.add_subplot(gs[1], sharex=fig.axes[0])
    _draw_colored_bar(ax, df, "skill.type", type_transitions, _TYPE_COLORS, fontsize=7)
    ax.set_title("skill.type", fontsize=10)
    # Legend
    from matplotlib.patches import Patch
    type_legend = [Patch(facecolor=c, edgecolor="white", label=t) for t, c in _TYPE_COLORS.items()
                   if t in df["skill.type"].unique()]
    ax.legend(handles=type_legend, fontsize=7, loc="upper right", ncol=len(type_legend))

    # 1c: Progress
    ax = fig.add_subplot(gs[2], sharex=fig.axes[0])
    progress = df["skill.progress"].values
    if hasattr(progress[0], "__len__"):
        progress = np.array([p[0] if hasattr(p, "__len__") else p for p in progress])
    ax.plot(frames, progress, linewidth=1.2, color="tab:green")
    ax.set_ylabel("Progress")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("skill.progress (0→1 per skill, state-based)", fontsize=10)

    # 1d: Goal joint[0] vs observation.state[0]
    ax = fig.add_subplot(gs[3], sharex=fig.axes[0])
    ax.plot(frames, obs_state[:, 0], label="observation.state[0]", linewidth=1, alpha=0.8)
    ax.plot(frames, goal_joint[:, 0], label="goal_position.joint[0]", linewidth=1, linestyle="--", color="tab:red")
    ax.set_ylabel("Normalized")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("skill.goal_position.joint[0] vs observation.state[0] (shoulder_pan)", fontsize=10)

    # 1e: Gripper
    ax = fig.add_subplot(gs[4], sharex=fig.axes[0])
    goal_grip = df["skill.goal_position.gripper"].values
    if hasattr(goal_grip[0], "__len__"):
        goal_grip = np.array([g[0] if hasattr(g, "__len__") else g for g in goal_grip])
    ax.plot(frames, obs_state[:, 5], label="observation.state[5]", linewidth=1, alpha=0.8)
    ax.plot(frames, goal_grip, label="goal_position.gripper", linewidth=1, linestyle="--", color="tab:orange")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("skill.goal_position.gripper vs observation.state[5] (gripper)", fontsize=10)

    gs.tight_layout(fig, rect=[0, 0, 1, 0.96])
    p = save_dir / "skill_analysis_timeline.png"
    fig.savefig(str(p), dpi=120)
    plt.close(fig)
    saved.append(str(p))

    # =============================================
    # Plot 3: All 6 Goal Joints vs Observation State — (7,1) layout
    # =============================================
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(7, 1, figure=fig, height_ratios=[1] + [2] * 6, hspace=0.25)
    fig.suptitle("Goal Joint vs Observation State (All 6 Axes)", fontsize=14)

    # Row 0: skill.natural_language bar (fallback to skill.type)
    ax_skill = fig.add_subplot(gs[0])
    _draw_colored_bar(ax_skill, df, nl_col, nl_transitions, _TYPE_COLORS,
                      fontsize=6, type_column="skill.type")
    ax_skill.set_title(nl_col, fontsize=10)

    # Row 1-6: each joint
    for idx, name in enumerate(joint_names):
        ax = fig.add_subplot(gs[idx + 1], sharex=ax_skill)
        ax.plot(frames, obs_state[:, idx], label="obs.state", linewidth=1, alpha=0.8)
        ax.plot(frames, goal_joint[:, idx], label="goal.joint", linewidth=1, linestyle="--", color="tab:orange")
        for t in nl_transitions[1:-1]:
            ax.axvline(t, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.set_ylabel(name, fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
        if idx == len(joint_names) - 1:
            ax.set_xlabel("Frame")

    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    p = save_dir / "skill_analysis_joints.png"
    fig.savefig(str(p), dpi=120)
    plt.close(fig)
    saved.append(str(p))

    # =============================================
    # Plot 4: Summary Table
    # =============================================
    rows = []
    for i in range(len(nl_transitions) - 1):
        start, end = nl_transitions[i], nl_transitions[i + 1]
        sl = df.iloc[start]
        rows.append([
            sl[nl_col],
            sl["skill.type"],
            f"{start}-{end - 1}",
            str(end - start),
        ])

    fig, ax = plt.subplots(figsize=(14, max(3, 1 + len(rows) * 0.5)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Natural Language", "Type", "Frame Range", "Frames"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    fig.suptitle("Skill Execution Summary", fontsize=14)
    plt.tight_layout()
    p = save_dir / "skill_analysis_summary.png"
    fig.savefig(str(p), dpi=120)
    plt.close(fig)
    saved.append(str(p))

    return saved
