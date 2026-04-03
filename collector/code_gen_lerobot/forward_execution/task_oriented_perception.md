# Task-Oriented Perception Module

## Overview

Turn 0–2 prompts form a **task-oriented perception module** that uses a VLM (Vision-Language Model)
as a multi-turn perception pipeline. The VLM receives a natural language task instruction and
workspace images, then progressively narrows from holistic scene understanding down to
pixel-level manipulation points — all within a single chat session so that context accumulates
across turns.

This module is **task-agnostic**: the same prompts handle pick-and-place, assembly, folding,
or any other manipulation task. The task instruction (`{instruction}`) is the only variable
that changes between runs.

## Design Intent

### Why VLM as a perception module?

Traditional robot perception pipelines use fixed, task-specific detectors
(e.g., Grounding DINO for bounding boxes, heuristic grasp planners for grasp points).
These require per-task engineering and cannot reason about *why* certain grasp points
or interaction points matter for a given task.

By using a VLM in a multi-turn session, we get:

1. **Task-conditioned perception** — The VLM understands the task instruction and identifies
   only the objects and points relevant to *that specific task*, rather than detecting everything.
2. **Coarse-to-fine reasoning** — Each turn progressively refines spatial understanding:
   scene-level → object-level → point-level. Later turns benefit from earlier context.
3. **Unified interface** — One model handles scene understanding, detection, and pointing,
   eliminating the need to integrate multiple specialized models.
4. **Zero-shot generalization** — No task-specific training or fine-tuning required.
   New tasks only need a new instruction string.

### Why separate from code generation?

The system prompt and turn prompts are designed so that Turn 0–2 perform **perception only**.
Code generation instructions are deliberately excluded from both the system prompt and
Turn 0–2 prompts. This separation prevents the VLM from generating code or execution plans
during perception turns (a problem observed with the previous prompt design where the system
prompt contained "Steps Planning → Steps Execution" instructions that leaked into Turn 0).

Code generation happens in Turn 3 via `turn3_code_gen_prompt()`, which is a separate concern
that consumes the structured output of the perception module.

## Architecture

```
Single VLM Chat Session (system_prompt: robot/environment facts only)
│
├── Turn 0: Scene Understanding          — holistic, qualitative
│   ├── Input:  overhead image + (optional) CAD images + task instruction
│   ├── Output: natural language scene description
│   └── File:   turn0_prompt.py → turn0_scene_understanding_prompt()
│
├── Turn 1: Object Detection             — object-level, quantitative
│   ├── Input:  overhead image (re-sent) + "detect bboxes" prompt
│   ├── Output: JSON array of {box_2d, label} per object
│   └── File:   turn1_prompt.py → turn1_detect_task_relevant_objects_prompt()
│
├── Turn 2 (×N objects): Crop Pointing   — point-level, quantitative
│   ├── Input:  cropped image of one object + "identify critical points" prompt
│   ├── Output: JSON {critical_points: [{point_2d, label, role, reasoning}]}
│   └── File:   turn2_prompt.py → turn2_crop_pointing_prompt()
│
│   ... repeated for each detected object ...
│
└── [Turn 3: Code Generation — NOT part of perception module]
```

## Turn-by-Turn Detail

### System Prompt (`system_prompt.py`)

Sets the VLM's identity and provides robot/environment facts that are relevant
across all turns:

- Robot spec: LeRobot SO-101, single 5-DOF arm, asymmetric gripper
- Workspace: table dimensions, world coordinate frame, reachable range
- Camera: overhead, looking straight down

Crucially, the system prompt contains **no procedural instructions**
(no "Steps Planning", no "write code"). It tells the VLM:
*"Answer only what each turn asks for — do not anticipate later turns."*

### Turn 0 — Scene Understanding

**Goal**: Build a qualitative understanding of the workspace before any quantitative detection.

The VLM analyzes the overhead image (and optional CAD reference images) to describe:
1. Object identification — color, shape, approximate size
2. Spatial layout — where each object is on the table
3. Object relationships — relative positions between objects

This turn explicitly prohibits code, coordinates, or execution plans.
The output is purely natural language, serving as context for subsequent turns.

**Why this matters**: Turn 1 (detection) benefits from Turn 0's analysis because
the VLM already knows which objects are task-relevant and what they look like.
Without Turn 0, the VLM in Turn 1 might detect irrelevant objects or
misidentify objects that look similar.

### Turn 1 — Object Detection (BBox)

**Goal**: Localize each task-relevant object with a bounding box.

The VLM outputs a JSON array where each entry contains:
- `box_2d`: `[ymin, xmin, ymax, xmax]` normalized to 0–1000
- `label`: matching the object name from Turn 0 analysis

These bboxes are used to crop the original image for Turn 2.
Only main task-relevant objects are included (not sub-parts).

### Turn 2 — Crop-then-Point (per object)

**Goal**: Identify precise manipulation points on each object.

For each object detected in Turn 1:
1. The bbox is padded and used to crop the overhead image
2. The crop is sent to the VLM with a pointing prompt
3. The VLM identifies critical points with two possible roles:
   - **grasp** — where the gripper should grasp the object
   - **interaction** — non-grasping functional sub-parts (pin, hole, slot, etc.)

Point coordinates are normalized to 0–1000 within the crop, then converted
back to full-image pixel coordinates, and finally transformed to world coordinates
via camera calibration.

**Why crop?**: Sending a cropped close-up image significantly improves pointing
accuracy compared to asking the VLM to localize points on the full-resolution
overhead image where objects may be small.

## Data Flow

```
Turn 0 (scene understanding)
  → natural language context for Turn 1

Turn 1 (bbox detection)
  → valid_objects: [{"box_2d": [y1,x1,y2,x2], "label": "red block"}, ...]
  → used to crop images for Turn 2

Turn 2 (crop pointing, per object)
  → all_points: [{"px": 320, "py": 240, "role": "grasp", ...}, ...]
  → _points_to_positions() converts pixel → world coordinates
  → positions: {"red block": {"position": [x,y,z]}}

  ↓ (passed to Turn 3 code generation as structured data)
```

## File Reference

| File | Function | Turn |
|------|----------|------|
| `system_prompt.py` | (docstring content) | All turns |
| `turn0_prompt.py` | `turn0_scene_understanding_prompt()` | Turn 0 |
| `turn1_prompt.py` | `turn1_detect_task_relevant_objects_prompt()` | Turn 1 |
| `turn2_prompt.py` | `turn2_crop_pointing_prompt()` | Turn 2 |
| `code_gen_with_skill.py` | `lerobot_code_gen_multi_turn()` | Orchestrator |

## Changelog

- **2026-02-21**: Refactored prompt role separation.
  - System prompt: removed bi-arm description, "Steps Planning/Execution" procedure,
    and Grasp Guidelines. Retained only robot/environment facts.
  - Turn 0: replaced "how to grasp and manipulate" with structured scene analysis
    (object identification, spatial layout, object relationships) + explicit
    no-code constraint.
  - Turn 2: moved point type definitions (grasp, interaction) above output format
    so the VLM knows what to look for before analyzing the image.
  - Turn 1: renamed function to `turn1_detect_task_relevant_objects_prompt()`
    for naming consistency.
  - Grasp Guidelines moved to `turn3_code_gen_prompt()` in `user_prompt.py`.
