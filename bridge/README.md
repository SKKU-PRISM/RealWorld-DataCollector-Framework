# RoboBridge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular robot manipulation framework with LangChain integration for natural language task planning and execution.

## Features

- **Natural Language Control** - Execute robot tasks using plain English
- **Modular Architecture** - Swap perception, planning, and control modules independently
- **Multi-Provider Support** - OpenAI, Anthropic, Google, Ollama, and custom models
- **Hardware Agnostic** - Works with Franka, UR robots, or simulation
- **Distributed Execution** - Run modules across multiple machines

## Installation

```bash
pip install robobridge

# With specific providers
pip install "robobridge[openai]"
pip install "robobridge[anthropic]"
pip install "robobridge[full]"  # All providers
```

## Quick Start

```python
from robobridge import RoboBridge

# Initialize (uses simulation by default)
robot = RoboBridge.initialize()

# Execute natural language command
robot.execute("Pick up the red cup and place it on the table")

# Or use direct control
robot.pick("red_cup")
robot.place("red_cup", position=[0.4, 0.2, 0.05])

robot.shutdown()
```

### Two-Stage Planning

The Planner uses a **two-stage LLM architecture**:

1. **ActionPlanner (LLM)**: Converts natural language → high-level actions (pick, place, push, pull, open, close)
2. **PrimitivePlanner (LLM)**: Converts each action → primitive sequences (go, move, grip) with coordinates

## Modules

| Module | Purpose | Providers |
|--------|---------|-----------|
| **Perception** | Object detection | Florence-2, YOLO, GPT-4V, Custom |
| **Planner** | Two-stage planning (LLM) | OpenAI, Anthropic, Google, Ollama, Custom |
| **Controller** | Trajectory generation | Primitives, VLA, MoveIt, Custom |
| **Robot** | Hardware interface | Simulation, Franka, UR, Custom |
| **Monitor** | Execution feedback (VLM) | OpenAI, Anthropic, Google, Custom |

## Configuration

```python
# config.py
MODULE_CONFIGS = {
    "planner": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.7,
        "has_mobility": False,  # True for mobile robots
        "action_list": ["pick", "place", "push", "pull", "open", "close"],
    },
    "perception": {
        "provider": "hf",
        "model": "florence-2",
        "device": "cuda:0",
    },
}
```

```python
from robobridge import RoboBridge

robot = RoboBridge.initialize(config_path="./config.py")
```

## Custom Models

Integrate your own models using wrapper classes:

```python
from robobridge import CustomPerception, Detection

class MyDetector(CustomPerception):
    def load_model(self):
        self._model = load_my_model()
    
    def detect(self, rgb, depth=None, object_list=None):
        results = self._model(rgb)
        return [Detection(name=r.name, confidence=r.conf) for r in results]
```

## Training VLA Adapters

RoboBridge supports LoRA fine-tuning of Vision-Language-Action (VLA) models on your own demonstration data.

### Supported VLA Backends

| Backend | Model | Notes |
|---------|-------|-------|
| `groot_n1.5` | `nvidia/GR00T-N1.5-3B` | Default, best overall performance |
| `openvla` | `openvla/openvla-7b` | Open-source VLA |
| `smolvla` | `HuggingFaceTB/SmolVLA-Base` | Lightweight |
| `pi05` | `physical-intelligence/pi0.5-3b` | PI0.5 |
| `lerobot` | LeRobot models | HuggingFace LeRobot ecosystem |

### Training Config (YAML)

```yaml
# configs/groot_libero_move.yaml
adapter:
  name: "move"
  loss_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1]

model:
  backend: "groot_n1.5"
  name: "nvidia/GR00T-N1.5-3B"
  action_dim: 12
  image_size: 224
  chunk_size: 16

lora:
  rank: 64
  alpha: 128
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 5.0e-5
  num_epochs: 50
  scheduler_type: "cosine"
  bf16: true

output:
  base_dir: "outputs/groot_libero"
```

### Training Command

```bash
# Train move adapter (trajectory control)
python scripts/train/train_lora_movegrip.py \
    --config configs/groot_libero_move.yaml \
    --task <task_name> \
    --data-dir data_libero/<task_name>

# Train grip adapter (gripper open/close)
python scripts/train/train_lora_movegrip.py \
    --config configs/groot_libero_move.yaml \
    --adapter-type grip \
    --task <task_name> \
    --data-dir data_libero/<task_name>
```

Preprocessed data directories should contain `images/`, `states.npy`, and `actions.npy`. See `scripts/preprocess/preprocess_libero.py` or `scripts/preprocess/preprocess_delta_base.py` for data preparation.

### SO101 Real Robot Training

```bash
# Preprocess SO101 demonstration data
python scripts/preprocess/preprocess_so101.py \
    --data-dir data_so101/<task_name>

# Joint-space preprocessing
python scripts/preprocess/preprocess_so101_joint.py \
    --data-dir data_so101/<task_name>

# Train move adapter for SO101
python scripts/train/train_lora_movegrip.py \
    --config configs/groot_so101.yaml \
    --task <task_name> \
    --data-dir data_so101/<task_name>
```

## Evaluation

### Direct Mode (Single VLA Model)

Evaluates the VLA model directly without the full pipeline:

```bash
python scripts/eval/eval_vla_robocasa.py --mode direct \
    --model groot \
    --move-adapter outputs/groot_libero/<task>/move_adapter/checkpoint-best \
    --grip-adapter outputs/groot_libero/<task>/grip_adapter/checkpoint-best \
    --tasks <task_name> --num-episodes 25
```

### Pipeline Mode (Perception → Planner → Controller)

Runs the full modular pipeline where each module can use a different model:

```bash
python scripts/eval/eval_vla_robocasa.py --mode pipeline \
    --vla-backend groot_n1.5 \
    --vla-model nvidia/GR00T-N1.5-3B \
    --move-adapter outputs/<task>/move_adapter/checkpoint-best \
    --grip-adapter outputs/<task>/grip_adapter/checkpoint-best \
    --action-stats outputs/<task>/move_adapter/data_stats.json \
    --planner-provider anthropic --planner-model claude-sonnet-4-20250514 \
    --monitor-provider openai --monitor-model gpt-4o \
    --perception-mode robocasa_gt \
    --tasks <task_name> --num-episodes 25
```

### Swapping Module Models

Each module's model can be independently changed via CLI flags:

```bash
# Planner: swap between providers
--planner-provider openai    --planner-model gpt-4o
--planner-provider anthropic --planner-model claude-sonnet-4-20250514
--planner-provider google    --planner-model gemini-2.0-flash
--planner-provider bedrock   --planner-model eu.anthropic.claude-opus-4-6-v1

# Monitor: swap or disable
--monitor-provider openai    --monitor-model gpt-4o
--monitor-provider anthropic --monitor-model claude-sonnet-4-20250514
--monitor-provider ""  # disable monitor

# Perception: swap detection backend
--perception-mode robocasa_gt       # ground truth (simulation)
--perception-mode florence2         # Florence-2
--perception-mode grounding_dino    # Grounding DINO

# VLA Controller: swap backbone
--vla-backend groot_n1.5  --vla-model nvidia/GR00T-N1.5-3B
--vla-backend openvla      --vla-model openvla/openvla-7b
--vla-backend smolvla      --vla-model HuggingFaceTB/SmolVLA-Base
--vla-backend pi05         --vla-model physical-intelligence/pi0.5-3b
```

### Programmatic Model Swapping

You can also swap models at runtime using the Python API:

```python
from robobridge import RoboBridge

robot = RoboBridge.initialize()

# Swap planner to Anthropic Claude
robot.set_model("planner", provider="anthropic", model="claude-sonnet-4-20250514")

# Swap monitor to Google Gemini
robot.set_model("monitor", provider="google", model="gemini-2.0-flash")

# Swap perception to a custom detector
robot.set_model("perception", provider="custom", model="/path/to/detector.py:MyDetector")

robot.execute("Pick up the red cup and place it on the table")
```

### LIBERO Evaluation

```bash
python scripts/eval/eval_vla_libero.py --mode template \
    --vla-backend groot_n1.5 --vla-model nvidia/GR00T-N1.5-3B \
    --adapter-dir outputs/groot_libero \
    --tasks <task_name> --num-episodes 50

# Pipeline mode with VLM planner
python scripts/eval/eval_vla_libero.py --mode pipeline \
    --vla-backend groot_n1.5 --vla-model nvidia/GR00T-N1.5-3B \
    --adapter-dir outputs/groot_libero \
    --planner-provider anthropic --planner-model claude-sonnet-4-20250514 \
    --tasks <task_name> --num-episodes 50
```

### SO101 Real Robot Evaluation

The SO101 evaluation runs a full modular pipeline on a real SO-101 robot arm:

```bash
# Full pipeline with Gemini planner + monitor
python scripts/eval/eval_so101.py \
    --port /dev/ttyACM0 \
    --planner-provider google --planner-model gemini-2.5-flash \
    --monitor-provider google --monitor-model gemini-2.5-flash \
    --tasks pick_redblock_place_bluedish --num-episodes 5

# Swap planner to Anthropic Claude
python scripts/eval/eval_so101.py \
    --port /dev/ttyACM0 \
    --planner-provider anthropic --planner-model claude-sonnet-4-20250514 \
    --tasks pick_redblock_place_bluedish

# No monitor (faster)
python scripts/eval/eval_so101.py \
    --port /dev/ttyACM0 \
    --planner-provider google --planner-model gemini-2.5-flash \
    --monitor-provider "" \
    --tasks pick_redblock_place_bluedish

# Dry-run (no robot, inference only)
python scripts/eval/eval_so101.py --dry-run \
    --tasks pick_redblock_place_bluedish
```

## CLI

```bash
# Run server
robobridge server --config ./config.py

# Run individual module
robobridge module --module planner

# Demo
robobridge demo
```

## Documentation

Full documentation available at: [https://prism-skku.github.io/robobridge](https://prism-skku.github.io/robobridge)

- [Installation](docs/get-started/installation.md)
- [Quickstart](docs/get-started/quickstart.md)
- [Architecture](docs/concepts/architecture.md)
- [Custom Models](docs/how-to/custom-models.md)
- [Distributed Deployment](docs/how-to/distributed.md)

## Directory Structure

```
├── src/robobridge/               # Framework core
│   ├── core/                     # RoboBridge orchestrator
│   ├── modules/                  # Perception, Planner, Controller, Robot, Monitor
│   ├── wrappers/                 # Custom model base classes
│   ├── client/                   # High-level RoboBridge API
│   ├── cli/                      # Command line interface
│   ├── config/                   # Configuration utilities
│   └── utils/                    # Helper functions
├── scripts/
│   ├── eval/                     # Evaluation scripts (RoboCasa, LIBERO, SO101)
│   ├── preprocess/               # Data preprocessing
│   ├── train/                    # LoRA training
│   ├── so101/                    # SO-101 robot client/server
│   └── serve/                    # VLA model server
├── configs/                      # Training & evaluation configs
├── docs/                         # Documentation
├── examples/                     # Example notebooks
├── tests/                        # Test suite
├── pyproject.toml
└── README.md
```

## License

MIT License
