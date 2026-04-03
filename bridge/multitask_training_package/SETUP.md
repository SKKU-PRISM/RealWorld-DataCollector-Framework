# Multi-Task VLA Training Package

## Directory Structure
```
multitask_training_package/
  train_lora.py          # Training script
  eval_robocasa.py       # Evaluation script
  configs/               # All YAML configs
    multitask_smolvla.yaml
    multitask_pi05.yaml
    multitask_groot.yaml
    multitask_openvla.yaml
  data/                  # Metadata (copy alongside NPZ data)
    metadata.json
    metadata_extended.json
  patches/               # Patched library files (MUST apply)
  requirements.txt
```

## Setup on New Server

### 1. Python Environment
```bash
conda create -n robobridge python=3.12
conda activate robobridge
pip install -r requirements.txt
```

### 2. Apply Patches (CRITICAL)
These patches are required. `pip install` will overwrite them, so apply AFTER installing packages.
```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")

# Transformers patches (PI0.5 AdaRMSNorm + siglip check)
cp patches/transformers/models/gemma/modeling_gemma.py $SITE/transformers/models/gemma/
cp patches/transformers/models/siglip/check.py $SITE/transformers/models/siglip/

# LeRobot patches (GROOT SDPA + PI0.5 KV cache fix)
cp patches/lerobot/policies/groot/groot_n1.py $SITE/lerobot/policies/groot/
cp patches/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py $SITE/lerobot/policies/groot/eagle2_hg_model/
cp patches/lerobot/policies/pi05/modeling_pi05.py $SITE/lerobot/policies/pi05/
```

### 3. Data
Copy the processed NPZ data (39GB):
```bash
# On source server:
rsync -avP ./data/processed/ TARGET:/path/to/data/

# Then update configs if data path differs:
sed -i 's|./data/processed|/your/data/path|g' configs/multitask_*.yaml
```

### 4. Update Output Path
```bash
sed -i 's|./outputs/vla_adapters_multitask|/your/output/path|g' configs/multitask_*.yaml
```

## Training Commands

All models: eff_batch=32, 19 epochs, lr=1e-4, cosine schedule, LoRA rank=64

```bash
# SmolVLA (~2.7h)
CUDA_VISIBLE_DEVICES=0 python train_lora.py --config configs/multitask_smolvla.yaml

# PI0.5 (~11h)
CUDA_VISIBLE_DEVICES=0 python train_lora.py --config configs/multitask_pi05.yaml

# GROOT (~23h)
CUDA_VISIBLE_DEVICES=0 python train_lora.py --config configs/multitask_groot.yaml

# OpenVLA (~11h)
CUDA_VISIBLE_DEVICES=0 python train_lora.py --config configs/multitask_openvla.yaml
```

### 2 GPU Parallel (saves ~50% time)
```bash
# GPU 0: SmolVLA then PI0.5
CUDA_VISIBLE_DEVICES=0 python train_lora.py --config configs/multitask_smolvla.yaml && \
CUDA_VISIBLE_DEVICES=0 python train_lora.py --config configs/multitask_pi05.yaml

# GPU 1: GROOT then OpenVLA
CUDA_VISIBLE_DEVICES=1 python train_lora.py --config configs/multitask_groot.yaml && \
CUDA_VISIBLE_DEVICES=1 python train_lora.py --config configs/multitask_openvla.yaml
```

## Training Config Summary
| Model | Params | Batch | GradAccum | Steps | Time |
|---|---|---|---|---|---|
| SmolVLA | ~500M | 8 | 4 | 19,760 | ~2.7h |
| PI0.5 | ~3B | 4 | 8 | 19,760 | ~11h |
| GROOT | ~3B | 2 | 16 | 24,795 | ~23h |
| OpenVLA | 7B | 2 | 16 | 19,760 | ~11h |

## Output
Adapters saved to: `{output_base_dir}/general/lora_adapter/`
