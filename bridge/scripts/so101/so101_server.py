#!/usr/bin/env python3
"""
SO101 GROOT Inference Server.

GPU 서버에서 GROOT N1.5 + LoRA 어댑터를 로딩하고
HTTP로 추론 결과(denormalized action chunk)를 제공합니다.

학습 데이터 포맷:
    State 12D: eef_pos(3) + eef_quat(4) + gripper_2d(2) + direction(3)
    Action 7D: delta_pos(3) + delta_rot_rpy(3) + gripper(1)
    Normalization: min_max (data_stats.json)

State format (matching data_so101):
    11D: ee_pos(3) + ee_quat_xyzw(4) + gripper(1) + goal_direction(3)
    Action 7D: delta_pos(3) + delta_rot_rpy(3) + gripper_binary(1)

Protocol:
    POST /predict
        {
            "image_b64": "<base64 JPEG 224x224>",
            "state": [11 floats],       # ee_pos(3)+quat_xyzw(4)+gripper(1)+direction(3)
            "instruction": "move"
        }
    Response:
        {
            "action_chunk": [[7 floats], ...],  # (16, 7) denormalized
            "inference_ms": 42.3
        }

    POST /reset     — 에피소드 시작 시 호출
    GET  /health
    GET  /info

Usage:
    CUDA_VISIBLE_DEVICES=0 python so101_server.py \
        --adapter outputs/groot_hf/pick_redblock_place_bluedish_30epi/move_adapter/checkpoint-best \
        --stats   outputs/groot_hf/pick_redblock_place_bluedish_30epi/move_adapter/data_stats.json \
        --port 8002
"""

import argparse
import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("so101_server")

for _mod in ("PIL", "urllib3", "transformers", "huggingface_hub"):
    logging.getLogger(_mod).setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*Fetching.*files.*")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_state(state: np.ndarray, smin: np.ndarray, smax: np.ndarray) -> np.ndarray:
    r = smax - smin + 1e-8
    return np.clip(2.0 * (state - smin) / r - 1.0, -1.0, 1.0)


def denormalize_action(action: np.ndarray, amin: np.ndarray, amax: np.ndarray) -> np.ndarray:
    return (action + 1.0) / 2.0 * (amax - amin) + amin


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def load_stats(path: str) -> Dict[str, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    stats = {
        "action_min": np.array(data["action_stats"]["min"], dtype=np.float32),
        "action_max": np.array(data["action_stats"]["max"], dtype=np.float32),
        "state_min": np.array(data["state_stats"]["min"], dtype=np.float32),
        "state_max": np.array(data["state_stats"]["max"], dtype=np.float32),
    }
    logger.info(f"Stats: action_dim={len(stats['action_min'])}, state_dim={len(stats['state_min'])}")
    return stats


# ---------------------------------------------------------------------------
# GROOT model loading
# ---------------------------------------------------------------------------

def load_groot(
    adapter_path: str,
    base_model: str,
    device,
    lora_rank: int = 64,
    lora_alpha: int = 128,
):
    """GROOT N1.5 + LoRA adapter 로딩."""
    import torch
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.groot.processor_groot import _build_eagle_processor

    ckpt = Path(adapter_path)

    # adapter config 에서 rank/alpha 읽기
    config_path = ckpt / "config.json"
    if config_path.is_file():
        with open(config_path) as f:
            cfg = json.load(f)
        lora_rank = cfg.get("lora_rank", lora_rank)
        lora_alpha = cfg.get("lora_alpha", lora_rank * 2)

    logger.info(f"Loading GROOT base: {base_model} (rank={lora_rank}, alpha={lora_alpha})")
    policy = GrootPolicy.from_pretrained(
        base_model, lora_rank=lora_rank, lora_alpha=lora_alpha,
    )

    # Load adapter weights
    weights_file = ckpt / "adapter_model.pt"
    if not weights_file.is_file():
        weights_file = ckpt / "model.safetensors"

    if weights_file.suffix == ".pt":
        adapter_weights = torch.load(weights_file, map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        adapter_weights = load_file(str(weights_file))

    missing, unexpected = policy.load_state_dict(adapter_weights, strict=False)
    logger.info(f"Adapter loaded: {len(adapter_weights)} tensors, "
                f"{len(missing)} missing, {len(unexpected)} unexpected")

    policy = policy.to(dtype=torch.bfloat16, device=device)
    policy.eval()

    # Eagle processor for image encoding
    eagle = _build_eagle_processor()
    ip = getattr(eagle, "image_processor", None)
    if ip is not None and not hasattr(ip, "_prepare_image_like_inputs"):
        if hasattr(ip, "_prepare_input_images"):
            ip._prepare_image_like_inputs = ip._prepare_input_images
        else:
            from transformers.image_processing_utils_fast import BaseImageProcessorFast
            import types
            ip._prepare_image_like_inputs = types.MethodType(
                BaseImageProcessorFast._prepare_image_like_inputs, ip,
            )

    total = sum(p.numel() for p in policy.parameters())
    logger.info(f"Model: {total / 1e6:.1f}M params on {device}")

    return policy, eagle


# ---------------------------------------------------------------------------
# Batch building & inference
# ---------------------------------------------------------------------------

def build_batch(
    image: np.ndarray,
    state: np.ndarray,
    instruction: str,
    stats: Dict[str, np.ndarray],
    eagle,
    device,
    image_size: int = 224,
    image2: np.ndarray = None,
) -> Dict:
    """이미지(들) + 11D state → GROOT 배치.

    State 11D: ee_pos(3) + ee_quat_xyzw(4) + gripper(1) + goal_direction(3)
    image: 탑뷰 (필수)
    image2: 그리퍼캠 (옵션, 있으면 멀티뷰)
    """
    import torch
    from PIL import Image as PILImage

    # State normalize → pad to 64D
    state_norm = normalize_state(state, stats["state_min"], stats["state_max"])
    state_pad = np.zeros(64, dtype=np.float32)
    state_pad[:len(state_norm)] = state_norm
    state_mask = np.zeros(64, dtype=bool)
    state_mask[:len(state_norm)] = True

    batch = {
        "state": torch.from_numpy(state_pad).float().unsqueeze(0).unsqueeze(0).to(device),
        "state_mask": torch.from_numpy(state_mask).unsqueeze(0).unsqueeze(0).to(device),
        "embodiment_id": torch.tensor([31], dtype=torch.long).to(device),
    }

    # Eagle image encoding
    pil_img = PILImage.fromarray(image)
    if pil_img.size != (image_size, image_size):
        pil_img = pil_img.resize((image_size, image_size), PILImage.BILINEAR)

    if image2 is not None:
        # 멀티뷰: 탑뷰 + 그리퍼캠 (GROOT은 알파벳순 정렬)
        pil_img2 = PILImage.fromarray(image2)
        if pil_img2.size != (image_size, image_size):
            pil_img2 = pil_img2.resize((image_size, image_size), PILImage.BILINEAR)
        eagle_inputs = eagle(
            text=[f"<image-1> <image-2> {instruction}"],
            images=[pil_img, pil_img2],
            images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
            return_tensors="pt",
            padding=True,
        )
    else:
        eagle_inputs = eagle(
            text=[f"<image-1> {instruction}"],
            images=[pil_img],
            images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
            return_tensors="pt",
            padding=True,
        )

    for k, v in eagle_inputs.items():
        batch[f"eagle_{k}"] = v.to(device)

    return batch


def build_batch_joint(
    image: np.ndarray,
    state_9d: np.ndarray,
    instruction: str,
    stats: Dict[str, np.ndarray],
    eagle,
    device,
    image_size: int = 224,
    image2: np.ndarray = None,
) -> Dict:
    """Joint 모델용: 이미지(들) + 9D state → GROOT 배치.

    State 9D: joint_angles(6, normalized) + ee_xyz(3)
    """
    import torch
    from PIL import Image as PILImage

    state_norm = normalize_state(state_9d, stats["state_min"], stats["state_max"])
    state_pad = np.zeros(64, dtype=np.float32)
    state_pad[:len(state_norm)] = state_norm
    state_mask = np.zeros(64, dtype=bool)
    state_mask[:len(state_norm)] = True

    batch = {
        "state": torch.from_numpy(state_pad).float().unsqueeze(0).unsqueeze(0).to(device),
        "state_mask": torch.from_numpy(state_mask).unsqueeze(0).unsqueeze(0).to(device),
        "embodiment_id": torch.tensor([31], dtype=torch.long).to(device),
    }

    pil_img = PILImage.fromarray(image)
    if pil_img.size != (image_size, image_size):
        pil_img = pil_img.resize((image_size, image_size), PILImage.BILINEAR)

    if image2 is not None:
        pil_img2 = PILImage.fromarray(image2)
        if pil_img2.size != (image_size, image_size):
            pil_img2 = pil_img2.resize((image_size, image_size), PILImage.BILINEAR)
        eagle_inputs = eagle(
            text=[f"<image-1> <image-2> {instruction}"],
            images=[pil_img, pil_img2],
            images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
            return_tensors="pt",
            padding=True,
        )
    else:
        eagle_inputs = eagle(
            text=[f"<image-1> {instruction}"],
            images=[pil_img],
            images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
            return_tensors="pt",
            padding=True,
        )

    for k, v in eagle_inputs.items():
        batch[f"eagle_{k}"] = v.to(device)

    return batch


def predict_and_denorm(
    policy, batch, stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """추론 → denormalize → (16, 7) numpy."""
    import torch

    with torch.no_grad():
        actions = policy.predict_action_chunk(batch)
    # (1, 16, 32) → (16, 32)
    chunk = actions[0].float().cpu().numpy()

    amin = stats["action_min"]
    amax = stats["action_max"]
    adim = len(amin)

    result = np.zeros((chunk.shape[0], 7), dtype=np.float32)
    for t in range(chunk.shape[0]):
        a = chunk[t, :adim].copy()
        a_denorm = denormalize_action(a, amin[:adim], amax[:adim])
        result[t, :min(7, len(a_denorm))] = a_denorm[:7]

    return result


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def create_app(policy, eagle, stats, device, denoise_steps, image_size, adapter_path):
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    _s = {"count": 0, "t0": time.time()}

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "requests": _s["count"],
                        "uptime_s": round(time.time() - _s["t0"], 1)})

    @app.route("/info", methods=["GET"])
    def info():
        return jsonify({
            "model": "groot_n1.5", "adapter": adapter_path,
            "state_dim": 11, "action_dim": 7, "chunk_size": 16,
            "state_format": "ee_pos(3)+ee_quat_xyzw(4)+gripper(1)+goal_direction(3)",
            "denoise_steps": denoise_steps, "image_size": image_size,
            "device": str(device),
        })

    @app.route("/reset", methods=["POST"])
    def reset():
        if hasattr(policy, "reset"):
            policy.reset()
        logger.info("Policy reset")
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        _s["count"] += 1
        t0 = time.time()

        data = request.get_json()
        # Accept both "state" (11D, new) and "state_12d" (legacy)
        state_raw = data.get("state") or data.get("state_12d") if data else None
        if not data or state_raw is None:
            return jsonify({"error": "Missing 'state' (11D)"}), 400

        # Image
        image = None
        if "image_b64" in data and data["image_b64"]:
            from PIL import Image as PILImage
            img_bytes = base64.b64decode(data["image_b64"])
            image = np.array(PILImage.open(io.BytesIO(img_bytes)).convert("RGB"))

        if image is None:
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        state = np.array(state_raw, dtype=np.float32)
        instruction = data.get("instruction", "move")

        try:
            batch = build_batch(image, state, instruction, stats, eagle, device, image_size)
            chunk = predict_and_denorm(policy, batch, stats)
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

        ms = (time.time() - t0) * 1000
        if _s["count"] <= 5 or _s["count"] % 10 == 0:
            logger.info(f"[{_s['count']}] {ms:.0f}ms  action[0]={chunk[0,:3].round(5).tolist()}")

        return jsonify({"action_chunk": chunk.tolist(), "inference_ms": round(ms, 1)})

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SO101 GROOT Inference Server")
    parser.add_argument("--adapter", required=True, help="LoRA adapter checkpoint 경로")
    parser.add_argument("--stats", required=True, help="data_stats.json 경로")
    parser.add_argument("--base-model", default="nvidia/GR00T-N1.5-3B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    import torch
    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("SO101 GROOT Inference Server")
    logger.info(f"  Adapter:  {args.adapter}")
    logger.info(f"  Stats:    {args.stats}")
    logger.info(f"  Device:   {device}")
    logger.info(f"  Denoise:  {args.denoise_steps} steps")
    logger.info("=" * 60)

    stats = load_stats(args.stats)
    policy, eagle = load_groot(args.adapter, args.base_model, device)

    # Warmup
    logger.info("Warmup...")
    try:
        dummy_img = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
        dummy_state = np.zeros(11, dtype=np.float32)
        b = build_batch(dummy_img, dummy_state, "move", stats, eagle, device, args.image_size)
        _ = predict_and_denorm(policy, b, stats)
        logger.info("Warmup done")
    except Exception as e:
        logger.warning(f"Warmup failed (non-fatal): {e}")

    app = create_app(policy, eagle, stats, device, args.denoise_steps, args.image_size, args.adapter)
    logger.info(f"Listening on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
