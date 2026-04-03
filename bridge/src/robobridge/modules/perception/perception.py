"""
Perception Module

Detects objects in images and estimates poses.
Supports various providers: HuggingFace, OpenAI, custom models, etc.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from robobridge.modules.base import BaseModule

from .types import Detection, PerceptionConfig, PerceptionResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Perception(BaseModule):
    """
    Perception Module

    Detects objects in RGB/RGBD images and outputs object poses.

    Supported providers:
    - "hf": HuggingFace models (florence-2, owl-vit, etc.)
    - "openai": OpenAI Vision API
    - "local": Local custom models
    - "custom": Custom wrapper class

    Args:
        provider: Model provider
        model: Model identifier
        device: Compute device (cuda:0, cpu)
        api_key: API key for provider
        image_size: Input image resize size
        conf_threshold: Detection confidence threshold
        nms_threshold: NMS IoU threshold
        max_dets: Maximum detections per frame
        pose_format: Output format (bbox, mask, pose_quat)
        frame_id: Coordinate frame for poses
        link_mode: Connection mode (socket, in_proc)
        adapter_endpoint: (host, port) for socket mode
        adapter_protocol: Protocol type (len_json)
        auth_token: Authentication token
        rgb_topic: RGB image input topic
        depth_topic: Depth image input topic
        object_list_topic: Object list/prompt input topic
        output_topic: Detection output topic
        timeout_s: Operation timeout
        max_retries: Max retry attempts
    """

    def __init__(
        self,
        provider: str,
        model: str,
        device: str = "cuda:0",
        api_key: Optional[str] = None,
        image_size: int = 640,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.50,
        max_dets: int = 50,
        pose_format: str = "pose_quat",
        frame_id: str = "camera_color_optical_frame",
        link_mode: str = "direct",
        adapter_endpoint: Optional[Tuple[str, int]] = None,
        adapter_protocol: str = "len_json",
        auth_token: Optional[str] = None,
        rgb_topic: str = "/camera/rgb",
        depth_topic: Optional[str] = "/camera/depth",
        object_list_topic: Optional[str] = "/perception/object_list",
        output_topic: str = "/perception/objects",
        timeout_s: float = 3.0,
        max_retries: int = 2,
        **kwargs,
    ):
        super().__init__(
            provider=provider,
            model=model,
            device=device,
            api_key=api_key,
            link_mode=link_mode,
            adapter_endpoint=adapter_endpoint,
            adapter_protocol=adapter_protocol,
            auth_token=auth_token,
            timeout_s=timeout_s,
            max_retries=max_retries,
            **kwargs,
        )

        self.perception_config = PerceptionConfig(
            image_size=image_size,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_dets=max_dets,
            pose_format=pose_format,
            frame_id=frame_id,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            object_list_topic=object_list_topic,
            output_topic=output_topic,
        )

        self._model: Any = None
        self._processor: Any = None
        self._custom_wrapper: Any = None
        self._robocasa_adapter: Any = None  # RoboCasa ground truth adapter
        self._current_object_list: List[str] = []
        self._latest_rgb: Any = None
        self._latest_depth: Any = None

    def initialize_model(self) -> None:
        """Initialize the detection model based on provider."""
        provider = self.config.provider.lower()

        if provider in ("hf", "florence2"):
            self._init_hf_model()
        elif provider == "grounding_dino":
            self._init_grounding_dino()
        elif provider == "openai":
            self._init_openai_client()
        elif provider == "local":
            self._init_local_model()
        elif provider == "custom":
            self._init_custom_wrapper()
        elif provider == "robocasa_gt":
            self._init_robocasa_gt()
        else:
            logger.warning(f"Unknown provider: {provider}, using stub mode")

        logger.info(f"Initialized {provider} model: {self.config.model}")

    def _init_custom_wrapper(self) -> None:
        """Initialize custom wrapper from model path."""
        from robobridge.utils import load_custom_class
        from robobridge.wrappers import CustomPerception

        try:
            custom_cls = load_custom_class(self.config.model, CustomPerception)
            self._custom_wrapper = custom_cls(
                model_path=self.config.model,
                device=self.config.device,
                conf_threshold=self.perception_config.conf_threshold,
                frame_id=self.perception_config.frame_id,
                link_mode=self.config.link_mode,
                adapter_endpoint=self.config.adapter_endpoint,
                auth_token=self.config.auth_token,
            )
            self._custom_wrapper.load_model()
            logger.info(f"Loaded custom wrapper from: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to load custom wrapper: {e}")
            raise

    def _init_hf_model(self) -> None:
        """Initialize HuggingFace model (Florence-2 or similar)."""
        try:
            import torch

            model_name = self._get_hf_model_name()

            # Florence-2 uses Florence2ForConditionalGeneration
            if "florence" in model_name.lower():
                try:
                    from transformers import AutoProcessor, Florence2ForConditionalGeneration

                    self._processor = AutoProcessor.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                    )
                    self._model = Florence2ForConditionalGeneration.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16
                        if "cuda" in self.config.device
                        else torch.float32,
                    )
                except ImportError:
                    # Fallback for older transformers versions
                    from transformers import AutoModelForCausalLM, AutoProcessor

                    self._processor = AutoProcessor.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16
                        if "cuda" in self.config.device
                        else torch.float32,
                    )
            else:
                # Generic HuggingFace model loading
                from transformers import AutoModelForCausalLM, AutoProcessor

                self._processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if "cuda" in self.config.device else torch.float32,
                )

            if "cuda" in self.config.device:
                self._model = self._model.to(self.config.device)

            self._model.eval()
            logger.info(f"Loaded HuggingFace model: {model_name}")

        except ImportError:
            logger.warning("transformers not installed, using stub mode")
        except Exception as e:
            logger.error(f"Failed to load HF model: {e}")

    def _init_openai_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            self._model = OpenAI(api_key=self.config.api_key)
        except ImportError:
            logger.warning("openai not installed, using stub mode")

    def _init_local_model(self) -> None:
        """Initialize local model."""
        logger.info("Local model mode - implement custom loading")

    def _init_grounding_dino(self) -> None:
        """Initialize GroundingDINO model."""
        try:
            import torch
            from groundingdino.util.inference import load_model

            model_path = self.config.model or ""
            # Support config_path:checkpoint_path format
            if ":" in model_path and not model_path.startswith("/"):
                config_path, checkpoint_path = model_path.split(":", 1)
            else:
                config_path = getattr(self, '_dino_config_path', None) or os.environ.get(
                    "DINO_CONFIG_PATH",
                    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
                )
                checkpoint_path = model_path or (
                    "/opt/models/groundingdino/groundingdino_swint_ogc.pth"
                )

            logger.info(f"Loading GroundingDINO from {checkpoint_path}")
            self._model = load_model(config_path, checkpoint_path)

            if torch.cuda.is_available() and "cuda" in self.config.device:
                self._model = self._model.to(self.config.device)
                logger.info(f"GroundingDINO on {self.config.device}")
            else:
                logger.info("GroundingDINO on CPU")

        except ImportError:
            logger.warning("groundingdino not installed, using stub mode")
        except Exception as e:
            logger.error(f"Failed to load GroundingDINO: {e}")

    def _init_robocasa_gt(self) -> None:
        """Initialize RoboCasa ground truth perception adapter."""
        from robobridge.wrappers.robocasa_perception import RoboCasaPerception

        self._robocasa_adapter = RoboCasaPerception()
        logger.info("Initialized RoboCasa ground truth perception")

    def set_environment_state(
        self,
        obs: Dict[str, Any],
        ep_meta: Optional[Dict[str, Any]] = None,
        env: Optional[Any] = None,
    ) -> None:
        """
        Set environment state for simulation-based perception (e.g., RoboCasa).

        This is only needed for providers that use ground truth from simulation.
        For vision-based providers (hf, openai), this is a no-op.

        Args:
            obs: Environment observation dictionary
            ep_meta: Episode metadata
            env: Environment reference
        """
        if self._robocasa_adapter is not None:
            self._robocasa_adapter.set_environment_state(obs, ep_meta, env)

    def _get_hf_model_name(self) -> str:
        """Map model shortname to HuggingFace model path."""
        model_map = {
            # Florence-2 models
            "florence-2": "microsoft/Florence-2-base",
            "florence-2-base": "microsoft/Florence-2-base",
            "florence-2-large": "microsoft/Florence-2-large",
            "florence-2-base-ft": "microsoft/Florence-2-base-ft",
            "florence-2-large-ft": "microsoft/Florence-2-large-ft",
            # OWL-ViT models
            "owl-vit": "google/owlvit-base-patch32",
            "owl-vit-large": "google/owlvit-large-patch14",
        }
        return model_map.get(self.config.model, self.config.model)

    def start(self) -> None:
        """Start perception with model initialization."""
        self.initialize_model()
        super().start()

        # Register topic handlers
        self.subscribe(self.perception_config.rgb_topic, self._on_rgb)
        if self.perception_config.depth_topic:
            self.subscribe(self.perception_config.depth_topic, self._on_depth)
        if self.perception_config.object_list_topic:
            self.subscribe(self.perception_config.object_list_topic, self._on_object_list)

    def _on_rgb(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle RGB image message."""
        self._latest_rgb = payload
        self._run_detection(trace)

    def _on_depth(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle depth image message."""
        self._latest_depth = payload

    def _on_object_list(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle object list message."""
        if isinstance(payload, dict):
            self._current_object_list = payload.get("objects", [])
        elif isinstance(payload, list):
            self._current_object_list = payload
        else:
            self._current_object_list = [str(payload)]

        logger.debug(f"Updated object list: {self._current_object_list}")

    def _run_detection(self, trace: Optional[dict] = None) -> None:
        """Run detection on latest images."""
        if self._latest_rgb is None:
            return

        try:
            detections = self.process(
                rgb=self._latest_rgb,
                depth=self._latest_depth,
                object_list=self._current_object_list,
            )

            # Create result
            result = PerceptionResult(
                timestamp=time.time(),
                frame_id=self.perception_config.frame_id,
                detections=detections,
            )

            # Publish results
            self.publish(
                self.perception_config.output_topic,
                {"data": json.dumps(result.to_dict())},
                trace,
            )

        except Exception as e:
            logger.error(f"Detection error: {e}")

    def process(
        self,
        rgb: Any,
        depth: Optional[Any] = None,
        object_list: Optional[List[str]] = None,
    ) -> List[Detection]:
        """
        Run object detection.

        Args:
            rgb: RGB image (dict with base64 data or numpy array)
            depth: Optional depth image
            object_list: Optional list of target objects

        Returns:
            List of Detection
        """
        # HTTP remote mode
        if self.config.link_mode == "http":
            payload = {"object_list": object_list}
            if rgb is not None:
                payload["rgb_b64"] = self._encode_image(rgb)
            result = self._http_post("process", payload, timeout=10.0)
            return [Detection.from_dict(d) for d in result.get("detections", [])]

        if self._robocasa_adapter is not None:
            # Use RoboCasa ground truth adapter
            return self._robocasa_adapter.process(rgb, depth, object_list)
        elif self._custom_wrapper is not None:
            # Use custom wrapper's detect method
            custom_detections = self._custom_wrapper.detect(rgb, depth, object_list)
            return [
                Detection(
                    name=d.name,
                    confidence=d.confidence,
                    bbox=d.bbox,
                    pose=d.pose,
                    mask=d.mask,
                    frame_id=self.perception_config.frame_id,
                )
                if not isinstance(d, Detection)
                else d
                for d in custom_detections
            ]
        elif self.config.provider.lower() in ("hf", "florence2") and self._model:
            return self._detect_hf(rgb, depth, object_list)
        elif self.config.provider.lower() == "grounding_dino" and self._model:
            return self._detect_grounding_dino(rgb, object_list)
        elif self.config.provider.lower() == "openai" and self._model:
            return self._detect_openai(rgb, object_list)
        else:
            return self._detect_stub(rgb, object_list)

    def _detect_hf(
        self,
        rgb: Any,
        depth: Optional[Any],
        object_list: Optional[List[str]],
    ) -> List[Detection]:
        """
        Detect using HuggingFace model (Florence-2).

        Florence-2 task prompts:
        - "<OD>" : Object detection (returns bboxes + labels)
        - "<CAPTION_TO_PHRASE_GROUNDING>" : Grounded detection with text query
        - "<OPEN_VOCABULARY_DETECTION>" : Open vocabulary detection
        """
        import torch
        from PIL import Image
        import numpy as np

        logger.debug("Running HF detection")

        if self._model is None or self._processor is None:
            logger.warning("HF model not initialized")
            return []

        try:
            # Convert input to PIL Image
            image = self._prepare_image(rgb)
            if image is None:
                return []

            # Determine task prompt based on object_list
            model_name = self._get_hf_model_name().lower()

            if "florence" in model_name:
                return self._detect_florence2(image, object_list)
            else:
                # Generic detection (stub for other models)
                logger.warning(f"Detection not implemented for model: {model_name}")
                return []

        except Exception as e:
            logger.error(f"HF detection error: {e}")
            return []

    def _detect_florence2(
        self,
        image: Any,
        object_list: Optional[List[str]],
    ) -> List[Detection]:
        """Run Florence-2 object detection."""
        import torch

        # Choose task based on whether we have a target object list
        if object_list and len(object_list) > 0:
            # Use caption-to-phrase grounding for specific objects
            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            text_input = ", ".join(object_list)
            inputs = self._processor(
                text=task_prompt,
                images=image,
                return_tensors="pt",
            )
            # Add the text query for grounding
            inputs["input_ids"] = self._processor.tokenizer(
                task_prompt + text_input,
                return_tensors="pt",
            )["input_ids"]
        else:
            # Use standard object detection
            task_prompt = "<OD>"
            inputs = self._processor(
                text=task_prompt,
                images=image,
                return_tensors="pt",
            )

        # Move to device
        device = self._model.device
        dtype = next(self._model.parameters()).dtype
        inputs = {
            k: v.to(device, dtype) if v.dtype.is_floating_point else v.to(device)
            for k, v in inputs.items()
        }

        # Generate
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=3,
            )

        # Decode
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Post-process to get structured output
        image_size = image.size  # (width, height)
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=image_size,
        )

        # Convert to Detection objects
        detections = []
        results = parsed.get(task_prompt, parsed)

        if isinstance(results, dict):
            bboxes = results.get("bboxes", [])
            labels = results.get("labels", [])

            for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                # Florence-2 returns absolute pixel coordinates
                # Normalize to 0-1 range
                x1, y1, x2, y2 = bbox
                w, h = image_size
                norm_bbox = [x1 / w, y1 / h, x2 / w, y2 / h]

                # Skip if confidence would be below threshold (Florence-2 doesn't give scores)
                # Use position in list as proxy (earlier = more confident)
                confidence = max(0.5, 1.0 - (i * 0.05))

                if confidence >= self.perception_config.conf_threshold:
                    detections.append(
                        Detection(
                            name=label,
                            confidence=confidence,
                            bbox=norm_bbox,
                            pose=self._estimate_pose_from_bbox(norm_bbox),
                            frame_id=self.perception_config.frame_id,
                        )
                    )

                if len(detections) >= self.perception_config.max_dets:
                    break

        logger.debug(f"Florence-2 detected {len(detections)} objects")
        return detections

    def _detect_grounding_dino(
        self,
        rgb: Any,
        object_list: Optional[List[str]],
    ) -> List[Detection]:
        """Run GroundingDINO open-vocabulary detection."""
        import math
        import os
        import tempfile

        import numpy as np
        import torch
        from groundingdino.util.inference import load_image, predict
        from PIL import Image

        if self._model is None:
            logger.warning("GroundingDINO model not initialized")
            return []

        image = self._prepare_image(rgb)
        if image is None:
            return []

        W, H = image.size

        # GroundingDINO requires file path for load_image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, format="JPEG")
            temp_path = tmp.name

        try:
            out = load_image(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        # Extract tensor from load_image output
        processed_image = self._extract_dino_tensor(out)
        if processed_image is None:
            logger.error("load_image() did not return a tensor")
            return []

        processed_image = processed_image.float()
        if processed_image.dim() == 4 and processed_image.size(0) == 1:
            processed_image = processed_image.squeeze(0)

        device = next(self._model.parameters()).device
        processed_image = processed_image.to(device)

        box_threshold = self.perception_config.conf_threshold
        text_threshold = max(0.25, box_threshold - 0.10)

        detections = []
        targets = object_list or []

        for text_prompt in targets:
            try:
                boxes, logits, phrases = predict(
                    model=self._model,
                    image=processed_image,
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )

                if hasattr(boxes, "detach"):
                    boxes_cpu = boxes.detach().cpu()
                else:
                    boxes_cpu = np.array(boxes)

                if hasattr(logits, "detach"):
                    logits_list = logits.detach().cpu().tolist()
                else:
                    logits_list = list(logits) if isinstance(logits, (list, tuple)) else []

                best_conf = -1.0
                best_bbox = None

                n = min(len(boxes_cpu), len(logits_list))
                for i in range(n):
                    xyxy = self._dino_to_xyxy(boxes_cpu[i], W, H)
                    conf = self._dino_to_conf(logits_list[i])
                    if conf > best_conf:
                        best_conf = conf
                        best_bbox = xyxy

                if best_bbox is not None and best_conf >= box_threshold:
                    x1, y1, x2, y2 = best_bbox
                    norm_bbox = [x1 / W, y1 / H, x2 / W, y2 / H]
                    detections.append(
                        Detection(
                            name=text_prompt,
                            confidence=best_conf,
                            bbox=norm_bbox,
                            pose=self._estimate_pose_from_bbox(norm_bbox),
                            frame_id=self.perception_config.frame_id,
                        )
                    )

            except Exception as e:
                logger.error(f"GroundingDINO detection failed for '{text_prompt}': {e}")

        logger.debug(f"GroundingDINO detected {len(detections)} objects")
        return detections

    @staticmethod
    def _extract_dino_tensor(x):
        """Extract tensor from GroundingDINO load_image output."""
        import torch

        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            for item in x:
                t = Perception._extract_dino_tensor(item)
                if t is not None:
                    return t
        return None

    @staticmethod
    def _dino_to_conf(v) -> float:
        import math
        try:
            fv = float(v)
        except Exception:
            return 0.0
        if 0.0 <= fv <= 1.0:
            return fv
        try:
            return 1.0 / (1.0 + math.exp(-fv))
        except OverflowError:
            return 0.0 if fv < 0 else 1.0

    @staticmethod
    def _dino_to_xyxy(box, W: int, H: int) -> tuple:
        import torch

        if hasattr(box, "detach"):
            box = box.detach().cpu().numpy()
        b = list(map(float, box))
        if len(b) != 4:
            return (0, 0, 0, 0)

        maxv = max(abs(b[0]), abs(b[1]), abs(b[2]), abs(b[3]))
        if maxv <= 1.2:
            # Normalized cxcywh or xyxy
            x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
            if x2 <= x1 or y2 <= y1:
                cx, cy, ww, hh = b
                x1 = (cx - ww / 2) * W
                y1 = (cy - hh / 2) * H
                x2 = (cx + ww / 2) * W
                y2 = (cy + hh / 2) * H
        else:
            x1, y1, x2, y2 = b
            if x2 <= x1 or y2 <= y1:
                cx, cy, ww, hh = b
                x1, y1 = cx - ww / 2, cy - hh / 2
                x2, y2 = cx + ww / 2, cy + hh / 2

        x1 = int(max(0, min(W - 1, x1)))
        y1 = int(max(0, min(H - 1, y1)))
        x2 = int(max(x1 + 1, min(W, x2)))
        y2 = int(max(y1 + 1, min(H, y2)))
        return (x1, y1, x2, y2)

    def _prepare_image(self, rgb: Any) -> Any:
        """Convert various input formats to PIL Image."""
        from PIL import Image
        import base64
        import io
        import numpy as np

        try:
            # Already a PIL Image
            if isinstance(rgb, Image.Image):
                return rgb.convert("RGB")

            # Dict with base64 data
            if isinstance(rgb, dict):
                if "data" in rgb:
                    data = rgb["data"]
                    if isinstance(data, str):
                        # Base64 encoded
                        img_bytes = base64.b64decode(data)
                        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    elif hasattr(data, "shape"):
                        # Numpy array in dict
                        return Image.fromarray(data).convert("RGB")

            # Numpy array
            if hasattr(rgb, "shape"):
                arr = np.asarray(rgb)
                if arr.dtype != np.uint8:
                    arr = (arr * 255).astype(np.uint8)
                return Image.fromarray(arr).convert("RGB")

            logger.warning(f"Unknown image format: {type(rgb)}")
            return None

        except Exception as e:
            logger.error(f"Failed to prepare image: {e}")
            return None

    def _estimate_pose_from_bbox(self, bbox: List[float]) -> dict:
        """
        Estimate a simple 3D pose from bounding box.

        This is a rough estimate assuming:
        - Camera at origin looking down -Z
        - Objects are roughly 0.5m from camera
        - Bbox center maps to XY position
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Map normalized coords to approximate 3D position
        # Assuming ~60 degree FOV, objects at ~0.5m depth
        depth = 0.5
        fov_scale = 0.6  # tan(30 degrees) ~ 0.58

        x = (cx - 0.5) * depth * fov_scale * 2
        y = (cy - 0.5) * depth * fov_scale * 2
        z = depth

        return {
            "position": {"x": float(x), "y": float(y), "z": float(z)},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }

    def _detect_openai(
        self,
        rgb: Any,
        object_list: Optional[List[str]],
    ) -> List[Detection]:
        """Detect using OpenAI Vision API."""
        # Implementation would go here
        logger.debug("Running OpenAI detection")
        return []

    def _detect_stub(
        self,
        rgb: Any,
        object_list: Optional[List[str]],
    ) -> List[Detection]:
        """Stub detection for testing."""
        logger.debug("Running stub detection")
        results = []

        if object_list:
            for i, obj in enumerate(object_list[: self.perception_config.max_dets]):
                results.append(
                    Detection(
                        name=obj,
                        confidence=0.9 - (i * 0.1),
                        bbox=[0.1 + i * 0.1, 0.1, 0.3 + i * 0.1, 0.3],
                        pose={
                            "position": {"x": 0.5 + i * 0.1, "y": 0.0, "z": 0.3},
                            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                        },
                        frame_id=self.perception_config.frame_id,
                    )
                )

        return results
