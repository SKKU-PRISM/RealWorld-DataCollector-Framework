"""
Custom Perception Wrapper

Wrap your own object detection model to use with RoboBridge.
Simply inherit this class and implement the `load_model` and `detect` methods.
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from robobridge.modules.base import BaseModule

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection."""

    name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] normalized 0-1
    pose: Optional[Dict[str, Any]] = None  # position + orientation
    mask: Optional[Any] = None
    frame_id: str = "camera_color_optical_frame"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "pose": self.pose,
            "frame_id": self.frame_id,
        }


class CustomPerception(BaseModule):
    """
    Base wrapper for custom object detection models.

    To use your own model:
    1. Inherit this class
    2. Implement `load_model()` to load your model
    3. Implement `detect()` to run inference

    Example:
        class MyYOLODetector(CustomPerception):
            def load_model(self):
                import torch
                self._model = torch.load(self.model_path)
                self._model.to(self.device)
                self._model.eval()

            def detect(self, rgb, depth=None, object_list=None):
                outputs = self._model(rgb)
                results = []
                for det in outputs:
                    if det.conf > self.conf_threshold:
                        results.append(Detection(
                            name=det.class_name,
                            confidence=float(det.conf),
                            bbox=det.bbox.tolist()
                        ))
                return results

    Input Topics:
        - /camera/rgb: RGB image (base64 or numpy)
        - /camera/depth: Depth image (optional)
        - /perception/object_list: Target objects (optional)

    Output Topics:
        - /perception/objects: Detection results
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda:0",
        # Connection settings
        link_mode: str = "direct",
        adapter_endpoint: Tuple[str, int] = ("127.0.0.1", 51001),
        auth_token: Optional[str] = None,
        # Topic settings
        rgb_topic: str = "/camera/rgb",
        depth_topic: str = "/camera/depth",
        object_list_topic: str = "/perception/object_list",
        output_topic: str = "/perception/objects",
        # Detection settings
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.50,
        max_dets: int = 50,
        frame_id: str = "camera_color_optical_frame",
        **kwargs,
    ):
        super().__init__(
            provider="custom",
            model=model_path,
            device=device,
            link_mode=link_mode,
            adapter_endpoint=adapter_endpoint,
            auth_token=auth_token,
            **kwargs,
        )

        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_dets = max_dets
        self.frame_id = frame_id

        # Topics
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.object_list_topic = object_list_topic
        self.output_topic = output_topic

        # State
        self._model: Any = None
        self._latest_rgb: Any = None
        self._latest_depth: Any = None
        self._current_object_list: List[str] = []

    @abstractmethod
    def load_model(self) -> None:
        """
        Load your custom detection model.

        This method is called once during initialization.
        Store your model in self._model or any attribute you prefer.

        Example:
            def load_model(self):
                import torch
                self._model = torch.load(self.model_path)
                self._model.to(self.device)
                self._model.eval()
        """
        pass

    @abstractmethod
    def detect(
        self,
        rgb: Any,
        depth: Optional[Any] = None,
        object_list: Optional[List[str]] = None,
    ) -> List[Detection]:
        """
        Run object detection on image.

        Args:
            rgb: RGB image
                - dict with "data" key (base64) or numpy array
            depth: Depth image (optional)
            object_list: List of target object names (optional)

        Returns:
            List of Detection objects

        Example:
            def detect(self, rgb, depth=None, object_list=None):
                image = self._preprocess(rgb)
                with torch.no_grad():
                    outputs = self._model(image)
                results = []
                for det in outputs:
                    if det.conf > self.conf_threshold:
                        results.append(Detection(
                            name=det.class_name,
                            confidence=float(det.conf),
                            bbox=det.bbox.tolist(),
                            pose=self._estimate_pose(det, depth)
                        ))
                return results
        """
        pass

    def start(self) -> None:
        """Start detector with model initialization."""
        logger.info(f"Loading custom detection model from: {self.model_path}")
        self.load_model()
        logger.info("Custom detection model loaded successfully")

        super().start()

        # Register topic handlers
        self.subscribe(self.rgb_topic, self._on_rgb)
        self.subscribe(self.depth_topic, self._on_depth)
        self.subscribe(self.object_list_topic, self._on_object_list)

    def _on_rgb(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle RGB image message and run detection."""
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

    def _run_detection(self, trace: Optional[dict] = None) -> None:
        """Run detection on latest images."""
        if self._latest_rgb is None:
            return

        try:
            detections = self.detect(
                rgb=self._latest_rgb,
                depth=self._latest_depth,
                object_list=self._current_object_list,
            )

            # Publish results
            result = {
                "detections": [d.to_dict() for d in detections],
                "frame_id": self.frame_id,
            }

            self.publish(self.output_topic, {"data": json.dumps(result)}, trace)

        except Exception as e:
            logger.error(f"Detection error: {e}")

    def process(self, *args, **kwargs) -> Any:
        """Required by BaseModule - use detect() for detection."""
        return self.detect(*args, **kwargs)
