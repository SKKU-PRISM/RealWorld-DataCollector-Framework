"""
Monitor Module (Feedback)

Continuous monitoring module that observes execution and rapidly outputs
binary success/fail decisions with confidence scores only.

On failure detection:
1. Immediately publishes stop signal to halt robot
2. Publishes failure signal for recovery handling

The module does NOT make recovery decisions (retry/replan/etc).
Recovery decisions are made by the user or higher-level modules.

Uses LangChain for unified VLM interface across different providers.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from robobridge.modules.base import BaseModule

from .types import FeedbackResult, MonitorConfig, MonitoringStats

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Monitor(BaseModule):
    """
    Monitor Module (LangChain-based)

    Simplified continuous monitoring module that outputs ONLY:
    - success: bool (binary success/fail)
    - confidence: float (0.0-1.0)

    On failure detection:
    1. Immediately publishes /robot/stop to halt robot motion
    2. Publishes failure signal with success=False and confidence

    The module does NOT make recovery decisions (retry/replan/stop).
    Recovery is handled externally - user or Planner decides
    where to restart from (perception, planning, controller).

    Supported providers:
    - "openai": OpenAI Vision (gpt-4o, gpt-4-vision-preview)
    - "anthropic": Anthropic Claude Vision (claude-3-opus, claude-3-sonnet)
    - "google": Google Gemini Vision (gemini-pro-vision, gemini-1.5-pro)
    - "ollama": Local multimodal models (llava, bakllava)
    - "custom": Custom wrapper class

    Input Topics:
        - /camera/rgb: RGB image for visual monitoring
        - /planning/high_level_plan: Current plan (for context)
        - /robot/execution_result: Execution result (optional context)

    Output Topics:
        - /feedback/failure_signal: Published when failure detected
          Format: {"success": false, "confidence": float, "consecutive_failures": int}
        - /robot/stop: Published immediately on failure to halt robot

    Args:
        provider: VLM provider (openai, anthropic, google, ollama, custom)
        model: Model identifier
        observation_rate_hz: Observation frequency (default 10 Hz)
        failure_confidence_threshold: Min confidence to trigger failure (default 0.7)
        stop_on_consecutive_failures: Auto-stop after N failures (default 2)
    """

    def __init__(
        self,
        provider: str,
        model: str,
        device: str = "cuda:0",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 200,
        image_size: int = 224,
        base_prompt: Optional[str] = None,
        link_mode: str = "direct",
        adapter_endpoint: Optional[Tuple[str, int]] = None,
        adapter_protocol: str = "len_json",
        auth_token: Optional[str] = None,
        # Topics
        rgb_topic: str = "/camera/rgb",
        plan_topic: str = "/planning/high_level_plan",
        exec_result_topic: str = "/robot/execution_result",
        output_topic: str = "/feedback/failure_signal",
        robot_stop_topic: str = "/robot/stop",
        # Continuous monitoring settings
        observation_rate_hz: float = 10.0,
        only_publish_on_failure: bool = True,
        failure_confidence_threshold: float = 0.7,
        enable_continuous_mode: bool = True,
        stop_on_consecutive_failures: int = 2,
        timeout_s: float = 4.0,
        max_retries: int = 1,
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

        self.api_base = api_base
        self.monitor_config = MonitorConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            image_size=image_size,
            base_prompt=base_prompt or "",
            rgb_topic=rgb_topic,
            plan_topic=plan_topic,
            exec_result_topic=exec_result_topic,
            output_topic=output_topic,
            robot_stop_topic=robot_stop_topic,
            observation_rate_hz=observation_rate_hz,
            only_publish_on_failure=only_publish_on_failure,
            failure_confidence_threshold=failure_confidence_threshold,
            enable_continuous_mode=enable_continuous_mode,
            stop_on_consecutive_failures=stop_on_consecutive_failures,
        )

        self._llm: Any = None
        self._parser: Any = None
        self._custom_wrapper: Any = None
        self._use_custom_wrapper: bool = False
        self._use_hf_vlm: bool = False
        self._hf_vlm_model: Any = None
        self._hf_vlm_processor: Any = None

        self._current_plan: Optional[Dict] = None
        self._current_step_id: int = 0
        self._latest_rgb: Any = None
        self._latest_exec_result: Optional[Dict] = None
        self._consecutive_failures = 0
        self._last_failure_result = None  # Last failure analysis for direct access

        # Rolling window for failure detection
        self._rolling_window_size = 20  # last N observations
        self._rolling_failure_threshold = 0.85  # 85%+ failure rate → trigger
        self._rolling_results: list = []  # deque-like list of bools (True=fail)

        # Continuous monitoring state
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active: bool = False
        self._observation_count: int = 0
        self._failure_count: int = 0
        self._last_observation_time: float = 0.0
        self._monitoring_lock = threading.Lock()

    def _create_rapid_prompt(self, skill: str, target: str, instruction: str = "") -> str:
        """
        Create simplified prompt for rapid binary classification.

        Only asks for success/fail and confidence - nothing else.
        """
        instruction_line = f"\nTask: {instruction}" if instruction else ""
        return f"""Observe the robot executing: {skill} {target}{instruction_line}
Is the action succeeding? Output ONLY JSON:
{{"success": true/false, "confidence": 0.0-1.0}}"""

    def _create_analysis_prompt(self, skill: str, target: str, instruction: str = "") -> str:
        """
        Create prompt for Phase 2 failure cause analysis.

        Called once after robot is stopped to determine recovery_target.
        """
        instruction_line = f"\nOriginal task: {instruction}" if instruction else ""
        return f"""The robot failed while executing: {skill} {target}{instruction_line}
Analyze the image and determine the failure cause.
Choose the most appropriate recovery strategy:
- "perception": object position is wrong or object not visible (re-detect objects)
- "planning": the plan itself is flawed, need a different approach (re-plan from scratch)
- "controller": trajectory/motion issue, same plan but regenerate trajectory
- "retry": minor issue, just retry the current primitive

Output ONLY JSON:
{{"recovery_target": "perception|planning|controller|retry", "reason": "brief explanation"}}"""

    def _analyze_failure(self, rgb: Any, plan: Optional[Dict] = None) -> FeedbackResult:
        """
        Phase 2: Analyze failure cause to determine recovery_target.

        Called once after failure detection and robot stop.
        Reuses the existing VLM backend (_llm or _hf_vlm_model).

        Returns:
            FeedbackResult with recovery_target set
        """
        step_info = self._get_current_step_info(plan)
        skill = step_info.get("skill", "unknown")
        target = step_info.get("target_object", "unknown")
        instruction = step_info.get("instruction", "")

        default_result = FeedbackResult(
            success=False,
            confidence=0.0,
            recovery_target="controller",
            consecutive_failures=self._consecutive_failures,
            metadata={"phase": "analysis", "reason": "analysis_failed"},
        )

        try:
            analysis_prompt = self._create_analysis_prompt(skill, target, instruction)

            response_text = None

            if self._custom_wrapper is not None:
                # Custom wrappers don't support analysis; use default
                return default_result

            elif self._hf_vlm_model is not None and self._hf_vlm_processor is not None:
                response_text = self._analyze_failure_with_hf_vlm(rgb, analysis_prompt)

            elif self._llm is not None:
                response_text = self._analyze_failure_with_llm(rgb, analysis_prompt)

            if response_text is None:
                return default_result

            # Parse response
            result_json = self._extract_json(response_text)
            if result_json and "recovery_target" in result_json:
                recovery = result_json["recovery_target"]
                valid_targets = ("perception", "planning", "controller", "retry")
                if recovery not in valid_targets:
                    recovery = "controller"
                reason = result_json.get("reason", "")
                logger.info(f"Phase 2 analysis: recovery_target={recovery}, reason={reason}")
                return FeedbackResult(
                    success=False,
                    confidence=0.0,
                    recovery_target=recovery,
                    consecutive_failures=self._consecutive_failures,
                    metadata={"phase": "analysis", "reason": reason},
                )

            logger.warning("Phase 2 analysis: failed to parse response, defaulting to 'controller'")
            return default_result

        except Exception as e:
            logger.error(f"Phase 2 failure analysis error: {e}")
            return default_result

    def _analyze_failure_with_llm(self, rgb: Any, prompt: str) -> Optional[str]:
        """Run failure analysis using LangChain LLM."""
        try:
            from langchain_core.messages import HumanMessage

            content: list = []
            img_b64 = self._encode_image(rgb)
            if img_b64:
                provider = self.config.provider.lower()
                if provider == "openai":
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    })
                elif provider == "anthropic":
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    })
                elif provider == "bedrock":
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    })
                elif provider in ("google", "vertex"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    })

            content.append({"type": "text", "text": prompt})
            message = HumanMessage(content=content)
            response = self._llm.invoke([message])
            resp_content = response.content
            if isinstance(resp_content, list):
                resp_content = "\n".join(
                    p.get("text", str(p)) if isinstance(p, dict) else str(p)
                    for p in resp_content
                )
            return resp_content
        except Exception as e:
            logger.error(f"LLM failure analysis error: {e}")
            return None

    def _analyze_failure_with_hf_vlm(self, rgb: Any, prompt: str) -> Optional[str]:
        """Run failure analysis using local HuggingFace VLM."""
        try:
            import torch

            image = self._prepare_image_for_vlm(rgb)
            if image is None:
                return None

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = self._hf_vlm_processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            device = self._hf_vlm_model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = self._hf_vlm_model.generate(
                    **inputs,
                    max_new_tokens=self.monitor_config.max_tokens,
                )

            generated_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
            ]
            return self._hf_vlm_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
        except Exception as e:
            logger.error(f"HF VLM failure analysis error: {e}")
            return None

    def _create_llm(self) -> Any:
        """Create LangChain VLM based on provider."""
        provider = self.config.provider.lower()
        model = self.config.model
        api_key = self.config.api_key
        temperature = self.monitor_config.temperature
        max_tokens = self.monitor_config.max_tokens

        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=self.api_base,
                )
            except ImportError:
                logger.warning("langchain-openai not installed")
                return None

        elif provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic

                return ChatAnthropic(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except ImportError:
                logger.warning("langchain-anthropic not installed")
                return None

        elif provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            except ImportError:
                logger.warning("langchain-google-genai not installed")
                return None

        elif provider == "bedrock":
            from robobridge.modules.bedrock_bearer import create_bedrock_bearer_chat
            return create_bedrock_bearer_chat(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        elif provider == "vertex":
            try:
                from langchain_google_vertexai import ChatVertexAI
                return ChatVertexAI(
                    model_name=model,
                    project="prism-485101",
                    location="global",
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            except ImportError:
                logger.warning("langchain-google-vertexai not installed")
                return None

        elif provider == "ollama":
            try:
                from langchain_ollama import ChatOllama

                return ChatOllama(
                    model=model,
                    temperature=temperature,
                    base_url=self.api_base or "http://localhost:11434",
                )
            except ImportError:
                logger.warning("langchain-ollama not installed")
                return None

        elif provider == "custom":
            # Custom wrapper mode - will be initialized in initialize_client()
            self._use_custom_wrapper = True
            logger.info(f"Custom provider selected, will load wrapper from: {model}")
            return None

        elif provider in ("hf", "huggingface", "local"):
            # Local HuggingFace VLM - handled separately
            self._use_hf_vlm = True
            logger.info(f"HuggingFace VLM selected: {model}")
            return None

        else:
            logger.warning(f"Unknown provider: {provider}, using stub mode")
            return None

    def initialize_client(self) -> None:
        """Initialize LangChain VLM."""
        try:
            self._llm = self._create_llm()

            # Handle custom wrapper mode
            if self._use_custom_wrapper:
                self._init_custom_wrapper()
                return

            # Handle HuggingFace VLM mode
            if self._use_hf_vlm:
                self._init_hf_vlm()
                return

            if self._llm:
                logger.info(
                    f"Initialized LangChain VLM with {self.config.provider}/{self.config.model}"
                )

        except ImportError as e:
            logger.error(f"Failed to import LangChain provider: {e}")
            logger.info("Install the required package: pip install langchain-{provider}")
        except Exception as e:
            logger.error(f"Failed to initialize VLM: {e}")

    def _init_hf_vlm(self) -> None:
        """
        Initialize HuggingFace Vision-Language Model.

        Supports models like:
        - Qwen/Qwen2-VL-2B-Instruct
        - Qwen/Qwen2.5-VL-3B-Instruct
        - Qwen/Qwen3-VL-2B-Instruct
        - vikhyatk/moondream2
        """
        try:
            import torch
            from transformers import AutoProcessor

            # Map shortnames to full model paths
            model_map = {
                "qwen-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
                "qwen-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
                "qwen-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
                "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
                "moondream": "vikhyatk/moondream2",
            }
            model_name = model_map.get(self.config.model.lower(), self.config.model)

            logger.info(f"Loading HuggingFace VLM: {model_name}")

            # Load processor
            self._hf_vlm_processor = AutoProcessor.from_pretrained(model_name)

            # Load model - try Qwen3VL first, then Qwen2VL, fallback to generic
            if "qwen" in model_name.lower() and "vl" in model_name.lower():
                # Qwen3-VL uses Qwen3VLForConditionalGeneration
                if "qwen3" in model_name.lower():
                    try:
                        from transformers import Qwen3VLForConditionalGeneration

                        self._hf_vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16
                            if "cuda" in self.config.device
                            else torch.float32,
                            device_map="auto" if "cuda" in self.config.device else None,
                        )
                    except ImportError:
                        from transformers import AutoModelForVision2Seq

                        self._hf_vlm_model = AutoModelForVision2Seq.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16
                            if "cuda" in self.config.device
                            else torch.float32,
                            device_map="auto" if "cuda" in self.config.device else None,
                            trust_remote_code=True,
                        )
                # Qwen2-VL uses Qwen2VLForConditionalGeneration
                else:
                    try:
                        from transformers import Qwen2VLForConditionalGeneration

                        self._hf_vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16
                            if "cuda" in self.config.device
                            else torch.float32,
                            device_map="auto" if "cuda" in self.config.device else None,
                        )
                    except ImportError:
                        from transformers import AutoModelForVision2Seq

                        self._hf_vlm_model = AutoModelForVision2Seq.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16
                            if "cuda" in self.config.device
                            else torch.float32,
                            device_map="auto" if "cuda" in self.config.device else None,
                        )
            else:
                from transformers import AutoModelForVision2Seq

                self._hf_vlm_model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if "cuda" in self.config.device else torch.float32,
                    device_map="auto" if "cuda" in self.config.device else None,
                    trust_remote_code=True,
                )

            logger.info(f"Loaded HuggingFace VLM: {model_name}")

        except ImportError as e:
            logger.error(f"Failed to import HuggingFace dependencies: {e}")
            logger.info("Install with: pip install transformers torch")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace VLM: {e}")

    def _init_custom_wrapper(self) -> None:
        """Initialize custom wrapper from model path."""
        from robobridge.utils import load_custom_class
        from robobridge.wrappers import CustomMonitor

        try:
            custom_cls = load_custom_class(self.config.model, CustomMonitor)
            self._custom_wrapper = custom_cls(
                model_path=self.config.model,
                device=self.config.device,
                observation_rate_hz=self.monitor_config.observation_rate_hz,
                failure_confidence_threshold=self.monitor_config.failure_confidence_threshold,
                stop_on_consecutive_failures=self.monitor_config.stop_on_consecutive_failures,
                only_publish_on_failure=self.monitor_config.only_publish_on_failure,
                image_size=self.monitor_config.image_size,
                link_mode=self.config.link_mode,
                adapter_endpoint=self.config.adapter_endpoint,
                auth_token=self.config.auth_token,
            )
            self._custom_wrapper.load_model()
            logger.info(f"Loaded custom wrapper from: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to load custom wrapper: {e}")
            raise

    def start(self) -> None:
        """Start monitor module with continuous monitoring."""
        self.initialize_client()
        super().start()

        # Register topic handlers
        self.subscribe(self.monitor_config.rgb_topic, self._on_rgb)
        self.subscribe(self.monitor_config.plan_topic, self._on_plan)
        self.subscribe(self.monitor_config.exec_result_topic, self._on_exec_result)

        # Start continuous monitoring if enabled
        if self.monitor_config.enable_continuous_mode:
            self._start_continuous_monitoring()
            logger.info(
                f"Started continuous monitoring at {self.monitor_config.observation_rate_hz} Hz"
            )

    def stop(self) -> None:
        """Stop monitor module and monitoring thread."""
        self._stop_continuous_monitoring()
        super().stop()

    def _start_continuous_monitoring(self) -> None:
        """Start the continuous monitoring thread."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="MonitorLoop",
            daemon=True,
        )
        self._monitoring_thread.start()
        logger.info("Continuous monitoring thread started")

    def _stop_continuous_monitoring(self) -> None:
        """Stop the continuous monitoring thread."""
        self._monitoring_active = False
        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=2.0)
            self._monitoring_thread = None
        logger.info("Continuous monitoring thread stopped")

    def _monitoring_loop(self) -> None:
        """Main continuous monitoring loop - runs independently."""
        interval = 1.0 / self.monitor_config.observation_rate_hz

        while self._monitoring_active:
            loop_start = time.time()

            try:
                # Get current observation
                with self._monitoring_lock:
                    rgb = self._latest_rgb
                    plan = self._current_plan

                # Skip if no image available
                if rgb is None:
                    time.sleep(interval)
                    continue

                # Perform rapid binary classification
                logger.info(f"[MONITOR-THREAD] Calling VLM observe (plan={plan is not None})...")
                result = self._rapid_observe(rgb, plan)
                self._observation_count += 1

                # Log observation
                logger.info(
                    f"[MONITOR-THREAD] Observation #{self._observation_count}: "
                    f"success={result.success}, confidence={result.confidence:.2f}"
                )

                # Rolling window failure detection
                is_failure = not result.success
                self._rolling_results.append(is_failure)
                if len(self._rolling_results) > self._rolling_window_size:
                    self._rolling_results = self._rolling_results[-self._rolling_window_size:]

                if is_failure:
                    self._failure_count += 1
                    self._consecutive_failures += 1
                else:
                    self._consecutive_failures = 0

                # Check rolling failure rate (only after window is full)
                if len(self._rolling_results) >= self._rolling_window_size:
                    failure_rate = sum(self._rolling_results) / len(self._rolling_results)

                    if failure_rate >= self._rolling_failure_threshold:
                        result.consecutive_failures = self._consecutive_failures

                        # Phase 1: IMMEDIATELY stop the robot
                        self._publish_robot_stop()

                        # Phase 2: Analyze failure cause (one-shot)
                        analysis = self._analyze_failure(rgb, plan)
                        result.recovery_target = analysis.recovery_target
                        if analysis.metadata.get("reason"):
                            result.metadata["failure_reason"] = analysis.metadata["reason"]

                        # Store last failure result for direct access by execution loop
                        self._last_failure_result = result

                        # Publish failure signal with recovery_target
                        self._publish_failure_signal(result)

                        # Reset rolling window after trigger
                        self._rolling_results.clear()
                        self._consecutive_failures = 0

                        logger.warning(
                            f"FAILURE DETECTED - Robot stopped! "
                            f"(failure_rate: {failure_rate:.0%} over {self._rolling_window_size} obs, "
                            f"recovery: {result.recovery_target})"
                        )

                self._last_observation_time = time.time()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            # Maintain observation rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _rapid_observe(self, rgb: Any, plan: Optional[Dict] = None) -> FeedbackResult:
        """
        Perform rapid binary observation (success/fail).

        This is the core observation method called at high frequency.
        Optimized for speed - outputs ONLY success and confidence.
        """
        step_info = self._get_current_step_info(plan)

        try:
            if self._custom_wrapper is not None:
                # Use custom wrapper's observe method
                return self._custom_wrapper.observe(rgb=rgb, plan=plan, current_step=step_info)
            elif self._hf_vlm_model is not None:
                # Use local HuggingFace VLM
                result = self._rapid_analyze_with_hf_vlm(rgb, step_info)
            elif self._llm:
                result = self._rapid_analyze_with_llm(rgb, step_info)
            else:
                result = self._analyze_stub()

            return self._parse_result(result)

        except Exception as e:
            logger.error(f"Rapid observation error: {e}")
            return self._create_fallback_result()

    def _rapid_analyze_with_llm(self, rgb: Any, step_info: Dict) -> Dict:
        """
        Rapid VLM analysis optimized for speed.

        Only outputs success (bool) and confidence (float).
        """
        from langchain_core.messages import HumanMessage

        # Build simplified prompt
        skill = step_info.get("skill", "unknown")
        target = step_info.get("target_object", "unknown")
        instruction = step_info.get("instruction", "")
        rapid_prompt = self._create_rapid_prompt(skill, target, instruction)

        # Build message with image
        content: list = []
        img_b64 = self._encode_image(rgb)
        if img_b64:
            provider = self.config.provider.lower()
            if provider == "openai":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )
            elif provider == "anthropic":
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    }
                )
            elif provider == "google":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )

        content.append({"type": "text", "text": rapid_prompt})
        message = HumanMessage(content=content)

        # Invoke LLM
        response = self._llm.invoke([message])

        # Parse response - only extract success and confidence
        resp_content = response.content
        if isinstance(resp_content, list):
            resp_content = "\n".join(
                p.get("text", str(p)) if isinstance(p, dict) else str(p)
                for p in resp_content
            )
        result = self._extract_json(resp_content)
        if result:
            return {
                "success": result.get("success", True),
                "confidence": result.get("confidence", 0.5),
            }

        # Fallback
        return {"success": True, "confidence": 0.5}

    def _rapid_analyze_with_hf_vlm(self, rgb: Any, step_info: Dict) -> Dict:
        """
        Rapid VLM analysis using local HuggingFace model (Qwen2-VL, etc.).

        Only outputs success (bool) and confidence (float).
        """
        import torch
        from PIL import Image

        if self._hf_vlm_model is None or self._hf_vlm_processor is None:
            logger.warning("HuggingFace VLM not initialized")
            return {"success": True, "confidence": 0.5}

        try:
            # Prepare image
            image = self._prepare_image_for_vlm(rgb)
            if image is None:
                return {"success": True, "confidence": 0.5}

            # Build prompt
            skill = step_info.get("skill", "unknown")
            target = step_info.get("target_object", "unknown")
            instruction = step_info.get("instruction", "")
            rapid_prompt = self._create_rapid_prompt(skill, target, instruction)

            # Format for Qwen2-VL style models
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": rapid_prompt},
                    ],
                }
            ]

            # Apply chat template and tokenize
            inputs = self._hf_vlm_processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Move to device
            device = self._hf_vlm_model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.inference_mode():
                output_ids = self._hf_vlm_model.generate(
                    **inputs,
                    max_new_tokens=self.monitor_config.max_tokens,
                )

            # Decode - trim input tokens
            generated_ids = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
            ]
            response_text = self._hf_vlm_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Parse response
            result = self._extract_json(response_text)
            if result:
                return {
                    "success": result.get("success", True),
                    "confidence": result.get("confidence", 0.5),
                }

            # Try to infer from text if no JSON
            text_lower = response_text.lower()
            if "fail" in text_lower or "no" in text_lower or "not" in text_lower:
                return {"success": False, "confidence": 0.6}
            elif "success" in text_lower or "yes" in text_lower:
                return {"success": True, "confidence": 0.7}

            return {"success": True, "confidence": 0.5}

        except Exception as e:
            logger.error(f"HuggingFace VLM analysis error: {e}")
            return {"success": True, "confidence": 0.5}

    def _prepare_image_for_vlm(self, rgb: Any) -> Any:
        """Prepare image for VLM input."""
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
                        img_bytes = base64.b64decode(data)
                        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    elif hasattr(data, "shape"):
                        return Image.fromarray(data).convert("RGB")

            # Numpy array
            if hasattr(rgb, "shape"):
                arr = np.asarray(rgb)
                if arr.dtype != np.uint8:
                    arr = (arr * 255).astype(np.uint8)
                return Image.fromarray(arr).convert("RGB")

            return None

        except Exception as e:
            logger.error(f"Failed to prepare image for VLM: {e}")
            return None

    def _publish_robot_stop(self) -> None:
        """
        Publish immediate stop signal to halt robot motion.

        This is called IMMEDIATELY when a failure is detected,
        before any other processing.
        """
        stop_payload = {
            "command": "stop",
            "reason": "failure_detected",
            "timestamp": time.time(),
            "consecutive_failures": self._consecutive_failures,
        }

        self.publish(self.monitor_config.robot_stop_topic, stop_payload, None)
        logger.warning("ROBOT STOP signal published!")

    def _publish_failure_signal(self, result: FeedbackResult) -> None:
        """Publish failure signal to the feedback topic."""
        payload = {
            "data": json.dumps(result.to_dict()),
            "timestamp": time.time(),
            "observation_count": self._observation_count,
            "failure_count": self._failure_count,
        }

        self.publish(self.monitor_config.output_topic, payload, None)
        logger.info(
            f"Published failure signal "
            f"(confidence: {result.confidence:.2f}, "
            f"consecutive: {result.consecutive_failures})"
        )

    def _publish_observation(self, result: FeedbackResult) -> None:
        """Publish observation (success or failure) to the feedback topic."""
        payload = {
            "data": json.dumps(result.to_dict()),
            "timestamp": time.time(),
            "observation_count": self._observation_count,
        }

        self.publish(self.monitor_config.output_topic, payload, None)

    def get_monitoring_stats(self) -> MonitoringStats:
        """Get monitoring statistics."""
        return MonitoringStats(
            observation_count=self._observation_count,
            failure_count=self._failure_count,
            consecutive_failures=self._consecutive_failures,
            monitoring_active=self._monitoring_active,
            observation_rate_hz=self.monitor_config.observation_rate_hz,
            last_observation_time=self._last_observation_time,
        )

    def reset_monitoring_stats(self) -> None:
        """Reset monitoring statistics."""
        with self._monitoring_lock:
            self._observation_count = 0
            self._failure_count = 0
            self._consecutive_failures = 0

    def _on_rgb(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle RGB image message."""
        self._latest_rgb = payload

    def _on_plan(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle plan message."""
        if isinstance(payload, dict):
            if "data" in payload:
                try:
                    self._current_plan = json.loads(payload["data"])
                except (json.JSONDecodeError, TypeError):
                    self._current_plan = payload
            else:
                self._current_plan = payload
        elif isinstance(payload, str):
            try:
                self._current_plan = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                self._current_plan = {"raw": payload}

        if self._current_plan:
            self._current_step_id = self._current_plan.get("current_step_index", 0)
        self._consecutive_failures = 0
        logger.debug(f"Updated plan, current step: {self._current_step_id}")

    def _on_exec_result(self, payload: Any, trace: Optional[dict]) -> None:
        """Handle execution result message (optional context)."""
        if isinstance(payload, dict):
            if "data" in payload:
                try:
                    exec_result = json.loads(payload["data"])
                except (json.JSONDecodeError, TypeError):
                    exec_result = payload
            else:
                exec_result = payload
        elif isinstance(payload, str):
            try:
                exec_result = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                exec_result = {"raw": payload}
        else:
            exec_result = {"data": payload}

        with self._monitoring_lock:
            self._latest_exec_result = exec_result

        logger.debug(f"Received execution result for: {exec_result.get('command_id', 'unknown')}")

    def process(
        self, rgb: Optional[Any] = None, plan: Optional[Dict] = None
    ) -> Optional[FeedbackResult]:
        """
        Perform single observation and return result.

        This is the public interface for manual/on-demand observation.
        For continuous monitoring, the internal loop calls _rapid_observe().

        Args:
            rgb: Current RGB observation
            plan: Current plan (for context)

        Returns:
            FeedbackResult with success and confidence only
        """
        # HTTP remote mode
        if self.config.link_mode == "http":
            payload = {"plan": plan}
            if rgb is not None:
                payload["rgb_b64"] = self._encode_image(rgb)
            result = self._http_post("process", payload, timeout=10.0)
            if result is None:
                return None
            return FeedbackResult.from_dict(result)

        if rgb is None:
            rgb = self._latest_rgb
        if plan is None:
            plan = self._current_plan

        return self._rapid_observe(rgb, plan)

    def _get_current_step_info(self, plan: Optional[Dict]) -> Dict:
        """Extract current step information from plan for prompt context."""
        if not plan:
            return {"skill": "unknown", "target_object": "unknown"}

        steps = plan.get("steps", [])
        current_idx = plan.get("current_step_index", 0)

        if current_idx < len(steps):
            step = steps[current_idx]
            return {
                "skill": step.get("skill", "unknown"),
                "target_object": step.get("target_object", "unknown"),
            }

        return {"skill": "unknown", "target_object": "unknown"}

    def _analyze_stub(self) -> Dict:
        """Stub analysis for testing (no VLM)."""
        return {"success": True, "confidence": 0.9}

    def _encode_image(self, rgb: Any) -> Optional[str]:
        """Encode image to base64."""
        try:
            if isinstance(rgb, dict) and "data" in rgb:
                return rgb["data"]

            try:
                from PIL import Image
                import io

                if hasattr(rgb, "shape"):
                    img = Image.fromarray(rgb)
                else:
                    return None

                # Resize for efficiency
                img = img.resize((self.monitor_config.image_size, self.monitor_config.image_size))

                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode()
            except ImportError:
                logger.warning("PIL not installed, cannot encode image")
                return None

        except Exception as e:
            logger.warning(f"Failed to encode image: {e}")
            return None

    def _parse_result(self, data: Dict) -> FeedbackResult:
        """Parse VLM response to simplified FeedbackResult."""
        return FeedbackResult(
            success=data.get("success", True),
            confidence=data.get("confidence", 0.5),
            timestamp=time.time(),
            metadata={"provider": self.config.provider, "model": self.config.model},
        )

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text response."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _create_fallback_result(self) -> FeedbackResult:
        """Create fallback result when analysis fails."""
        return FeedbackResult(
            success=True,
            confidence=0.5,
            timestamp=time.time(),
            metadata={"fallback": True},
        )
