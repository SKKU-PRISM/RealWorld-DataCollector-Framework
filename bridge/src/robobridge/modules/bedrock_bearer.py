"""
Bedrock Bearer Token Chat Client.

Minimal LangChain-compatible wrapper that uses direct HTTP requests
with Bearer token authentication for AWS Bedrock ABSK API keys.

boto3/botocore use SigV4 signing which is incompatible with Bedrock
API Keys (ABSK format). This wrapper bypasses boto3 and calls the
Bedrock invoke endpoint directly with Bearer token in the Authorization header.

Supports automatic region failover on 429 rate limits.
Available regions for Opus 4.6:
  EU: eu-west-1, eu-west-2, eu-west-3, eu-central-1, eu-north-1 (prefix: eu.)
  US: us-east-1, us-west-2 (prefix: us.)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Region -> model prefix mapping
# EU regions use "eu." prefix, US regions use "us." prefix
REGION_PREFIX = {
    "eu-west-1": "eu.",
    "eu-west-2": "eu.",
    "eu-west-3": "eu.",
    "eu-central-1": "eu.",
    "eu-north-1": "eu.",
    "us-east-1": "us.",
    "us-east-2": "us.",
    "us-west-2": "us.",
}

# Default failover regions (ordered by priority)
DEFAULT_FAILOVER_REGIONS = [
    "eu-west-3",
    "eu-central-1",
    "eu-west-1",
    "eu-west-2",
    "eu-north-1",
    "us-east-1",
    "us-west-2",
]


def _model_for_region(base_model: str, region: str) -> str:
    """Get the correct model ID for a given region.

    Bedrock uses region-prefixed model IDs (eu.anthropic.xxx, us.anthropic.xxx).
    This swaps the prefix to match the target region.
    """
    prefix = REGION_PREFIX.get(region, "eu.")
    # Strip existing prefix (eu. or us.) if present
    model = base_model
    for p in ("eu.", "us."):
        if model.startswith(p):
            model = model[len(p):]
            break
    return f"{prefix}{model}"


@dataclass
class ChatResponse:
    """Minimal response object compatible with LangChain's AIMessage."""
    content: str
    response_metadata: Dict[str, Any] = field(default_factory=dict)


class BedrockBearerChat:
    """
    LangChain-compatible chat client using Bedrock Bearer token auth.

    Implements .invoke(messages) -> ChatResponse with .content attribute,
    matching the interface used by Planner and Monitor modules.

    Supports automatic failover:
      - Multiple bearer tokens (rotated on 429)
      - Multiple regions (rotated when all tokens exhausted in a region)
    """

    def __init__(
        self,
        model: str,
        bearer_token: str,
        region: str = "eu-west-3",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        extra_tokens: Optional[List[str]] = None,
        failover_regions: Optional[List[str]] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Build token pool
        self._tokens = [bearer_token]
        env_extras = os.environ.get("AWS_BEARER_TOKENS_EXTRA", "")
        if env_extras:
            self._tokens.extend(t.strip() for t in env_extras.split(",") if t.strip())
        if extra_tokens:
            self._tokens.extend(extra_tokens)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for t in self._tokens:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        self._tokens = unique
        self._token_idx = 0

        # Build region pool: primary region first, then failover regions
        if failover_regions:
            self._regions = failover_regions
        else:
            # Auto-build: start with primary, then add defaults
            self._regions = [region]
            for r in DEFAULT_FAILOVER_REGIONS:
                if r not in self._regions:
                    self._regions.append(r)
        self._region_idx = 0

        if len(self._tokens) > 1:
            logger.info(f"Bedrock token pool: {len(self._tokens)} keys")
        logger.info(f"Bedrock region pool: {len(self._regions)} regions "
                     f"({', '.join(self._regions)})")

    @property
    def bearer_token(self) -> str:
        return self._tokens[self._token_idx]

    @property
    def region(self) -> str:
        return self._regions[self._region_idx]

    @property
    def base_url(self) -> str:
        return f"https://bedrock-runtime.{self.region}.amazonaws.com"

    @property
    def _current_model(self) -> str:
        """Model ID adjusted for current region's prefix."""
        return _model_for_region(self.model, self.region)

    def _rotate_token(self) -> bool:
        """Rotate to next token. Returns True if wrapped around to first."""
        self._token_idx = (self._token_idx + 1) % len(self._tokens)
        return self._token_idx == 0

    def _rotate_region(self) -> bool:
        """Rotate to next region. Returns True if wrapped around to first."""
        old_region = self.region
        self._region_idx = (self._region_idx + 1) % len(self._regions)
        logger.warning(f"Switching region: {old_region} -> {self.region}")
        return self._region_idx == 0

    def invoke(self, messages: Any) -> ChatResponse:
        """
        Call Bedrock invoke endpoint with Bearer token.

        Failover strategy on 429:
          1. Rotate to next API key (if multiple keys)
          2. When all keys exhausted -> rotate to next region
          3. When all regions exhausted -> backoff and retry from first

        Args:
            messages: List of LangChain message objects or dicts.

        Returns:
            ChatResponse with .content string.
        """
        converted = self._convert_messages(messages)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": converted["messages"],
        }
        if converted.get("system"):
            body["system"] = converted["system"]

        backoff = 1.0

        while True:
            url = f"{self.base_url}/model/{self._current_model}/invoke"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.bearer_token}",
            }
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=60)
                resp.raise_for_status()
                result = resp.json()

                content_text = ""
                for block in result.get("content", []):
                    if block.get("type") == "text":
                        content_text += block["text"]

                return ChatResponse(
                    content=content_text,
                    response_metadata={
                        "model": result.get("model", self.model),
                        "stop_reason": result.get("stop_reason"),
                        "usage": result.get("usage", {}),
                        "region": self.region,
                    },
                )
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Step 1: Try next token
                    tokens_wrapped = self._rotate_token()
                    if not tokens_wrapped:
                        logger.warning(
                            f"Rate limited (429) in {self.region}, "
                            f"trying key {self._token_idx + 1}/{len(self._tokens)}"
                        )
                        continue

                    # Step 2: All tokens exhausted in this region -> try next region
                    regions_wrapped = self._rotate_region()
                    if not regions_wrapped:
                        logger.warning(
                            f"All keys exhausted in previous region, "
                            f"trying {self.region}"
                        )
                        continue

                    # Step 3: All regions exhausted -> backoff
                    wait = backoff + random.uniform(0, backoff * 0.5)
                    logger.warning(
                        f"All {len(self._regions)} regions rate limited, "
                        f"backoff {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 1.5, 60.0)
                    continue

                error_body = ""
                try:
                    error_body = e.response.text[:500]
                except Exception:
                    pass
                logger.error(f"Bedrock API error: {e.response.status_code} - {error_body}")
                raise
            except requests.exceptions.Timeout:
                logger.warning("Bedrock API timeout, retrying in 5s...")
                time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"Bedrock API call failed: {e}")
                raise

    def _convert_messages(self, messages: Any) -> Dict[str, Any]:
        """Convert LangChain messages to Anthropic Messages API format."""
        system_text = ""
        api_messages = []

        for msg in messages:
            role, content = self._extract_role_content(msg)

            if role == "system":
                system_text = content if isinstance(content, str) else str(content)
                continue

            if isinstance(content, str):
                api_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                })
            elif isinstance(content, list):
                # Already structured content (text + images)
                api_content = []
                for item in content:
                    if isinstance(item, dict):
                        api_content.append(self._convert_content_block(item))
                    elif isinstance(item, str):
                        api_content.append({"type": "text", "text": item})
                api_messages.append({"role": role, "content": api_content})
            else:
                api_messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": str(content)}],
                })

        result = {"messages": api_messages}
        if system_text:
            result["system"] = system_text
        return result

    def _extract_role_content(self, msg: Any) -> tuple:
        """Extract role and content from a LangChain message or dict."""
        if isinstance(msg, dict):
            return msg.get("role", "user"), msg.get("content", "")

        # LangChain message objects
        type_name = type(msg).__name__
        if "System" in type_name:
            return "system", msg.content
        elif "AI" in type_name or "Assistant" in type_name:
            return "assistant", msg.content
        else:
            return "user", msg.content

    def _convert_content_block(self, block: dict) -> dict:
        """Convert a content block to Anthropic format."""
        block_type = block.get("type", "text")

        if block_type == "text":
            return {"type": "text", "text": block.get("text", "")}

        elif block_type == "image":
            # Already in Anthropic format
            return block

        elif block_type == "image_url":
            # OpenAI format -> Anthropic format
            url = block.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                # data:image/png;base64,<data>
                parts = url.split(",", 1)
                media_type = parts[0].split(":")[1].split(";")[0]
                data = parts[1] if len(parts) > 1 else ""
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            return {"type": "text", "text": f"[image: {url}]"}

        return block


def create_bedrock_bearer_chat(
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    region: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> Optional[BedrockBearerChat]:
    """
    Factory function to create a BedrockBearerChat client.

    Reads bearer token and region from environment if not provided.
    Supports automatic failover:
      - AWS_BEARER_TOKEN_BEDROCK: primary token
      - AWS_BEARER_TOKENS_EXTRA: comma-separated additional tokens
      - AWS_BEARER_TOKEN_BEDROCK_1, _2, ... _9: numbered extra tokens
      - Automatic region failover across 7 regions on 429

    Returns None if no bearer token is available.
    """
    token = bearer_token or os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
    if not token:
        logger.error("No AWS_BEARER_TOKEN_BEDROCK set")
        return None

    rgn = region or os.environ.get("AWS_BEDROCK_REGION", "eu-west-3")

    # Collect extra tokens from numbered env vars
    extra_tokens = []
    for i in range(1, 10):
        t = os.environ.get(f"AWS_BEARER_TOKEN_BEDROCK_{i}", "")
        if t:
            extra_tokens.append(t)

    client = BedrockBearerChat(
        model=model,
        bearer_token=token,
        region=rgn,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_tokens=extra_tokens if extra_tokens else None,
    )
    n_keys = len(client._tokens)
    n_regions = len(client._regions)
    logger.info(f"Initialized Bedrock Bearer Chat: {model} ({rgn}), "
                f"{n_keys} key(s), {n_regions} failover regions")
    return client
