"""Anthropic Messages 协议实现"""
import base64
import json5
import uuid
from pathlib import Path
from src.protocols.base import ProtocolBuilder

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "images"


def _load_test_image_base64() -> str:
    img_path = IMAGES_DIR / "test.png"
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


_TOOL_DEF = {
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name, e.g. San Francisco"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}


class AnthropicBuilder(ProtocolBuilder):

    def build_minimal(self, model: str, prompt: str) -> dict:
        return {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }

    def build_non_stream(self, model: str, prompt: str, **kwargs) -> dict:
        body = self.build_minimal(model, prompt)
        body["stream"] = False
        body.update(kwargs)
        return body

    def build_stream(self, model: str, prompt: str, **kwargs) -> dict:
        body = self.build_minimal(model, prompt)
        body["stream"] = True
        body.update(kwargs)
        return body

    def build_with_option(self, model: str, prompt: str,
                          option_name: str, **kwargs) -> dict:
        img_b64 = _load_test_image_base64()

        option_values = {
            # ===== 系统提示 =====
            "system": {"system": "You are a helpful assistant. Be concise."},

            # ===== 采样参数 =====
            "temperature": {"temperature": 0.7},
            "top_p": {"top_p": 0.9},
            "top_k": {"top_k": 40},

            # ===== 输出控制 =====
            "stop_sequences": {"stop_sequences": ["\n\n\n"]},

            # ===== 工具调用 =====
            "tools": {"tools": [_TOOL_DEF]},
            "tool_choice": {"tools": [_TOOL_DEF], "tool_choice": {"type": "auto"}},

            # ===== 多模态 - 图片 base64 =====
            "messages_image_base64": {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        }},
                    ]},
                ],
            } if img_b64 else {},

            # ===== 多模态 - 图片 URL =====
            "messages_image_url": {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image", "source": {
                            "type": "url",
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
                        }},
                    ]},
                ],
            },

            # ===== 多模态 - PDF base64 =====
            "messages_pdf_base64": {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is in this document?"},
                        {"type": "document", "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "",  # 需要实际 PDF base64
                        }},
                    ]},
                ],
            },

            # ===== 提示词缓存 =====
            "cache_control": {},  # 单独由 build_cache_test 处理

            # ===== 扩展思考 =====
            "thinking": {"thinking": {"type": "enabled", "budget_tokens": 5000}},

            # ===== 元数据 =====
            "metadata": {"metadata": {"user_id": "test-user-001"}},

            # ===== 服务层级 =====
            "service_tier": {},  # beta, 暂不构建
        }

        body = self.build_non_stream(model, prompt, **kwargs)
        if option_name in option_values:
            updates = option_values[option_name]
            if updates:
                body.update(updates)

        # thinking 需要更大的 max_tokens 且不能和 temperature 同时用
        if option_name == "thinking":
            body["max_tokens"] = 16000
            body.pop("temperature", None)
            body.pop("top_k", None)

        return body

    def build_cache_test(self, model: str, long_prompt: str) -> dict:
        """构建提示词缓存测试请求 - 需要足够的 token 量 + 随机前缀避免干扰"""
        random_prefix = f"[cache-test-{uuid.uuid4().hex[:8]}] "
        return {
            "model": model,
            "max_tokens": 256,
            "system": [
                {
                    "type": "text",
                    "text": random_prefix + long_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "Summarize in one sentence."}],
        }

    def assert_non_stream_response(self, data: dict) -> list[str]:
        errors = []
        if "id" not in data:
            errors.append("Missing 'id' in response")
        if data.get("type") != "message":
            errors.append(f"Expected type='message', got '{data.get('type')}'")
        if "content" not in data:
            errors.append("Missing 'content' in response")
        elif not data["content"]:
            errors.append("Empty 'content' array")
        else:
            valid_types = {"text", "tool_use", "thinking"}
            block = data["content"][0]
            if block.get("type") not in valid_types:
                errors.append(f"Unexpected content block type: {block.get('type')}")
        if data.get("role") != "assistant":
            errors.append(f"Expected role='assistant', got '{data.get('role')}'")
        if "stop_reason" not in data:
            errors.append("Missing 'stop_reason'")
        return errors

    def assert_stream_events(self, events: list[str]) -> list[str]:
        errors = []
        if not events:
            errors.append("No stream events received")
            return errors

        event_types = set()
        has_text = False
        for ev in events:
            if ev == "[DONE]":
                continue
            try:
                data = json5.loads(ev)
                etype = data.get("type", "")
                event_types.add(etype)
                if etype == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        has_text = True
            except Exception:
                pass

        if "message_start" not in event_types:
            errors.append("Missing 'message_start' event")
        if "content_block_start" not in event_types:
            errors.append("Missing 'content_block_start' event")
        if "message_stop" not in event_types:
            errors.append("Missing 'message_stop' event")
        if not has_text:
            errors.append("No text content in stream")
        return errors

    def assert_option_response(self, option_name: str, data: dict) -> list[str]:
        errors = []
        content_blocks = data.get("content", [])

        if option_name in ("tools", "tool_choice"):
            # 如果 stop_reason 是 tool_use，那么 content 里必须有 tool_use block
            if data.get("stop_reason") == "tool_use":
                has_tool_use = any(b.get("type") == "tool_use" for b in content_blocks)
                if not has_tool_use:
                    errors.append(f"[{option_name}] stop_reason='tool_use' but no tool_use block in content")
                for b in content_blocks:
                    if b.get("type") == "tool_use":
                        if not b.get("id"):
                            errors.append(f"[{option_name}] tool_use block missing 'id'")
                        if not b.get("name"):
                            errors.append(f"[{option_name}] tool_use block missing 'name'")
                        if "input" not in b:
                            errors.append(f"[{option_name}] tool_use block missing 'input'")

        elif option_name == "thinking":
            has_thinking = any(b.get("type") == "thinking" for b in content_blocks)
            if not has_thinking:
                errors.append("[thinking] No thinking block in content")
            for b in content_blocks:
                if b.get("type") == "thinking":
                    if not b.get("thinking"):
                        errors.append("[thinking] thinking block has empty 'thinking' text")

        elif option_name == "system":
            # system 提示不改变响应结构，只要有正常文本就行
            has_text = any(b.get("type") == "text" and b.get("text") for b in content_blocks)
            if not has_text:
                errors.append("[system] No text content in response after setting system prompt")

        elif option_name in ("messages_image_base64", "messages_image_url"):
            has_text = any(b.get("type") == "text" and b.get("text") for b in content_blocks)
            if not has_text:
                errors.append(f"[{option_name}] No text content for image description")
            else:
                text_len = sum(len(b.get("text", "")) for b in content_blocks if b.get("type") == "text")
                if text_len < 2:
                    errors.append(f"[{option_name}] Image description too short ({text_len} chars)")

        elif option_name == "stop_sequences":
            # 如果触发了 stop_sequence，stop_reason 应该是 stop_sequence
            # 但也可能没触发（内容太短），所以只在 stop_reason=stop_sequence 时检查
            if data.get("stop_reason") == "stop_sequence":
                if data.get("stop_sequence") is None:
                    errors.append("[stop_sequences] stop_reason='stop_sequence' but stop_sequence field is null")

        return errors

    def extract_usage(self, data: dict) -> dict | None:
        return data.get("usage")

    def extract_stream_usage(self, events: list[str]) -> dict | None:
        """anthropic 的 usage 在 message_start 和 message_delta 事件中"""
        usage = {}
        for ev in events:
            if ev == "[DONE]":
                continue
            try:
                data = json5.loads(ev)
                if data.get("type") == "message_start":
                    msg = data.get("message", {})
                    u = msg.get("usage", {})
                    usage.update(u)
                elif data.get("type") == "message_delta":
                    u = data.get("usage", {})
                    usage.update(u)
            except Exception:
                pass
        return usage if usage else None
