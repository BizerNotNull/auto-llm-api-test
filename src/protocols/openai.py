"""OpenAI Chat Completions 协议实现"""
import base64
import json5
from pathlib import Path
from src.protocols.base import ProtocolBuilder

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "images"


def _load_test_image_base64() -> str:
    """加载测试图片并返回 base64 编码"""
    img_path = IMAGES_DIR / "test.png"
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# 公共的工具定义，复用
_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. San Francisco"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}


class OpenAIBuilder(ProtocolBuilder):

    def build_minimal(self, model: str, prompt: str) -> dict:
        return {
            "model": model,
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
        body["stream_options"] = {"include_usage": True}
        body.update(kwargs)
        return body

    def build_with_option(self, model: str, prompt: str,
                          option_name: str, **kwargs) -> dict:
        """构建带指定可选参数的请求"""
        img_b64 = _load_test_image_base64()

        option_values = {
            # ===== 采样参数 =====
            "temperature": {"temperature": 0.7},
            "top_p": {"top_p": 0.9},
            "frequency_penalty": {"frequency_penalty": 0.5},
            "presence_penalty": {"presence_penalty": 0.5},
            "seed": {"seed": 42},
            "logprobs": {"logprobs": True},
            "top_logprobs": {"logprobs": True, "top_logprobs": 3},

            # ===== 输出控制 =====
            "max_tokens": {"max_tokens": 100},
            "max_completion_tokens": {"max_completion_tokens": 100},
            "stop": {"stop": ["\n\n\n"]},
            "n": {"n": 1},
            "response_format": {"response_format": {"type": "json_object"},
                                "messages": [
                                    {"role": "system",
                                     "content": "You are a helpful assistant. Always respond in valid JSON."},
                                    {"role": "user", "content": prompt},
                                ]},

            # ===== 工具调用 =====
            "tools": {"tools": [_TOOL_DEF]},
            "tool_choice": {"tools": [_TOOL_DEF], "tool_choice": "auto"},
            "parallel_tool_calls": {"tools": [_TOOL_DEF], "parallel_tool_calls": True},

            # ===== 多模态 =====
            "messages_image_url": {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
                        }},
                    ]},
                ],
            },
            "messages_image_base64": {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }},
                    ]},
                ],
            } if img_b64 else {},
            "modalities": {"modalities": ["text"]},
            "audio": {"modalities": ["text", "audio"],
                      "audio": {"voice": "alloy", "format": "wav"}},

            # ===== 推理模型 =====
            "reasoning_effort": {"reasoning_effort": "medium"},

            # ===== 预测输出 =====
            "prediction": {"prediction": {
                "type": "content",
                "content": "Hello! How can I help you today?"
            }},

            # ===== 网络搜索 =====
            "web_search_options": {"web_search_options": {"search_context_size": "medium"}},

            # ===== 流式选项 =====
            "stream_options": {},  # 仅在流式中生效，非流式下为空

            # ===== 元数据 =====
            "user": {"user": "test-user-001"},
            "store": {"store": True},
            "metadata": {"metadata": {"test_run": "auto-llm-api-test"}},

            # ===== 服务层级 =====
            "service_tier": {"service_tier": "auto"},
        }

        body = self.build_non_stream(model, prompt, **kwargs)
        if option_name in option_values:
            updates = option_values[option_name]
            if updates:
                body.update(updates)
        return body

    def assert_non_stream_response(self, data: dict) -> list[str]:
        errors = []
        if "id" not in data:
            errors.append("Missing 'id' in response")
        if "choices" not in data:
            errors.append("Missing 'choices' in response")
        elif not data["choices"]:
            errors.append("Empty 'choices' array")
        else:
            choice = data["choices"][0]
            if "message" not in choice:
                errors.append("Missing 'message' in choices[0]")
            elif ("content" not in choice["message"]
                  and "tool_calls" not in choice["message"]
                  and choice["message"].get("content") is not None):
                errors.append("Missing 'content' or 'tool_calls' in message")
            if "finish_reason" not in choice:
                errors.append("Missing 'finish_reason' in choices[0]")
        return errors

    def assert_stream_events(self, events: list[str]) -> list[str]:
        errors = []
        if not events:
            errors.append("No stream events received")
            return errors

        has_content = False
        has_done = False
        for ev in events:
            if ev == "[DONE]":
                has_done = True
                continue
            try:
                data = json5.loads(ev)
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content") or delta.get("tool_calls"):
                        has_content = True
            except Exception:
                pass

        if not has_content:
            errors.append("No content in stream events")
        if not has_done:
            errors.append("Missing [DONE] event")
        return errors

    def assert_option_response(self, option_name: str, data: dict) -> list[str]:
        errors = []
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})

        if option_name in ("tools", "tool_choice", "parallel_tool_calls"):
            # 请求中包含了 tools，响应应该有 tool_calls 或者普通 content
            # tool_choice=auto 时模型可以选择不调用，所以只在 finish_reason=tool_calls 时断言
            if choice.get("finish_reason") == "tool_calls":
                tc = message.get("tool_calls")
                if not tc:
                    errors.append(f"[{option_name}] finish_reason='tool_calls' but no tool_calls in message")
                else:
                    for i, call in enumerate(tc):
                        if call.get("type") != "function":
                            errors.append(f"[{option_name}] tool_calls[{i}].type != 'function'")
                        fn = call.get("function", {})
                        if not fn.get("name"):
                            errors.append(f"[{option_name}] tool_calls[{i}].function.name is empty")
                        if "arguments" not in fn:
                            errors.append(f"[{option_name}] tool_calls[{i}].function.arguments missing")

        elif option_name == "response_format":
            # response_format=json_object → content 必须是合法 JSON
            content = message.get("content", "")
            if content:
                try:
                    json5.loads(content)
                except Exception:
                    errors.append(f"[response_format] content is not valid JSON: {content[:100]}")

        elif option_name == "logprobs":
            lp = choice.get("logprobs")
            if lp is None:
                errors.append("[logprobs] logprobs is null in response")
            elif "content" not in lp:
                errors.append("[logprobs] logprobs.content missing")

        elif option_name == "top_logprobs":
            lp = choice.get("logprobs")
            if lp is None:
                errors.append("[top_logprobs] logprobs is null in response")
            elif lp.get("content"):
                first = lp["content"][0]
                top = first.get("top_logprobs", [])
                if not top:
                    errors.append("[top_logprobs] logprobs.content[0].top_logprobs is empty")

        elif option_name == "n":
            choices = data.get("choices", [])
            n_requested = 1  # 我们在 option_values 中设为 1
            if len(choices) < n_requested:
                errors.append(f"[n] Expected {n_requested} choices, got {len(choices)}")

        elif option_name == "seed":
            # seed 本身不改变响应结构，但 system_fingerprint 应存在以支持可复现
            if "system_fingerprint" not in data:
                errors.append("[seed] Missing system_fingerprint (needed for reproducibility)")

        elif option_name in ("messages_image_url", "messages_image_base64"):
            # 多模态：响应应有文本内容（对图片的描述）
            content = message.get("content", "")
            if not content or len(content.strip()) < 2:
                errors.append(f"[{option_name}] Response content is empty or too short for image description")

        elif option_name == "service_tier":
            if "service_tier" not in data:
                errors.append("[service_tier] Missing service_tier in response")

        return errors

    def extract_usage(self, data: dict) -> dict | None:
        return data.get("usage")

    def extract_stream_usage(self, events: list[str]) -> dict | None:
        """从流式事件中提取最后一个包含 usage 的事件"""
        for ev in reversed(events):
            if ev == "[DONE]":
                continue
            try:
                data = json5.loads(ev)
                usage = data.get("usage")
                if usage:
                    return usage
            except Exception:
                pass
        return None
