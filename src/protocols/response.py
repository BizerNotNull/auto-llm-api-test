"""OpenAI Response API 协议实现"""
import base64
import json5
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
    "type": "function",
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
}


class ResponseBuilder(ProtocolBuilder):

    def build_minimal(self, model: str, prompt: str) -> dict:
        return {
            "model": model,
            "input": prompt,
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
            # ===== 系统指令 =====
            "instructions": {"instructions": "You are a helpful assistant. Be concise."},

            # ===== 采样参数 =====
            "temperature": {"temperature": 0.7},
            "top_p": {"top_p": 0.9},

            # ===== 输出控制 =====
            "max_output_tokens": {"max_output_tokens": 1024},
            "text": {"text": {"format": {"type": "text"}}},

            # ===== 工具调用 =====
            "tools": {"tools": [_TOOL_DEF]},
            "tool_choice": {"tools": [_TOOL_DEF], "tool_choice": "auto"},
            "function_call": {
                "tools": [_TOOL_DEF],
                "tool_choice": "required",
                "input": "What is the weather in San Francisco?",
            },
            "parallel_tool_calls": {"tools": [_TOOL_DEF], "parallel_tool_calls": True},

            # ===== 多模态 - 图片 URL =====
            "input_image_url": {
                "input": [
                    {"role": "user", "content": [
                        {"type": "input_text", "text": "What is in this image?"},
                        {"type": "input_image", "image_url": "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"},
                    ]},
                ],
            },

            # ===== 多模态 - 图片 base64 =====
            "input_image_base64": {
                "input": [
                    {"role": "user", "content": [
                        {"type": "input_text", "text": "What is in this image?"},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                    ]},
                ],
            } if img_b64 else {},

            # ===== 推理模型 =====
            "reasoning": {"reasoning": {"effort": "medium"}},

            # ===== 多轮上下文 =====
            "previous_response_id": {},  # 需要先有一个 response, 暂不构建

            # ===== 响应包含项 =====
            "include": {"include": ["usage"]},

            # ===== 截断策略 =====
            "truncation": {"truncation": "auto"},

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
        if data.get("status") != "completed":
            errors.append(f"Expected status='completed', got '{data.get('status')}'")
        output = data.get("output")
        if not output:
            errors.append("Missing or empty 'output'")
        else:
            has_message = False
            for item in output:
                if item.get("type") == "message":
                    has_message = True
                    content = item.get("content", [])
                    if not content:
                        errors.append("Empty content in output message")
                    else:
                        has_text = any(
                            c.get("type") == "output_text" and c.get("text")
                            for c in content
                        )
                        if not has_text:
                            errors.append("No output_text in message content")
                elif item.get("type") == "function_call":
                    has_message = True  # function_call 也算有效输出
            if not has_message:
                errors.append("No message or function_call in output")
        return errors

    def assert_stream_events(self, events: list[str]) -> list[str]:
        errors = []
        if not events:
            errors.append("No stream events received")
            return errors

        event_types = set()
        has_text = False
        has_function_call = False
        for ev in events:
            if ev == "[DONE]":
                continue
            try:
                data = json5.loads(ev)
                etype = data.get("type", "")
                event_types.add(etype)
                if etype == "response.output_text.delta":
                    if data.get("delta"):
                        has_text = True
                elif etype == "response.function_call_arguments.delta":
                    has_function_call = True
            except Exception:
                pass

        if "response.created" not in event_types:
            errors.append("Missing 'response.created' event")
        if "response.completed" not in event_types:
            errors.append("Missing 'response.completed' event")
        if not has_text and not has_function_call:
            errors.append("No text delta or function_call in stream events")
        return errors

    def assert_option_response(self, option_name: str, data: dict) -> list[str]:
        errors = []
        output = data.get("output", [])

        if option_name in ("tools", "tool_choice", "parallel_tool_calls"):
            # 检查是否有 function_call 输出项
            has_fc = any(item.get("type") == "function_call" for item in output)
            has_msg = any(item.get("type") == "message" for item in output)
            if not has_fc and not has_msg:
                errors.append(f"[{option_name}] No function_call or message in output")
            if has_fc:
                for item in output:
                    if item.get("type") == "function_call":
                        if not item.get("name"):
                            errors.append(f"[{option_name}] function_call.name is empty")
                        if "arguments" not in item:
                            errors.append(f"[{option_name}] function_call.arguments missing")

        elif option_name == "function_call":
            # tool_choice=required 强制函数调用，必须返回 function_call
            has_fc = any(item.get("type") == "function_call" for item in output)
            if not has_fc:
                errors.append("[function_call] Expected function_call in output (tool_choice=required) but got none")
            for item in output:
                if item.get("type") == "function_call":
                    if not item.get("name"):
                        errors.append("[function_call] function_call.name is empty")
                    if "arguments" not in item:
                        errors.append("[function_call] function_call.arguments missing")

        elif option_name == "text":
            # text format 配置后，输出里应有 output_text
            for item in output:
                if item.get("type") == "message":
                    content = item.get("content", [])
                    has_text = any(c.get("type") == "output_text" for c in content)
                    if not has_text:
                        errors.append("[text] No output_text in message content after text format config")

        elif option_name in ("input_image_url", "input_image_base64"):
            # 多模态：应有文本描述
            has_text = False
            for item in output:
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text" and c.get("text"):
                            has_text = True
            if not has_text:
                errors.append(f"[{option_name}] No text response for image description")

        elif option_name == "reasoning":
            has_reasoning = any(item.get("type") == "reasoning" for item in output)
            if not has_reasoning:
                errors.append("[reasoning] No reasoning item in output")

        elif option_name == "service_tier":
            if "service_tier" not in data:
                errors.append("[service_tier] Missing service_tier in response")

        return errors

    def extract_text_content(self, data: dict) -> str:
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        return c.get("text", "")
        return ""

    def build_multi_turn(self, model: str, turns: list[tuple[str, str]],
                         **kwargs) -> dict:
        input_items = [{"role": role, "content": content} for role, content in turns]
        body = {"model": model, "input": input_items, "stream": False}
        body.update(kwargs)
        return body

    def extract_usage(self, data: dict) -> dict | None:
        return data.get("usage")

    def extract_stream_usage(self, events: list[str]) -> dict | None:
        """usage 通常在 response.completed 事件中"""
        for ev in reversed(events):
            if ev == "[DONE]":
                continue
            try:
                data = json5.loads(ev)
                if data.get("type") == "response.completed":
                    resp = data.get("response", {})
                    return resp.get("usage")
                usage = data.get("usage")
                if usage:
                    return usage
            except Exception:
                pass
        return None
