"""Vertex AI (Gemini) 协议实现"""
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
    "functionDeclarations": [{
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {
                    "type": "STRING",
                    "description": "City name, e.g. San Francisco",
                },
                "unit": {
                    "type": "STRING",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    }],
}


class VertexBuilder(ProtocolBuilder):

    def build_minimal(self, model: str, prompt: str) -> dict:
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
        }

    def build_non_stream(self, model: str, prompt: str, **kwargs) -> dict:
        body = self.build_minimal(model, prompt)
        body.update(kwargs)
        return body

    def build_stream(self, model: str, prompt: str, **kwargs) -> dict:
        body = self.build_minimal(model, prompt)
        body.update(kwargs)
        return body

    def build_with_option(self, model: str, prompt: str,
                          option_name: str, **kwargs) -> dict:
        img_b64 = _load_test_image_base64()

        option_values = {
            # ===== 系统指令 =====
            "systemInstruction": {
                "systemInstruction": {
                    "parts": [{"text": "You are a helpful assistant. Be concise."}]
                }
            },

            # ===== generationConfig 整体 =====
            "generationConfig": {
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 200,
                }
            },

            # ===== 采样参数 (各自放在 generationConfig 内) =====
            "temperature": {
                "generationConfig": {"temperature": 0.7}
            },
            "topP": {
                "generationConfig": {"topP": 0.9}
            },
            "topK": {
                "generationConfig": {"topK": 40}
            },
            "presencePenalty": {
                "generationConfig": {"presencePenalty": 0.5}
            },
            "frequencyPenalty": {
                "generationConfig": {"frequencyPenalty": 0.5}
            },
            "seed": {
                "generationConfig": {"seed": 42}
            },

            # ===== 输出控制 =====
            "maxOutputTokens": {
                "generationConfig": {"maxOutputTokens": 100}
            },
            "stopSequences": {
                "generationConfig": {"stopSequences": ["\n\n\n"]}
            },
            "candidateCount": {
                "generationConfig": {"candidateCount": 1}
            },
            "responseMimeType": {
                "generationConfig": {"responseMimeType": "application/json"}
            },
            "responseSchema": {
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "answer": {"type": "STRING"},
                        },
                        "required": ["answer"],
                    },
                }
            },
            "responseLogprobs": {
                "generationConfig": {"responseLogprobs": True}
            },
            "logprobs": {
                "generationConfig": {"responseLogprobs": True, "logprobs": 3}
            },

            # ===== 工具调用 =====
            "tools": {
                "tools": [_TOOL_DEF],
            },
            "toolConfig": {
                "tools": [_TOOL_DEF],
                "toolConfig": {
                    "functionCallingConfig": {"mode": "AUTO"}
                },
            },
            "function_call": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "What is the weather in San Francisco?"}],
                    }
                ],
                "tools": [_TOOL_DEF],
                "toolConfig": {
                    "functionCallingConfig": {"mode": "ANY"}
                },
            },

            # ===== 多模态 - 图片 base64 =====
            "contents_image_base64": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": "What is in this image?"},
                            {"inlineData": {
                                "mimeType": "image/png",
                                "data": img_b64,
                            }},
                        ],
                    }
                ],
            } if img_b64 else {},

            # ===== 多模态 - 图片 URL =====
            "contents_image_url": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": "What is in this image?"},
                            {"fileData": {
                                "mimeType": "image/png",
                                "fileUri": "gs://cloud-samples-data/generative-ai/image/scones.jpg",
                            }},
                        ],
                    }
                ],
            },

            # ===== 安全设置 =====
            "safetySettings": {
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                ]
            },

            # ===== 缓存 =====
            "cachedContent": {},  # 需要先创建缓存资源，暂不构建
        }

        body = self.build_non_stream(model, prompt, **kwargs)
        if option_name in option_values:
            updates = option_values[option_name]
            if updates:
                # 对 generationConfig 做合并而非覆盖
                if "generationConfig" in updates and "generationConfig" in body:
                    body["generationConfig"].update(updates["generationConfig"])
                    # 合并完其他 key
                    for k, v in updates.items():
                        if k != "generationConfig":
                            body[k] = v
                else:
                    body.update(updates)
        return body

    def assert_non_stream_response(self, data: dict) -> list[str]:
        errors = []
        candidates = data.get("candidates")
        if not candidates:
            errors.append("Missing or empty 'candidates'")
            return errors
        candidate = candidates[0]
        content = candidate.get("content")
        if not content:
            errors.append("Missing 'content' in candidate")
        else:
            parts = content.get("parts")
            if not parts:
                errors.append("Missing or empty 'parts' in content")
            elif not any(p.get("text") or p.get("functionCall") for p in parts):
                errors.append("No text or functionCall in parts")
        return errors

    def assert_stream_events(self, events: list[str]) -> list[str]:
        errors = []
        if not events:
            errors.append("No stream events received")
            return errors

        has_text = False
        for ev in events:
            if ev == "[DONE]":
                continue
            try:
                data = json5.loads(ev)
                candidates = data.get("candidates", [])
                for c in candidates:
                    parts = c.get("content", {}).get("parts", [])
                    for p in parts:
                        if p.get("text"):
                            has_text = True
            except Exception:
                pass

        if not has_text:
            errors.append("No text content in stream events")
        return errors

    def assert_option_response(self, option_name: str, data: dict) -> list[str]:
        errors = []
        candidates = data.get("candidates", [])
        candidate = candidates[0] if candidates else {}
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        if option_name in ("tools", "toolConfig"):
            # 如果模型选择了函数调用，parts 里应有 functionCall
            has_fc = any(p.get("functionCall") for p in parts)
            has_text = any(p.get("text") for p in parts)
            if not has_fc and not has_text:
                errors.append(f"[{option_name}] No functionCall or text in response parts")
            if has_fc:
                for i, p in enumerate(parts):
                    fc = p.get("functionCall")
                    if fc:
                        if not fc.get("name"):
                            errors.append(f"[{option_name}] functionCall[{i}].name is empty")
                        if "args" not in fc:
                            errors.append(f"[{option_name}] functionCall[{i}].args missing")

        elif option_name == "function_call":
            # mode=ANY 强制函数调用，必须返回 functionCall
            has_fc = any(p.get("functionCall") for p in parts)
            if not has_fc:
                errors.append("[function_call] Expected functionCall in response (mode=ANY) but got none")
            for i, p in enumerate(parts):
                fc = p.get("functionCall")
                if fc:
                    if not fc.get("name"):
                        errors.append(f"[function_call] functionCall[{i}].name is empty")
                    if "args" not in fc:
                        errors.append(f"[function_call] functionCall[{i}].args missing")

        elif option_name in ("responseMimeType", "responseSchema"):
            # 期望返回 JSON 内容
            text_parts = [p.get("text", "") for p in parts if p.get("text")]
            if text_parts:
                combined = text_parts[0]
                try:
                    json5.loads(combined)
                except Exception:
                    errors.append(f"[{option_name}] Response text is not valid JSON: {combined[:100]}")

        elif option_name in ("contents_image_base64", "contents_image_url"):
            has_text = any(p.get("text") for p in parts)
            if not has_text:
                errors.append(f"[{option_name}] No text response for image description")
            else:
                text_len = sum(len(p.get("text", "")) for p in parts if p.get("text"))
                if text_len < 2:
                    errors.append(f"[{option_name}] Image description too short ({text_len} chars)")

        elif option_name == "seed":
            # Vertex 有 seed 时不改变响应结构，但可以检查 modelVersion 存在
            if "modelVersion" not in data:
                errors.append("[seed] Missing modelVersion in response")

        elif option_name in ("responseLogprobs", "logprobs"):
            # 检查候选中有 avgLogprobs
            if "avgLogprobs" not in candidate and "logprobsResult" not in candidate:
                errors.append(f"[{option_name}] No logprobs data in candidate (avgLogprobs or logprobsResult)")

        return errors

    def extract_usage(self, data: dict) -> dict | None:
        return data.get("usageMetadata")

    def extract_stream_usage(self, events: list[str]) -> dict | None:
        """Vertex 的 usage 通常在最后一个事件中"""
        for ev in reversed(events):
            if ev == "[DONE]":
                continue
            try:
                data = json5.loads(ev)
                usage = data.get("usageMetadata")
                if usage:
                    return usage
            except Exception:
                pass
        return None
