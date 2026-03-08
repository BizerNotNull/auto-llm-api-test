"""httpx 异步客户端封装"""
import httpx
import json5
from src.config import ProtocolConfig


class LLMClient:
    """LLM API 异步客户端"""

    def __init__(self, protocol: ProtocolConfig, timeout: float = 120.0):
        self.protocol = protocol
        self.timeout = timeout

    def _build_headers(self, extra_headers: dict | None = None) -> dict:
        """构建请求头"""
        headers = {"Content-Type": "application/json"}

        if self.protocol.name == "anthropic":
            if self.protocol.auth_header == "x-api-key":
                headers["x-api-key"] = self.protocol.api_key
            else:
                headers["Authorization"] = f"Bearer {self.protocol.api_key}"
            headers["anthropic-version"] = "2023-06-01"
        elif self.protocol.name == "vertex":
            headers["Authorization"] = f"Bearer {self.protocol.api_key}"
        else:
            # openai / response
            headers["Authorization"] = f"Bearer {self.protocol.api_key}"

        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _build_url(self, model: str = "", stream: bool = False) -> str:
        """构建请求 URL"""
        base = self.protocol.base_url.rstrip("/")

        if self.protocol.name == "openai":
            return f"{base}"
        elif self.protocol.name == "anthropic":
            return f"{base}"
        elif self.protocol.name == "vertex":
            action = "streamGenerateContent" if stream else "generateContent"
            return f"{base}/{model}:{action}"
        elif self.protocol.name == "response":
            return f"{base}"
        return base

    async def request(
        self,
        body: dict,
        model: str = "",
        stream: bool = False,
        extra_headers: dict | None = None,
    ) -> tuple[int, str, dict]:
        """
        发送非流式请求
        返回: (status_code, response_body_text, parsed_json_or_empty)
        """
        headers = self._build_headers(extra_headers)
        url = self._build_url(model=model, stream=stream)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
            text = resp.text
            try:
                parsed = json5.loads(text)
            except Exception:
                parsed = {}
            return resp.status_code, text, parsed

    async def request_stream(
        self,
        body: dict,
        model: str = "",
        extra_headers: dict | None = None,
    ) -> tuple[int, list[str], str]:
        """
        发送流式请求，收集所有 SSE 事件
        返回: (status_code, events_list, full_response_text)
        """
        headers = self._build_headers(extra_headers)
        url = self._build_url(model=model, stream=True)

        events = []
        full_text = ""

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                status = resp.status_code
                async for line in resp.aiter_lines():
                    full_text += line + "\n"
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            events.append("[DONE]")
                        else:
                            events.append(data)
                    elif line.strip() and not line.startswith(":"):
                        # Vertex 等非 SSE 格式也收集
                        events.append(line)

        return status, events, full_text

    def get_request_info(self, body: dict, model: str = "",
                         stream: bool = False,
                         extra_headers: dict | None = None) -> tuple[str, str, dict]:
        """获取请求信息（用于日志记录），返回 (method, url, headers)"""
        return "POST", self._build_url(model, stream), self._build_headers(extra_headers)
