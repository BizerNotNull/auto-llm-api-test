"""后置中间件 - tenacity 重试 + AI 检验"""
import json
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from src.config import Config, RetryConfig, AIValidationConfig


class RequestFailed(Exception):
    """请求失败异常（用于触发重试）"""
    def __init__(self, status_code: int, body: str, reason: str = ""):
        self.status_code = status_code
        self.body = body
        self.reason = reason
        super().__init__(reason)


def make_retry_decorator(retry_cfg: RetryConfig):
    """根据配置创建重试装饰器"""
    if not retry_cfg.enabled:
        def no_retry(func):
            return func
        return no_retry

    return retry(
        retry=retry_if_exception_type(RequestFailed),
        stop=stop_after_attempt(retry_cfg.max_attempts),
        wait=wait_exponential(
            multiplier=retry_cfg.multiplier,
            max=retry_cfg.max_wait,
        ),
        reraise=True,
    )


async def ai_validate(
    ai_cfg: AIValidationConfig,
    curl_str: str,
    response_str: str,
) -> tuple[bool, str]:
    """
    使用 AI 判断失败的请求是否符合预期
    返回: (is_expected, explanation)
    """
    if not ai_cfg.enabled or not ai_cfg.api_key:
        return False, "AI validation not enabled"

    prompt = f"""You are an API testing expert. Analyze the following failed API request and response.
Determine if the failure is EXPECTED behavior (e.g., invalid parameter correctly rejected,
unsupported feature properly returning error) or UNEXPECTED (e.g., server error, wrong error format,
feature that should work but doesn't).

Request:
```
{curl_str}
```

Response:
```
{response_str}
```

Respond in this JSON format:
{{"expected": true/false, "reason": "brief explanation"}}
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ai_cfg.api_key}",
    }
    body = {
        "model": ai_cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{ai_cfg.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=body,
            )
            if resp.status_code != 200:
                return False, f"AI validation request failed: HTTP {resp.status_code}"

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            # 尝试解析 JSON
            import json5
            result = json5.loads(content)
            return result.get("expected", False), result.get("reason", "")
    except Exception as e:
        return False, f"AI validation error: {e}"
