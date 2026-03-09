"""日志记录模块 - 记录成功/失败的请求 curl 和 response"""
import json
import os
import datetime
from pathlib import Path

from src.config import LOGS_DIR

# 时间戳由 conftest.py 在 pytest_configure 阶段设置到环境变量
# 直接运行时 (非 pytest) 仍需要 fallback
_SESSION_TS = os.environ.get(
    "LLMTEST_SESSION_TS",
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
)


def _ensure_logs_dir():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_curl(method: str, url: str, headers: dict, body: dict | None,
                 redact: bool = True) -> str:
    """将请求格式化为 curl 命令

    Args:
        redact: 是否隐藏 API key 中间部分，默认 True
    """
    parts = [f"curl -X {method.upper()} '{url}'"]
    for k, v in headers.items():
        val = str(v)
        if redact and any(secret in k.lower() for secret in ["authorization", "x-api-key", "api-key"]):
            if len(val) > 12:
                val = val[:8] + "..." + val[-4:]
        parts.append(f"  -H '{k}: {val}'")
    if body is not None:
        body_str = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        parts.append(f"  -d '{body_str}'")
    return " \\\n".join(parts)


def _format_response(status_code: int, body: str) -> str:
    """格式化响应"""
    lines = [f"HTTP {status_code}"]
    try:
        parsed = json.loads(body)
        lines.append(json.dumps(parsed, ensure_ascii=False, indent=2))
    except (json.JSONDecodeError, TypeError):
        # 截断过长的响应
        text = body if len(body) <= 2000 else body[:2000] + "...(truncated)"
        lines.append(text)
    return "\n".join(lines)


def _write_log(filename: str, test_name: str, curl_str: str, response_str: str,
               extra: str = ""):
    """写入日志文件"""
    _ensure_logs_dir()
    path = LOGS_DIR / filename
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{_timestamp()}] {test_name}\n")
        if extra:
            f.write(f"{extra}\n")
        f.write(f"\n--- Request ---\n{curl_str}\n")
        f.write(f"\n--- Response ---\n{response_str}\n")


def log_success(test_name: str, method: str, url: str, headers: dict,
                body: dict | None, status_code: int, response_body: str):
    """记录成功的请求"""
    curl_str = format_curl(method, url, headers, body)
    resp_str = _format_response(status_code, response_body)
    _write_log(f"success_{_SESSION_TS}.log", test_name, curl_str, resp_str)


def log_failure(test_name: str, method: str, url: str, headers: dict,
                body: dict | None, status_code: int, response_body: str,
                reason: str = ""):
    """记录失败的请求"""
    curl_str = format_curl(method, url, headers, body)
    resp_str = _format_response(status_code, response_body)
    extra = f"Failure reason: {reason}" if reason else ""
    _write_log(f"failure_{_SESSION_TS}.log", test_name, curl_str, resp_str, extra)


def get_curl_and_response(method: str, url: str, headers: dict,
                          body: dict | None, status_code: int,
                          response_body: str) -> tuple[str, str]:
    """返回格式化后的 curl 和 response 字符串（用于 AI 检验）"""
    return (
        format_curl(method, url, headers, body),
        _format_response(status_code, response_body),
    )
