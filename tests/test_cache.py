"""
Anthropic 提示词缓存独立测试
- 非流式 + 流式
- 可通过 --cache-rounds N 配置测试轮数 (默认 2: 1次创建 + 1次命中)
"""
import uuid as _uuid
import pytest
from tenacity import RetryError

from src.config import load_config
from src.middleware import RequestFailed, make_retry_decorator, ai_validate
from src.logger import get_curl_and_response, log_multi_phase
from src.protocols.anthropic import AnthropicBuilder
from conftest import get_client


config = load_config()


def _anthropic_cache_params():
    """生成 anthropic 缓存测试参数"""
    params = []
    proto = config.protocols.get("anthropic")
    if not proto or not proto.all_models:
        return params
    req_cfg = config.request_configs.get("anthropic", {})
    if not req_cfg.get("cache_control", False):
        return params
    for model in proto.models_non_thinking:
        params.append(("anthropic", model))
    return params


ANTHROPIC_CACHE_PARAMS = _anthropic_cache_params()


def _get_cache_rounds(request) -> int:
    """获取缓存测试轮数"""
    return request.config.getoption("--cache-rounds", default=2)


def _assert_cache_usage(usage: dict, phase: str) -> list[str]:
    """缓存专属断言，返回错误列表"""
    errors = []
    if phase == "create":
        has_creation = "cache_creation_input_tokens" in usage
        has_read = "cache_read_input_tokens" in usage
        if not has_creation and not has_read:
            errors.append(
                "usage 中缺少缓存字段 "
                "(cache_creation_input_tokens 和 cache_read_input_tokens 均不存在), "
                f"可能提示词 token 数不足最低缓存要求. usage={usage}")
        elif usage.get("cache_creation_input_tokens", 0) == 0 \
                and usage.get("cache_read_input_tokens", 0) == 0:
            errors.append(
                "缓存未创建 "
                f"(cache_creation_input_tokens=0, cache_read_input_tokens=0). "
                f"usage={usage}")
    elif phase.startswith("read"):
        if "cache_read_input_tokens" not in usage:
            errors.append(
                f"usage 中缺少 cache_read_input_tokens 字段. usage={usage}")
        elif usage.get("cache_read_input_tokens", 0) == 0:
            errors.append(
                f"cache_read_input_tokens=0, 缓存未命中. usage={usage}")
    return errors


# ===== 非流式缓存测试 =====

@pytest.mark.parametrize("protocol_name,model", ANTHROPIC_CACHE_PARAMS,
                         ids=[f"anthropic-cache-{m}" for _, m in ANTHROPIC_CACHE_PARAMS])
@pytest.mark.asyncio
async def test_anthropic_prompt_caching(protocol_name, model, long_prompt, request):
    """Anthropic 提示词缓存测试 - 多轮请求验证缓存创建与命中

    测试逻辑:
    1. 第一次请求: 带 cache_control 的 system prompt → 期望 cache_creation_input_tokens > 0
    2. 后续请求: 相同内容 → 期望 cache_read_input_tokens > 0
    轮数通过 --cache-rounds 配置 (默认 2)
    """
    rounds = _get_cache_rounds(request)
    builder = AnthropicBuilder()
    client = get_client(config, "anthropic")
    session_id = _uuid.uuid4().hex[:12]
    test_name = f"cache_anthropic_{model}"

    retry_decorator = make_retry_decorator(config.retry)
    phase_records: list[dict] = []

    async def _single_request(body, phase: str):
        @retry_decorator
        async def _do():
            status, text, data = await client.request(body, model=model)
            method, url, headers = client.get_request_info(body, model=model)

            record = dict(phase=phase, method=method, url=url,
                          headers=headers, body=body,
                          status_code=status, response_body=text)
            phase_records.append(record)

            if status != 200:
                raise RequestFailed(status, text, f"HTTP {status}")

            errors = builder.assert_non_stream_response(data)

            usage = builder.extract_usage(data) or {}
            errors.extend(_assert_cache_usage(usage, phase))

            if errors:
                raise RequestFailed(status, text, "; ".join(errors))

            return data

        return await _do()

    body = builder.build_cache_test(model, long_prompt, session_id=session_id)

    try:
        # 第 1 轮: 创建缓存
        await _single_request(body, "create")
        # 第 2..N 轮: 命中缓存
        for i in range(1, rounds):
            await _single_request(body, f"read#{i}")

        log_multi_phase(test_name, phase_records, success=True)

    except (RetryError, RequestFailed) as e:
        last = e.last_attempt.exception() if isinstance(e, RetryError) else e
        reason = str(last)
        log_multi_phase(test_name, phase_records, success=False, reason=reason)

        if config.ai_validation.enabled and isinstance(last, RequestFailed):
            method, url, headers = client.get_request_info(body, model=model)
            curl_str, resp_str = get_curl_and_response(
                method, url, headers, body, last.status_code, last.body)
            expected, ai_reason = await ai_validate(
                config.ai_validation, curl_str, resp_str)
            if expected:
                pytest.skip(f"UNSTABLE: AI says expected - {ai_reason}")
        pytest.fail(f"Anthropic cache test failed: {last}")


# ===== 流式缓存测试 =====

@pytest.mark.parametrize("protocol_name,model", ANTHROPIC_CACHE_PARAMS,
                         ids=[f"anthropic-cache-stream-{m}" for _, m in ANTHROPIC_CACHE_PARAMS])
@pytest.mark.asyncio
async def test_anthropic_prompt_caching_stream(protocol_name, model, long_prompt, request):
    """Anthropic 提示词缓存流式测试 - 多轮流式请求验证缓存创建与命中

    测试逻辑与非流式版本相同，但使用 SSE 流式传输。
    轮数通过 --cache-rounds 配置 (默认 2)
    """
    rounds = _get_cache_rounds(request)
    builder = AnthropicBuilder()
    client = get_client(config, "anthropic")
    session_id = _uuid.uuid4().hex[:12]
    test_name = f"cache_stream_anthropic_{model}"

    retry_decorator = make_retry_decorator(config.retry)
    phase_records: list[dict] = []

    async def _single_request(body, phase: str):
        @retry_decorator
        async def _do():
            status, events, full_text = await client.request_stream(
                body, model=model)
            method, url, headers = client.get_request_info(
                body, model=model, stream=True)

            record = dict(phase=phase, method=method, url=url,
                          headers=headers, body=body,
                          status_code=status, response_body=full_text)
            phase_records.append(record)

            if status != 200:
                raise RequestFailed(status, full_text, f"HTTP {status}")

            errors = builder.assert_stream_events(events)

            usage = builder.extract_stream_usage(events) or {}
            errors.extend(_assert_cache_usage(usage, phase))

            if errors:
                raise RequestFailed(status, full_text, "; ".join(errors))

            return events

        return await _do()

    body = builder.build_cache_test(model, long_prompt, session_id=session_id)
    body["stream"] = True

    try:
        await _single_request(body, "create")
        for i in range(1, rounds):
            await _single_request(body, f"read#{i}")

        log_multi_phase(test_name, phase_records, success=True)

    except (RetryError, RequestFailed) as e:
        last = e.last_attempt.exception() if isinstance(e, RetryError) else e
        reason = str(last)
        log_multi_phase(test_name, phase_records, success=False, reason=reason)

        if config.ai_validation.enabled and isinstance(last, RequestFailed):
            method, url, headers = client.get_request_info(
                body, model=model, stream=True)
            curl_str, resp_str = get_curl_and_response(
                method, url, headers, body, last.status_code, last.body)
            expected, ai_reason = await ai_validate(
                config.ai_validation, curl_str, resp_str)
            if expected:
                pytest.skip(f"UNSTABLE: AI says expected - {ai_reason}")
        pytest.fail(f"Anthropic cache stream test failed: {last}")
