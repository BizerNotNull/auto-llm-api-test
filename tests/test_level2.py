"""
第二层测试 - 功能可用性测试
- 每个非必须项是否可用（非流式 + 流式）
- anthropic beta 头测试（非流式 + 流式）
- 多轮对话测试（非流式 + 流式）
"""
import pytest
import json5
from tenacity import RetryError

from src.config import load_config, get_optional_fields
from src.client import LLMClient
from src.middleware import RequestFailed, make_retry_decorator, ai_validate
from src.logger import log_success, log_failure, get_curl_and_response, log_multi_phase
from src.protocols.anthropic import AnthropicBuilder
from conftest import get_builder, get_client


def _extract_text_from_stream_events(events: list[str], protocol_name: str) -> str:
    """从流式事件中提取完整文本内容（用于多轮对话流式测试的关键词验证）"""
    text_parts = []
    for ev in events:
        if ev == "[DONE]":
            continue
        try:
            data = json5.loads(ev)
        except Exception:
            continue

        if protocol_name == "openai":
            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    text_parts.append(delta["content"])
        elif protocol_name == "anthropic":
            if data.get("type") == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta" and delta.get("text"):
                    text_parts.append(delta["text"])
        elif protocol_name == "vertex":
            for c in data.get("candidates", []):
                for p in c.get("content", {}).get("parts", []):
                    if p.get("text"):
                        text_parts.append(p["text"])
        elif protocol_name == "response":
            if data.get("type") == "response.output_text.delta":
                if data.get("delta"):
                    text_parts.append(data["delta"])

    return "".join(text_parts)


def _make_stream_body(builder, protocol_name: str, body: dict) -> dict:
    """将非流式请求体转换为流式请求体"""
    stream_body = dict(body)
    if protocol_name == "vertex":
        # Vertex 通过 URL 区分流式/非流式，body 不需要 stream 字段
        pass
    else:
        stream_body["stream"] = True
    if protocol_name == "openai":
        stream_body["stream_options"] = {"include_usage": True}
    return stream_body


config = load_config()


# ===== 生成可选参数测试用例 =====

def _optional_params():
    """生成 (protocol_name, model, option_name) 参数列表"""
    params = []
    for name, proto in config.protocols.items():
        if not proto.all_models:
            continue
        req_cfg = config.request_configs.get(name, {})
        optional = get_optional_fields(name, req_cfg)
        # 用第一个 non_thinking 模型测试（减少 API 调用）
        model = (proto.models_non_thinking or proto.models_thinking or [None])[0]
        if model is None:
            continue
        for opt in optional:
            if name == "anthropic" and opt == "cache_control":
                continue  # 由 test_anthropic_prompt_caching 专项测试覆盖
            params.append((name, model, opt))
    return params


OPTIONAL_PARAMS = _optional_params()


def _opt_id(param):
    name, model, opt = param
    return f"{name}-{model}-{opt}"


@pytest.mark.parametrize("protocol_name,model,option_name", OPTIONAL_PARAMS,
                         ids=[_opt_id(p) for p in OPTIONAL_PARAMS])
@pytest.mark.asyncio
async def test_optional_field(protocol_name, model, option_name, short_prompt):
    """测试每个非必须配置项是否可用"""
    builder = get_builder(protocol_name)
    client = get_client(config, protocol_name)
    body = builder.build_with_option(model, short_prompt, option_name)
    test_name = f"L2_optional_{protocol_name}_{model}_{option_name}"

    retry_decorator = make_retry_decorator(config.retry)

    @retry_decorator
    async def _do_request():
        status, text, data = await client.request(body, model=model)
        method, url, headers = client.get_request_info(body, model=model)

        if status != 200:
            log_failure(test_name, method, url, headers, body, status, text,
                        reason=f"HTTP {status}")
            raise RequestFailed(status, text, f"HTTP {status}")

        # 通用断言
        errors = builder.assert_non_stream_response(data)
        # 参数专属断言
        errors += builder.assert_option_response(option_name, data)

        if errors:
            log_failure(test_name, method, url, headers, body, status, text,
                        reason="; ".join(errors))
            raise RequestFailed(status, text, "; ".join(errors))

        log_success(test_name, method, url, headers, body, status, text)
        return data

    try:
        await _do_request()
    except RetryError as e:
        last = e.last_attempt.exception()
        if config.ai_validation.enabled and isinstance(last, RequestFailed):
            method, url, headers = client.get_request_info(body, model=model)
            curl_str, resp_str = get_curl_and_response(
                method, url, headers, body, last.status_code, last.body)
            expected, reason = await ai_validate(
                config.ai_validation, curl_str, resp_str)
            if expected:
                pytest.skip(f"UNSTABLE: AI says expected - {reason}")
        pytest.fail(f"Optional field '{option_name}' test failed: {last}")


# ===== 可选参数流式测试 =====

@pytest.mark.parametrize("protocol_name,model,option_name", OPTIONAL_PARAMS,
                         ids=[_opt_id(p) + "-stream" for p in OPTIONAL_PARAMS])
@pytest.mark.asyncio
async def test_optional_field_stream(protocol_name, model, option_name, short_prompt):
    """测试每个非必须配置项在流式模式下是否可用"""
    builder = get_builder(protocol_name)
    client = get_client(config, protocol_name)
    body = builder.build_with_option(model, short_prompt, option_name)
    body = _make_stream_body(builder, protocol_name, body)
    test_name = f"L2_optional_stream_{protocol_name}_{model}_{option_name}"

    retry_decorator = make_retry_decorator(config.retry)

    @retry_decorator
    async def _do_request():
        status, events, full_text = await client.request_stream(
            body, model=model)
        method, url, headers = client.get_request_info(
            body, model=model, stream=True)

        if status != 200:
            log_failure(test_name, method, url, headers, body, status, full_text,
                        reason=f"HTTP {status}")
            raise RequestFailed(status, full_text, f"HTTP {status}")

        errors = builder.assert_stream_events(events)

        if errors:
            log_failure(test_name, method, url, headers, body, status, full_text,
                        reason="; ".join(errors))
            raise RequestFailed(status, full_text, "; ".join(errors))

        log_success(test_name, method, url, headers, body, status, full_text)
        return events

    try:
        await _do_request()
    except RetryError as e:
        last = e.last_attempt.exception()
        if config.ai_validation.enabled and isinstance(last, RequestFailed):
            method, url, headers = client.get_request_info(
                body, model=model, stream=True)
            curl_str, resp_str = get_curl_and_response(
                method, url, headers, body, last.status_code, last.body)
            expected, reason = await ai_validate(
                config.ai_validation, curl_str, resp_str)
            if expected:
                pytest.skip(f"UNSTABLE: AI says expected - {reason}")
        pytest.fail(f"Optional field '{option_name}' stream test failed: {last}")


# ===== Anthropic Beta 头测试 =====

ANTHROPIC_BETAS = [
    "messages-2023-12-15",
    "prompt-caching-2024-07-31",
    "token-counting-2024-11-01",
    "max-tokens-3-5-sonnet-2024-07-15",
]


def _anthropic_beta_params():
    """生成 anthropic beta 头测试参数"""
    params = []
    proto = config.protocols.get("anthropic")
    if not proto or not proto.all_models:
        return params
    model = (proto.models_non_thinking or proto.models_thinking or [None])[0]
    if model is None:
        return params
    for beta in ANTHROPIC_BETAS:
        params.append(("anthropic", model, beta))
    return params


ANTHROPIC_BETA_PARAMS = _anthropic_beta_params()


@pytest.mark.parametrize("protocol_name,model,beta_header", ANTHROPIC_BETA_PARAMS,
                         ids=[f"anthropic-beta-{b}" for _, _, b in ANTHROPIC_BETA_PARAMS])
@pytest.mark.asyncio
async def test_anthropic_beta_headers(protocol_name, model, beta_header, short_prompt):
    """测试 Anthropic beta 头是否可用"""
    builder = AnthropicBuilder()
    client = get_client(config, "anthropic")
    body = builder.build_non_stream(model, short_prompt)
    extra_headers = {"anthropic-beta": beta_header}
    test_name = f"L2_beta_anthropic_{model}_{beta_header}"

    retry_decorator = make_retry_decorator(config.retry)

    @retry_decorator
    async def _do_request():
        status, text, data = await client.request(
            body, model=model, extra_headers=extra_headers)
        method, url, headers = client.get_request_info(
            body, model=model, extra_headers=extra_headers)

        if status != 200:
            log_failure(test_name, method, url, headers, body, status, text,
                        reason=f"HTTP {status}")
            raise RequestFailed(status, text, f"HTTP {status}")

        errors = builder.assert_non_stream_response(data)
        if errors:
            log_failure(test_name, method, url, headers, body, status, text,
                        reason="; ".join(errors))
            raise RequestFailed(status, text, "; ".join(errors))

        log_success(test_name, method, url, headers, body, status, text)
        return data

    try:
        await _do_request()
    except RetryError as e:
        last = e.last_attempt.exception()
        if config.ai_validation.enabled and isinstance(last, RequestFailed):
            method, url, headers = client.get_request_info(
                body, model=model, extra_headers=extra_headers)
            curl_str, resp_str = get_curl_and_response(
                method, url, headers, body, last.status_code, last.body)
            expected, reason = await ai_validate(
                config.ai_validation, curl_str, resp_str)
            if expected:
                pytest.skip(f"UNSTABLE: AI says expected - {reason}")
        pytest.fail(f"Beta header '{beta_header}' test failed: {last}")


# ===== Anthropic Beta 头流式测试 =====

@pytest.mark.parametrize("protocol_name,model,beta_header", ANTHROPIC_BETA_PARAMS,
                         ids=[f"anthropic-beta-stream-{b}" for _, _, b in ANTHROPIC_BETA_PARAMS])
@pytest.mark.asyncio
async def test_anthropic_beta_headers_stream(protocol_name, model, beta_header, short_prompt):
    """测试 Anthropic beta 头在流式模式下是否可用"""
    builder = AnthropicBuilder()
    client = get_client(config, "anthropic")
    body = builder.build_stream(model, short_prompt)
    extra_headers = {"anthropic-beta": beta_header}
    test_name = f"L2_beta_stream_anthropic_{model}_{beta_header}"

    retry_decorator = make_retry_decorator(config.retry)

    @retry_decorator
    async def _do_request():
        status, events, full_text = await client.request_stream(
            body, model=model, extra_headers=extra_headers)
        method, url, headers = client.get_request_info(
            body, model=model, stream=True, extra_headers=extra_headers)

        if status != 200:
            log_failure(test_name, method, url, headers, body, status, full_text,
                        reason=f"HTTP {status}")
            raise RequestFailed(status, full_text, f"HTTP {status}")

        errors = builder.assert_stream_events(events)
        if errors:
            log_failure(test_name, method, url, headers, body, status, full_text,
                        reason="; ".join(errors))
            raise RequestFailed(status, full_text, "; ".join(errors))

        log_success(test_name, method, url, headers, body, status, full_text)
        return events

    try:
        await _do_request()
    except RetryError as e:
        last = e.last_attempt.exception()
        if config.ai_validation.enabled and isinstance(last, RequestFailed):
            method, url, headers = client.get_request_info(
                body, model=model, stream=True, extra_headers=extra_headers)
            curl_str, resp_str = get_curl_and_response(
                method, url, headers, body, last.status_code, last.body)
            expected, reason = await ai_validate(
                config.ai_validation, curl_str, resp_str)
            if expected:
                pytest.skip(f"UNSTABLE: AI says expected - {reason}")
        pytest.fail(f"Beta header '{beta_header}' stream test failed: {last}")


# ===== 多轮对话测试 =====

MULTI_TURN_SCENARIOS = [
    (
        "context_name",
        "My name is Alice. Please confirm you remember my name.",
        "What is my name? Reply with just my name.",
        "alice",
    ),
    (
        "context_word",
        "My favorite fruit is banana. Please confirm you remember it.",
        "What is my favorite fruit? Reply with just the fruit name.",
        "banana",
    ),
]


def _multi_turn_params():
    """生成 (protocol_name, model, scenario_id, first_msg, followup_msg, keyword) 参数"""
    params = []
    for name, proto in config.protocols.items():
        if not proto.all_models:
            continue
        model = (proto.models_non_thinking or proto.models_thinking or [None])[0]
        if model is None:
            continue
        for scenario_id, first_msg, followup_msg, keyword in MULTI_TURN_SCENARIOS:
            params.append((name, model, scenario_id, first_msg, followup_msg, keyword))
    return params


MULTI_TURN_PARAMS = _multi_turn_params()


@pytest.mark.parametrize(
    "protocol_name,model,scenario_id,first_msg,followup_msg,expected_keyword",
    MULTI_TURN_PARAMS,
    ids=[f"{n}-{m}-{s}" for n, m, s, *_ in MULTI_TURN_PARAMS],
)
@pytest.mark.asyncio
async def test_multi_turn_conversation(
    protocol_name, model, scenario_id, first_msg, followup_msg, expected_keyword,
):
    """多轮对话测试 - 验证模型能正确利用上下文

    测试逻辑:
    1. 第一次请求: 发送包含特定信息的消息 → 获取助手回复
    2. 第二次请求: 携带完整对话历史 + 后续问题 → 验证回复包含该信息
    """
    builder = get_builder(protocol_name)
    client = get_client(config, protocol_name)
    test_name = f"L2_multi_turn_{protocol_name}_{model}_{scenario_id}"

    retry_decorator = make_retry_decorator(config.retry)
    phase_records: list[dict] = []
    last_body = None

    async def _single_request(body, phase: str):
        nonlocal last_body
        last_body = body

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

            if phase == "turn_2":
                response_text = builder.extract_text_content(data).lower()
                if expected_keyword not in response_text:
                    errors.append(
                        f"多轮对话上下文丢失: 回复中未包含 '{expected_keyword}'. "
                        f"response={response_text[:200]}")

            if errors:
                raise RequestFailed(status, text, "; ".join(errors))

            return data

        return await _do()

    try:
        # ---- 第 1 轮: 告知信息 ----
        body_1 = builder.build_non_stream(model, first_msg)
        data_1 = await _single_request(body_1, "turn_1")

        assistant_text = builder.extract_text_content(data_1)
        if not assistant_text:
            raise RequestFailed(200, "", "无法从第一轮响应中提取文本内容")

        # ---- 第 2 轮: 携带历史上下文询问 ----
        turns = [
            ("user", first_msg),
            ("assistant", assistant_text),
            ("user", followup_msg),
        ]
        body_2 = builder.build_multi_turn(model, turns)
        await _single_request(body_2, "turn_2")

        log_multi_phase(test_name, phase_records, success=True)

    except (RetryError, RequestFailed) as e:
        last = e.last_attempt.exception() if isinstance(e, RetryError) else e
        reason = str(last)
        log_multi_phase(test_name, phase_records, success=False, reason=reason)

        if config.ai_validation.enabled and isinstance(last, RequestFailed):
            method, url, headers = client.get_request_info(
                last_body or body_1, model=model)
            curl_str, resp_str = get_curl_and_response(
                method, url, headers, last_body or body_1,
                last.status_code, last.body)
            expected, ai_reason = await ai_validate(
                config.ai_validation, curl_str, resp_str)
            if expected:
                pytest.skip(f"UNSTABLE: AI says expected - {ai_reason}")
        pytest.fail(f"Multi-turn conversation test failed: {last}")


# ===== 多轮对话流式测试 =====

@pytest.mark.parametrize(
    "protocol_name,model,scenario_id,first_msg,followup_msg,expected_keyword",
    MULTI_TURN_PARAMS,
    ids=[f"{n}-{m}-{s}-stream" for n, m, s, *_ in MULTI_TURN_PARAMS],
)
@pytest.mark.asyncio
async def test_multi_turn_conversation_stream(
    protocol_name, model, scenario_id, first_msg, followup_msg, expected_keyword,
):
    """多轮对话流式测试 - 第一轮非流式获取上下文，第二轮流式验证

    测试逻辑:
    1. 第一次请求 (非流式): 发送包含特定信息的消息 → 获取助手回复
    2. 第二次请求 (流式): 携带完整对话历史 + 后续问题 → 验证流式回复包含该信息
    """
    builder = get_builder(protocol_name)
    client = get_client(config, protocol_name)
    test_name = f"L2_multi_turn_stream_{protocol_name}_{model}_{scenario_id}"

    retry_decorator = make_retry_decorator(config.retry)
    phase_records: list[dict] = []
    last_body = None

    async def _non_stream_request(body, phase: str):
        """第一轮: 非流式请求获取助手回复"""
        nonlocal last_body
        last_body = body

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
            if errors:
                raise RequestFailed(status, text, "; ".join(errors))

            return data

        return await _do()

    async def _stream_request(body, phase: str):
        """第二轮: 流式请求验证上下文"""
        nonlocal last_body
        last_body = body

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

            # 从流式事件中提取文本并验证关键词
            response_text = _extract_text_from_stream_events(
                events, protocol_name).lower()
            if expected_keyword not in response_text:
                errors.append(
                    f"多轮对话上下文丢失 (流式): 回复中未包含 '{expected_keyword}'. "
                    f"response={response_text[:200]}")

            if errors:
                raise RequestFailed(status, full_text, "; ".join(errors))

            return events

        return await _do()

    try:
        # ---- 第 1 轮: 告知信息 (非流式) ----
        body_1 = builder.build_non_stream(model, first_msg)
        data_1 = await _non_stream_request(body_1, "turn_1")

        assistant_text = builder.extract_text_content(data_1)
        if not assistant_text:
            raise RequestFailed(200, "", "无法从第一轮响应中提取文本内容")

        # ---- 第 2 轮: 携带历史上下文询问 (流式) ----
        turns = [
            ("user", first_msg),
            ("assistant", assistant_text),
            ("user", followup_msg),
        ]
        body_2 = builder.build_multi_turn(model, turns)
        body_2 = _make_stream_body(builder, protocol_name, body_2)
        await _stream_request(body_2, "turn_2_stream")

        log_multi_phase(test_name, phase_records, success=True)

    except (RetryError, RequestFailed) as e:
        last = e.last_attempt.exception() if isinstance(e, RetryError) else e
        reason = str(last)
        log_multi_phase(test_name, phase_records, success=False, reason=reason)

        if config.ai_validation.enabled and isinstance(last, RequestFailed):
            method, url, headers = client.get_request_info(
                last_body or body_1, model=model, stream=True)
            curl_str, resp_str = get_curl_and_response(
                method, url, headers, last_body or body_1,
                last.status_code, last.body)
            expected, ai_reason = await ai_validate(
                config.ai_validation, curl_str, resp_str)
            if expected:
                pytest.skip(f"UNSTABLE: AI says expected - {ai_reason}")
        pytest.fail(f"Multi-turn conversation stream test failed: {last}")
