"""
第一层测试 - 基础连通性测试
- 连通性
- 非流式 usage
- 流式事件返回程度和 usage
- 只发送最小可用请求（必须项）
"""
import pytest
import json5
from tenacity import RetryError

from src.config import load_config
from src.client import LLMClient
from src.middleware import RequestFailed, make_retry_decorator, ai_validate
from src.logger import log_success, log_failure, get_curl_and_response
from conftest import get_builder, get_client


config = load_config()


def _protocol_model_params():
    """生成 (protocol_name, model, model_type) 参数列表"""
    params = []
    for name, proto in config.protocols.items():
        if not proto.all_models:
            continue
        for model in proto.models_non_thinking:
            params.append((name, model, "non_thinking"))
        for model in proto.models_thinking:
            params.append((name, model, "thinking"))
    return params


PROTOCOL_MODEL_PARAMS = _protocol_model_params()


def _test_id(param):
    name, model, mtype = param
    return f"{name}-{model}-{mtype}"


# ===== 1. 连通性测试: 最小可用请求 =====

@pytest.mark.parametrize("protocol_name,model,model_type", PROTOCOL_MODEL_PARAMS,
                         ids=[_test_id(p) for p in PROTOCOL_MODEL_PARAMS])
@pytest.mark.asyncio
async def test_connectivity_minimal(protocol_name, model, model_type, short_prompt):
    """连通性测试 - 发送最小可用请求体"""
    builder = get_builder(protocol_name)
    client = get_client(config, protocol_name)
    body = builder.build_minimal(model, short_prompt)
    test_name = f"L1_connectivity_{protocol_name}_{model}"

    retry_decorator = make_retry_decorator(config.retry)

    @retry_decorator
    async def _do_request():
        status, text, data = await client.request(body, model=model)
        method, url, headers = client.get_request_info(body, model=model)

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
        data = await _do_request()
    except RetryError as e:
        last = e.last_attempt.exception()
        # AI 检验
        if config.ai_validation.enabled and isinstance(last, RequestFailed):
            method, url, headers = client.get_request_info(body, model=model)
            curl_str, resp_str = get_curl_and_response(
                method, url, headers, body, last.status_code, last.body)
            expected, reason = await ai_validate(
                config.ai_validation, curl_str, resp_str)
            if expected:
                pytest.skip(f"UNSTABLE: AI says expected - {reason}")
        pytest.fail(f"Connectivity failed after retries: {last}")


# ===== 2. 非流式 usage 测试 =====

@pytest.mark.parametrize("protocol_name,model,model_type", PROTOCOL_MODEL_PARAMS,
                         ids=[_test_id(p) for p in PROTOCOL_MODEL_PARAMS])
@pytest.mark.asyncio
async def test_non_stream_usage(protocol_name, model, model_type, short_prompt):
    """非流式请求 usage 测试"""
    builder = get_builder(protocol_name)
    client = get_client(config, protocol_name)
    body = builder.build_non_stream(model, short_prompt)
    test_name = f"L1_non_stream_usage_{protocol_name}_{model}"

    retry_decorator = make_retry_decorator(config.retry)

    @retry_decorator
    async def _do_request():
        status, text, data = await client.request(body, model=model)
        method, url, headers = client.get_request_info(body, model=model)

        if status != 200:
            log_failure(test_name, method, url, headers, body, status, text,
                        reason=f"HTTP {status}")
            raise RequestFailed(status, text, f"HTTP {status}")

        # 断言响应结构
        errors = builder.assert_non_stream_response(data)
        if errors:
            log_failure(test_name, method, url, headers, body, status, text,
                        reason="; ".join(errors))
            raise RequestFailed(status, text, "; ".join(errors))

        # 断言 usage
        usage = builder.extract_usage(data)
        if usage is None:
            log_failure(test_name, method, url, headers, body, status, text,
                        reason="Missing usage in response")
            raise RequestFailed(status, text, "Missing usage")

        log_success(test_name, method, url, headers, body, status, text)
        return usage

    try:
        usage = await _do_request()
        assert usage is not None, "Usage should not be None"
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
        pytest.fail(f"Non-stream usage test failed after retries: {last}")


# ===== 3. 流式事件 + usage 测试 =====

@pytest.mark.parametrize("protocol_name,model,model_type", PROTOCOL_MODEL_PARAMS,
                         ids=[_test_id(p) for p in PROTOCOL_MODEL_PARAMS])
@pytest.mark.asyncio
async def test_stream_events_and_usage(protocol_name, model, model_type, short_prompt):
    """流式请求事件返回程度和 usage 测试"""
    builder = get_builder(protocol_name)
    client = get_client(config, protocol_name)
    body = builder.build_stream(model, short_prompt)
    test_name = f"L1_stream_{protocol_name}_{model}"

    retry_decorator = make_retry_decorator(config.retry)

    @retry_decorator
    async def _do_request():
        status, events, full_text = await client.request_stream(
            body, model=model)
        method, url, headers = client.get_request_info(body, model=model, stream=True)

        if status != 200:
            log_failure(test_name, method, url, headers, body, status, full_text,
                        reason=f"HTTP {status}")
            raise RequestFailed(status, full_text, f"HTTP {status}")

        # 断言流式事件
        errors = builder.assert_stream_events(events)

        # 提取 usage
        usage = builder.extract_stream_usage(events)
        if usage is None:
            errors.append("Missing usage in stream events")

        if errors:
            log_failure(test_name, method, url, headers, body, status, full_text,
                        reason="; ".join(errors))
            raise RequestFailed(status, full_text, "; ".join(errors))

        log_success(test_name, method, url, headers, body, status, full_text)
        return events, usage

    try:
        events, usage = await _do_request()
        assert len(events) > 0, "Should receive stream events"
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
        pytest.fail(f"Stream test failed after retries: {last}")
