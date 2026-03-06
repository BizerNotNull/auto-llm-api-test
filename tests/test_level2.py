"""
第二层测试 - 功能可用性测试
- 每个非必须项是否可用
- anthropic 提示词缓存测试
- anthropic beta 头测试
"""
import pytest
import json5
from tenacity import RetryError

from src.config import load_config, get_optional_fields
from src.client import LLMClient
from src.middleware import RequestFailed, make_retry_decorator, ai_validate
from src.logger import log_success, log_failure, get_curl_and_response
from src.protocols.anthropic import AnthropicBuilder
from conftest import get_builder, get_client


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


# ===== Anthropic 提示词缓存测试 =====

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


@pytest.mark.parametrize("protocol_name,model", ANTHROPIC_CACHE_PARAMS,
                         ids=[f"anthropic-cache-{m}" for _, m in ANTHROPIC_CACHE_PARAMS])
@pytest.mark.asyncio
async def test_anthropic_prompt_caching(protocol_name, model, long_prompt):
    """Anthropic 提示词缓存测试 - 需要足够 token 量 + 随机前缀"""
    builder = AnthropicBuilder()
    client = get_client(config, "anthropic")
    body = builder.build_cache_test(model, long_prompt)
    test_name = f"L2_cache_anthropic_{model}"

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

        # 检查 usage 中的缓存相关字段
        usage = builder.extract_usage(data)
        if usage:
            # 第一次请求应该有 cache_creation_input_tokens
            cache_creation = usage.get("cache_creation_input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            if cache_creation == 0 and cache_read == 0:
                errors.append(
                    "No cache tokens in usage "
                    "(cache_creation_input_tokens=0, cache_read_input_tokens=0)")

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
        pytest.fail(f"Anthropic cache test failed: {last}")


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
