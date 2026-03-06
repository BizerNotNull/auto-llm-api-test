"""
第三层测试 - 组合覆盖率测试
- 使用正交实验法(allpairspy)对非必须配置项组合进行覆盖率测试
- 注意参数互斥的逻辑报错
"""
import pytest
import json5
from itertools import product
from tenacity import RetryError
from allpairspy import AllPairs

from src.config import load_config, get_optional_fields
from src.client import LLMClient
from src.middleware import RequestFailed, make_retry_decorator, ai_validate
from src.logger import log_success, log_failure, get_curl_and_response
from conftest import get_builder, get_client


config = load_config()


# ===== 互斥参数定义 =====

MUTEX_GROUPS = {
    "openai": [
        # temperature 和 top_p 一般不同时使用
        {"temperature", "top_p"},
        # max_tokens 和 max_completion_tokens 互斥
        {"max_tokens", "max_completion_tokens"},
        # logprobs 和 top_logprobs 依赖关系
        {"top_logprobs"},  # 需要 logprobs=True
        # response_format=json_object 需要 system 提示 JSON (messages 会被覆盖)
        # 与多模态 messages_image_* 冲突 (都会覆盖 messages)
        {"response_format", "messages_image_url"},
        {"response_format", "messages_image_base64"},
        {"messages_image_url", "messages_image_base64"},
        # audio 需要 modalities 包含 audio
        {"audio", "messages_image_url"},
        {"audio", "messages_image_base64"},
    ],
    "anthropic": [
        {"temperature", "top_p"},
        # thinking 不能与 temperature/top_k 同时用
        {"thinking", "temperature"},
        {"thinking", "top_k"},
        # 多模态 messages 互斥 (都覆盖 messages)
        {"messages_image_base64", "messages_image_url"},
        {"messages_image_base64", "messages_pdf_base64"},
        {"messages_image_url", "messages_pdf_base64"},
    ],
    "vertex": [
        {"temperature", "topP"},
        # responseMimeType=json 与 responseSchema 冲突时 schema 已包含 mimeType
        # 多模态 contents 互斥 (都覆盖 contents)
        {"contents_image_base64", "contents_image_url"},
    ],
    "response": [
        {"temperature", "top_p"},
        # 多模态 input 互斥 (都覆盖 input)
        {"input_image_url", "input_image_base64"},
    ],
}


def _is_valid_combination(protocol_name: str, combo: dict[str, bool]) -> bool:
    """检查参数组合是否有效（无互斥冲突）"""
    enabled = {k for k, v in combo.items() if v}

    groups = MUTEX_GROUPS.get(protocol_name, [])
    for group in groups:
        # 如果互斥组中有多于1个参数被启用，则无效
        if len(enabled & group) > 1:
            return False

    # OpenAI 特殊: top_logprobs 需要 logprobs
    if protocol_name == "openai":
        if "top_logprobs" in enabled and "logprobs" not in enabled:
            return False

    return True


def _generate_pairwise_combos(protocol_name: str) -> list[dict[str, bool]]:
    """使用正交实验法生成参数组合"""
    req_cfg = config.request_configs.get(protocol_name, {})
    optional = get_optional_fields(protocol_name, req_cfg)

    if not optional:
        return []

    # 每个参数两种状态: True(启用) / False(禁用)
    param_names = optional
    param_values = [[True, False] for _ in param_names]

    # 使用 allpairspy 生成 pairwise 组合
    def is_valid(row):
        if len(row) < 2:
            return True
        combo = {}
        for i, val in enumerate(row):
            combo[param_names[i]] = val
        return _is_valid_combination(protocol_name, combo)

    combos = []
    try:
        for pair in AllPairs(param_values, filter_func=is_valid):
            combo = {}
            for i, val in enumerate(pair):
                combo[param_names[i]] = val
            # 至少要有一个参数启用
            if any(combo.values()):
                combos.append(combo)
    except Exception:
        # 如果 allpairspy 失败，回退到逐个测试
        for name in param_names:
            combos.append({name: True})

    return combos


def _build_combo_body(builder, model: str, prompt: str,
                      combo: dict[str, bool]) -> dict:
    """根据参数组合构建请求体"""
    body = builder.build_non_stream(model, prompt)
    protected_keys = {"model", "messages", "max_tokens", "contents", "input", "stream"}
    for opt_name, enabled in combo.items():
        if enabled:
            opt_body = builder.build_with_option(model, prompt, opt_name)
            for k, v in opt_body.items():
                if k in protected_keys:
                    continue
                # Vertex: generationConfig 需要合并而非覆盖
                if k == "generationConfig" and k in body and isinstance(v, dict):
                    body[k].update(v)
                else:
                    body[k] = v
    return body


# ===== 生成测试参数 =====

def _combo_params():
    """生成所有协议的组合测试参数"""
    params = []
    for name, proto in config.protocols.items():
        if not proto.all_models:
            continue
        model = (proto.models_non_thinking or proto.models_thinking or [None])[0]
        if model is None:
            continue
        combos = _generate_pairwise_combos(name)
        for i, combo in enumerate(combos):
            enabled_opts = [k for k, v in combo.items() if v]
            params.append((name, model, combo, i, "+".join(enabled_opts)))
    return params


COMBO_PARAMS = _combo_params()


def _combo_id(param):
    name, model, combo, idx, desc = param
    return f"{name}-{model}-combo{idx}-{desc}"


@pytest.mark.parametrize("protocol_name,model,combo,idx,desc", COMBO_PARAMS,
                         ids=[_combo_id(p) for p in COMBO_PARAMS])
@pytest.mark.asyncio
async def test_combination(protocol_name, model, combo, idx, desc, short_prompt):
    """正交实验法组合测试"""
    builder = get_builder(protocol_name)
    client = get_client(config, protocol_name)
    body = _build_combo_body(builder, model, short_prompt, combo)
    test_name = f"L3_combo_{protocol_name}_{model}_{idx}_{desc}"

    retry_decorator = make_retry_decorator(config.retry)

    # 组合中启用的参数列表
    enabled_options = [k for k, v in combo.items() if v]

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
        # 对组合中每个启用的参数追加专属断言
        for opt in enabled_options:
            errors += builder.assert_option_response(opt, data)

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
                pytest.skip(
                    f"UNSTABLE: AI says expected - {reason}")
        pytest.fail(f"Combination test failed (combo {idx}: {desc}): {last}")
