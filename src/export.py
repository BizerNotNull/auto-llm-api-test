"""导出所有测试用例为 curl 命令（不实际发送请求）"""
import datetime
from pathlib import Path

from allpairspy import AllPairs

from src.config import (
    Config, load_config, load_prompt, get_optional_fields, LOGS_DIR,
)
from src.client import LLMClient
from src.logger import format_curl
from src.protocols.openai import OpenAIBuilder
from src.protocols.anthropic import AnthropicBuilder
from src.protocols.vertex import VertexBuilder
from src.protocols.response import ResponseBuilder

BUILDERS = {
    "openai": OpenAIBuilder(),
    "anthropic": AnthropicBuilder(),
    "vertex": VertexBuilder(),
    "response": ResponseBuilder(),
}

# L3 互斥参数定义 (与 test_level3.py 保持一致)
MUTEX_GROUPS = {
    "openai": [
        {"temperature", "top_p"},
        {"max_tokens", "max_completion_tokens"},
        {"top_logprobs"},
        {"response_format", "messages_image_url"},
        {"response_format", "messages_image_base64"},
        {"messages_image_url", "messages_image_base64"},
        {"audio", "messages_image_url"},
        {"audio", "messages_image_base64"},
    ],
    "anthropic": [
        {"temperature", "top_p"},
        {"thinking", "temperature"},
        {"thinking", "top_k"},
        {"messages_image_base64", "messages_image_url"},
        {"messages_image_base64", "messages_pdf_base64"},
        {"messages_image_url", "messages_pdf_base64"},
    ],
    "vertex": [
        {"temperature", "topP"},
        {"contents_image_base64", "contents_image_url"},
    ],
    "response": [
        {"temperature", "top_p"},
        {"input_image_url", "input_image_base64"},
    ],
}

ANTHROPIC_BETAS = [
    "messages-2023-12-15",
    "prompt-caching-2024-07-31",
    "token-counting-2024-11-01",
    "max-tokens-3-5-sonnet-2024-07-15",
]


def _get_builder(name: str):
    return BUILDERS.get(name)


def _get_client(config: Config, name: str) -> LLMClient:
    return LLMClient(config.protocols[name])


def _make_curl(client: LLMClient, body: dict, model: str = "",
               stream: bool = False, extra_headers: dict | None = None) -> str:
    """构建一条 curl 命令（不隐藏 API key）"""
    method, url, headers = client.get_request_info(
        body, model=model, stream=stream, extra_headers=extra_headers)
    return format_curl(method, url, headers, body, redact=False)


# ─── Level 1 ────────────────────────────────────────────────────────

def _export_level1(config: Config, prompt: str) -> list[tuple[str, str]]:
    """导出 L1 测试用例，返回 [(test_name, curl_str), ...]"""
    results = []
    for name, proto in config.protocols.items():
        if not proto.all_models:
            continue
        builder = _get_builder(name)
        client = _get_client(config, name)

        for model in proto.all_models:
            # 1) connectivity - minimal
            body = builder.build_minimal(model, prompt)
            curl = _make_curl(client, body, model=model)
            results.append((f"L1_connectivity_{name}_{model}", curl))

            # 2) non-stream usage
            body = builder.build_non_stream(model, prompt)
            curl = _make_curl(client, body, model=model)
            results.append((f"L1_non_stream_usage_{name}_{model}", curl))

            # 3) stream
            body = builder.build_stream(model, prompt)
            curl = _make_curl(client, body, model=model, stream=True)
            results.append((f"L1_stream_{name}_{model}", curl))

    return results


# ─── Level 2 ────────────────────────────────────────────────────────

def _export_level2(config: Config, short_prompt: str,
                   long_prompt: str) -> list[tuple[str, str]]:
    """导出 L2 测试用例"""
    results = []

    for name, proto in config.protocols.items():
        if not proto.all_models:
            continue
        builder = _get_builder(name)
        client = _get_client(config, name)
        req_cfg = config.request_configs.get(name, {})
        optional = get_optional_fields(name, req_cfg)
        model = (proto.models_non_thinking or proto.models_thinking or [None])[0]
        if model is None:
            continue

        # 可选参数逐个测试
        for opt in optional:
            body = builder.build_with_option(model, short_prompt, opt)
            curl = _make_curl(client, body, model=model)
            results.append((f"L2_optional_{name}_{model}_{opt}", curl))

    # Anthropic 缓存测试
    anthropic_proto = config.protocols.get("anthropic")
    if anthropic_proto and anthropic_proto.all_models:
        req_cfg = config.request_configs.get("anthropic", {})
        if req_cfg.get("cache_control", False):
            anthropic_builder = AnthropicBuilder()
            client = _get_client(config, "anthropic")
            for model in anthropic_proto.models_non_thinking:
                body = anthropic_builder.build_cache_test(
                    model, long_prompt, session_id="export_test")
                curl = _make_curl(client, body, model=model)
                results.append((f"L2_cache_anthropic_{model}", curl))

    # Anthropic beta 头测试
    if anthropic_proto and anthropic_proto.all_models:
        anthropic_builder = AnthropicBuilder()
        client = _get_client(config, "anthropic")
        model = (anthropic_proto.models_non_thinking
                 or anthropic_proto.models_thinking or [None])[0]
        if model:
            for beta in ANTHROPIC_BETAS:
                body = anthropic_builder.build_non_stream(model, short_prompt)
                extra_headers = {"anthropic-beta": beta}
                curl = _make_curl(client, body, model=model,
                                  extra_headers=extra_headers)
                results.append((f"L2_beta_anthropic_{model}_{beta}", curl))

    return results


# ─── Level 3 ────────────────────────────────────────────────────────

def _is_valid_combination(protocol_name: str, combo: dict[str, bool]) -> bool:
    enabled = {k for k, v in combo.items() if v}
    groups = MUTEX_GROUPS.get(protocol_name, [])
    for group in groups:
        if len(enabled & group) > 1:
            return False
    if protocol_name == "openai":
        if "top_logprobs" in enabled and "logprobs" not in enabled:
            return False
    return True


def _build_combo_body(builder, model: str, prompt: str,
                      combo: dict[str, bool]) -> dict:
    body = builder.build_non_stream(model, prompt)
    protected_keys = {"model", "messages", "max_tokens", "contents", "input", "stream"}
    for opt_name, enabled in combo.items():
        if enabled:
            opt_body = builder.build_with_option(model, prompt, opt_name)
            for k, v in opt_body.items():
                if k in protected_keys:
                    continue
                if k == "generationConfig" and k in body and isinstance(v, dict):
                    body[k].update(v)
                else:
                    body[k] = v
    return body


def _export_level3(config: Config, prompt: str) -> list[tuple[str, str]]:
    """导出 L3 测试用例"""
    results = []
    for name, proto in config.protocols.items():
        if not proto.all_models:
            continue
        builder = _get_builder(name)
        client = _get_client(config, name)
        model = (proto.models_non_thinking or proto.models_thinking or [None])[0]
        if model is None:
            continue

        req_cfg = config.request_configs.get(name, {})
        optional = get_optional_fields(name, req_cfg)
        if not optional:
            continue

        param_values = [[True, False] for _ in optional]

        def is_valid(row, _name=name, _optional=optional):
            if len(row) < 2:
                return True
            combo = {}
            for i, val in enumerate(row):
                combo[_optional[i]] = val
            return _is_valid_combination(_name, combo)

        combos = []
        try:
            for pair in AllPairs(param_values, filter_func=is_valid):
                combo = {}
                for i, val in enumerate(pair):
                    combo[optional[i]] = val
                if any(combo.values()):
                    combos.append(combo)
        except Exception:
            for opt_name in optional:
                combos.append({opt_name: True})

        for idx, combo in enumerate(combos):
            enabled_opts = [k for k, v in combo.items() if v]
            desc = "+".join(enabled_opts)
            body = _build_combo_body(builder, model, prompt, combo)
            curl = _make_curl(client, body, model=model)
            results.append((f"L3_combo_{name}_{model}_{idx}_{desc}", curl))

    return results


# ─── 主入口 ──────────────────────────────────────────────────────────

def export_all_curls(config: Config | None = None,
                     protocol_filter: str | None = None) -> Path:
    """导出所有测试用例为 curl 命令文件

    Returns:
        输出文件路径
    """
    if config is None:
        config = load_config()

    short_prompt = load_prompt("short")
    long_prompt = load_prompt("long")

    # 如果指定了协议过滤，裁剪 config
    if protocol_filter:
        config.protocols = {
            k: v for k, v in config.protocols.items() if k == protocol_filter
        }

    sections: list[tuple[str, list[tuple[str, str]]]] = []

    if config.test_levels.get("level1", True):
        l1 = _export_level1(config, short_prompt)
        if l1:
            sections.append(("Level 1 - Basic Connectivity", l1))

    if config.test_levels.get("level2", False):
        l2 = _export_level2(config, short_prompt, long_prompt)
        if l2:
            sections.append(("Level 2 - Feature Availability", l2))

    if config.test_levels.get("level3", False):
        l3 = _export_level3(config, short_prompt)
        if l3:
            sections.append(("Level 3 - Combination Coverage", l3))

    # 写入文件
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = LOGS_DIR / f"curl_export_{ts}.sh"

    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(f"# Auto-generated curl export - {ts}\n")
        f.write(f"# Generated by: python run.py --export-curl\n\n")

        for section_title, items in sections:
            f.write(f"\n{'#' * 72}\n")
            f.write(f"# {section_title}  ({len(items)} requests)\n")
            f.write(f"{'#' * 72}\n\n")
            for test_name, curl_str in items:
                f.write(f"# {test_name}\n")
                f.write(f"{curl_str}\n\n")
                total += 1

    return out_path, total, sections
