"""pytest 根 conftest - 全局 fixtures 和配置"""
import pytest
import asyncio
from src.config import load_config, Config, load_prompt, get_optional_fields
from src.client import LLMClient
from src.protocols.openai import OpenAIBuilder
from src.protocols.anthropic import AnthropicBuilder
from src.protocols.vertex import VertexBuilder
from src.protocols.response import ResponseBuilder
from src.console import TestDisplay, console


BUILDERS = {
    "openai": OpenAIBuilder(),
    "anthropic": AnthropicBuilder(),
    "vertex": VertexBuilder(),
    "response": ResponseBuilder(),
}


def pytest_addoption(parser):
    parser.addoption("--protocol", action="store", default=None,
                     help="Run tests for a specific protocol only")


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def short_prompt() -> str:
    return load_prompt("short")


@pytest.fixture(scope="session")
def long_prompt() -> str:
    return load_prompt("long")


def _enabled_protocols(config: Config, request) -> list[str]:
    """获取需要测试的协议列表"""
    specific = request.config.getoption("--protocol")
    if specific:
        return [specific] if specific in config.protocols else []
    return [
        name for name, proto in config.protocols.items()
        if proto.all_models  # 只测有配置模型的协议
    ]


@pytest.fixture(scope="session")
def enabled_protocols(config, request) -> list[str]:
    return _enabled_protocols(config, request)


def get_builder(protocol_name: str):
    return BUILDERS.get(protocol_name)


def get_client(config: Config, protocol_name: str) -> LLMClient:
    return LLMClient(config.protocols[protocol_name])


# ===== pytest 收集阶段: 根据配置跳过未开启的测试层 =====

def pytest_collection_modifyitems(config, items):
    """根据 config.yaml 中的 test_levels 跳过未开启的测试层"""
    try:
        cfg = load_config()
    except Exception:
        return

    level_map = {
        "test_level1": cfg.test_levels.get("level1", True),
        "test_level2": cfg.test_levels.get("level2", False),
        "test_level3": cfg.test_levels.get("level3", False),
    }

    for item in items:
        # 根据测试文件名判断层级
        for level_key, enabled in level_map.items():
            if level_key in item.nodeid and not enabled:
                item.add_marker(pytest.mark.skip(
                    reason=f"{level_key} is disabled in config.yaml"
                ))
