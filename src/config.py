"""配置加载模块"""
import json5
import yaml
from pathlib import Path
from dataclasses import dataclass, field


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
PROTOCOLS_DIR = ROOT_DIR / "protocols"
PROMPTS_DIR = ROOT_DIR / "prompts"
LOGS_DIR = ROOT_DIR / "logs"


@dataclass
class ProtocolConfig:
    name: str
    base_url: str
    api_key: str
    models_thinking: list[str] = field(default_factory=list)
    models_non_thinking: list[str] = field(default_factory=list)
    auth_header: str = "authorization"  # anthropic 专用

    @property
    def all_models(self) -> list[str]:
        seen = set()
        result = []
        for m in self.models_thinking + self.models_non_thinking:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result


@dataclass
class RetryConfig:
    enabled: bool = True
    max_attempts: int = 3
    multiplier: float = 1.0
    max_wait: float = 30.0


@dataclass
class AIValidationConfig:
    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass
class Config:
    protocols: dict[str, ProtocolConfig] = field(default_factory=dict)
    test_levels: dict[str, bool] = field(default_factory=dict)
    retry: RetryConfig = field(default_factory=RetryConfig)
    ai_validation: AIValidationConfig = field(default_factory=AIValidationConfig)

    # 协议的请求体配置(哪些字段需要测试)
    request_configs: dict[str, dict[str, bool]] = field(default_factory=dict)
    # 协议的响应体模板
    response_templates: dict[str, dict] = field(default_factory=dict)


def load_config() -> Config:
    """加载主配置和协议配置"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = Config()

    # 解析协议配置
    for name, proto_raw in raw.get("protocols", {}).items():
        models = proto_raw.get("models", {})
        config.protocols[name] = ProtocolConfig(
            name=name,
            base_url=proto_raw.get("base_url", ""),
            api_key=proto_raw.get("api_key", ""),
            models_thinking=models.get("thinking", []) or [],
            models_non_thinking=models.get("non_thinking", []) or [],
            auth_header=proto_raw.get("auth_header", "authorization"),
        )

    # 测试层级
    config.test_levels = raw.get("test_levels", {
        "level1": True, "level2": False, "level3": False
    })

    # 重试配置
    retry_raw = raw.get("retry", {})
    config.retry = RetryConfig(
        enabled=retry_raw.get("enabled", True),
        max_attempts=retry_raw.get("max_attempts", 3),
        multiplier=retry_raw.get("multiplier", 1.0),
        max_wait=retry_raw.get("max_wait", 30.0),
    )

    # AI 检验配置
    ai_raw = raw.get("ai_validation", {})
    config.ai_validation = AIValidationConfig(
        enabled=ai_raw.get("enabled", False),
        base_url=ai_raw.get("base_url", ""),
        api_key=ai_raw.get("api_key", ""),
        model=ai_raw.get("model", ""),
    )

    # 加载各协议的请求体配置 (jsonc) 和响应体模板
    for name in config.protocols:
        req_path = PROTOCOLS_DIR / f"{name}_request.jsonc"
        resp_path = PROTOCOLS_DIR / f"{name}_response.json"

        if req_path.exists():
            with open(req_path, "r", encoding="utf-8") as f:
                config.request_configs[name] = json5.load(f)

        if resp_path.exists():
            with open(resp_path, "r", encoding="utf-8") as f:
                config.response_templates[name] = json5.load(f)

    return config


def get_required_fields(request_config: dict[str, bool]) -> list[str]:
    """从 jsonc 配置注释中提取必须字段（通过硬编码映射）"""
    # 各协议的必须字段
    REQUIRED = {
        "openai": ["model", "messages"],
        "anthropic": ["model", "messages", "max_tokens"],
        "vertex": ["model", "contents"],
        "response": ["model", "input"],
    }
    return REQUIRED


def get_optional_fields(protocol_name: str, request_config: dict[str, bool]) -> list[str]:
    """获取配置中需要测试的非必须字段"""
    required_map = get_required_fields(request_config)
    required = set(required_map.get(protocol_name, []))
    return [k for k, v in request_config.items() if v and k not in required]


def load_prompt(name: str) -> str:
    """加载提示词文件"""
    path = PROMPTS_DIR / name
    if not path.exists():
        # 尝试加 .txt
        path = PROMPTS_DIR / f"{name}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
