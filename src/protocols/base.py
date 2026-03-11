"""协议基类 - 定义各协议的请求构建和响应断言接口"""
from abc import ABC, abstractmethod


class ProtocolBuilder(ABC):
    """协议请求构建器基类"""

    @abstractmethod
    def build_minimal(self, model: str, prompt: str) -> dict:
        """构建最小可用请求体（仅必须项）"""
        ...

    @abstractmethod
    def build_non_stream(self, model: str, prompt: str, **kwargs) -> dict:
        """构建非流式请求体"""
        ...

    @abstractmethod
    def build_stream(self, model: str, prompt: str, **kwargs) -> dict:
        """构建流式请求体"""
        ...

    @abstractmethod
    def build_with_option(self, model: str, prompt: str,
                          option_name: str, **kwargs) -> dict:
        """构建带某个可选参数的请求体"""
        ...

    @abstractmethod
    def assert_non_stream_response(self, data: dict) -> list[str]:
        """断言非流式响应，返回失败原因列表（空=通过）"""
        ...

    @abstractmethod
    def assert_stream_events(self, events: list[str]) -> list[str]:
        """断言流式事件，返回失败原因列表"""
        ...

    @abstractmethod
    def extract_usage(self, data: dict) -> dict | None:
        """从非流式响应中提取 usage"""
        ...

    @abstractmethod
    def extract_stream_usage(self, events: list[str]) -> dict | None:
        """从流式事件中提取 usage"""
        ...

    def assert_option_response(self, option_name: str, data: dict) -> list[str]:
        """
        针对特定参数的专属断言。在通用断言之后调用。
        返回失败原因列表（空=通过）。

        默认实现：无额外检查。子类通过 option_assertions 字典注册。
        """
        return []

    @abstractmethod
    def extract_text_content(self, data: dict) -> str:
        """从非流式响应中提取纯文本内容"""
        ...

    @abstractmethod
    def build_multi_turn(self, model: str, turns: list[tuple[str, str]],
                         **kwargs) -> dict:
        """构建多轮对话请求体

        Args:
            turns: [(role, content), ...] - role 使用标准 user/assistant
        """
        ...
