"""pytest 根 conftest - 全局 fixtures, 配置, Rich reporter plugin"""
import datetime
import io
import os
import threading
import pytest
from src.config import load_config, Config, load_prompt, get_optional_fields
from src.client import LLMClient
from src.protocols.openai import OpenAIBuilder
from src.protocols.anthropic import AnthropicBuilder
from src.protocols.vertex import VertexBuilder
from src.protocols.response import ResponseBuilder
from src.console import TestDisplay, print_summary, STATUS_COLORS, console

# Ensure session timestamp is set early (before logger.py is imported)
if not os.environ.get("LLMTEST_SESSION_TS"):
    os.environ["LLMTEST_SESSION_TS"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.markup import escape as rich_escape


# =====================================================================
# Rich Reporter Plugin - replaces default terminal reporter
# =====================================================================

class StubTerminalReporter:
    """Minimal stub so xdist doesn't crash looking for 'terminalreporter'."""

    def __init__(self, config):
        self.config = config
        self._tw = io.StringIO()
        self.stats = {}
        self.hasmarkup = False
        self.isatty = False

    # no-op methods that xdist / other plugins may call
    def write_line(self, *args, **kwargs): pass
    def rewrite(self, *args, **kwargs): pass
    def ensure_newline(self, *args, **kwargs): pass
    def section(self, *args, **kwargs): pass
    def write_sep(self, *args, **kwargs): pass
    def write(self, *args, **kwargs): pass
    def line(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass


class RichReporter:
    """Rich-based test reporter with progress bar + recent results."""

    def __init__(self, config):
        self._cfg = config
        self._lock = threading.Lock()
        self._display = None
        self._live = None
        self._total = 0
        self._passed = 0
        self._failed = 0
        self._skipped = 0
        self._unstable = 0
        self._failures: list[tuple[str, str]] = []  # (nodeid, reason)
        self._show_failure_details = True
        self._finished = False

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _shorten_nodeid(nodeid: str) -> str:
        """Strip file path prefix, keep test name + params."""
        if "::" in nodeid:
            return nodeid.split("::", 1)[1]
        return nodeid

    def _build_live_table(self) -> Table:
        """Build a Rich renderable combining progress bar + recent results."""
        table = Table.grid(padding=(0, 0))
        table.add_row(self._display.progress)

        if self._display.recent_results:
            results_table = Table(show_header=False, box=None, padding=(0, 1))
            for name, status in self._display.recent_results:
                color = STATUS_COLORS.get(status, "white")
                results_table.add_row(
                    Text(f"  {status:>4}", style=color),
                    Text(f"  {name}"),
                )
            table.add_row(results_table)

        return table

    # -- pytest hooks -----------------------------------------------------

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        self._total = len(items)

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        """xdist controller receives total item count from workers."""
        if self._total == 0:
            self._total = len(ids)

    def pytest_sessionstart(self, session):
        try:
            app_cfg = load_config()
            self._show_failure_details = app_cfg.output.show_failure_details
        except Exception:
            pass

    def _ensure_display(self):
        """Lazily create display + live on first use (after collection is done)."""
        if self._display is not None:
            return
        self._display = TestDisplay(max(self._total, 1))
        # Don't call progress.start() — Progress inherits from Live, so
        # starting it would register a live display on the console and
        # conflict with the Live object below.
        self._live = Live(
            self._build_live_table(),
            console=console,
            transient=True,
            refresh_per_second=8,
        )
        self._live.start()

    def pytest_runtest_logreport(self, report):
        self._ensure_display()
        # Only process the "call" phase (or "setup" for errors)
        if report.when == "call":
            short_id = self._shorten_nodeid(report.nodeid)
            with self._lock:
                if report.passed:
                    self._passed += 1
                    self._display.update(short_id, "PASS")
                elif report.failed:
                    self._failed += 1
                    # Extract one-line failure reason
                    reason = self._extract_reason(report)
                    self._failures.append((short_id, reason))
                    self._display.update(short_id, "FAIL")
                elif report.skipped:
                    skip_reason = self._get_skip_reason(report)
                    if "UNSTABLE" in skip_reason.upper():
                        self._unstable += 1
                        self._display.update(short_id, "UNSTABLE")
                    else:
                        self._skipped += 1
                        self._display.update(short_id, "SKIP")
                if self._live:
                    self._live.update(self._build_live_table())

        elif report.when == "setup" and report.skipped:
            short_id = self._shorten_nodeid(report.nodeid)
            with self._lock:
                skip_reason = self._get_skip_reason(report)
                if "UNSTABLE" in skip_reason.upper():
                    self._unstable += 1
                    self._display.update(short_id, "UNSTABLE")
                else:
                    self._skipped += 1
                    self._display.update(short_id, "SKIP")
                if self._live:
                    self._live.update(self._build_live_table())

    def pytest_sessionfinish(self, session, exitstatus):
        if self._finished:
            return
        self._finished = True

        # Stop live display
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass
        # progress was never start()'ed separately — no need to stop it

        # Nothing to report if display was never started
        if self._display is None:
            return

        # Print failures list
        if self._failures and self._show_failure_details:
            console.print()
            console.rule("[bold red]Failures")
            for nodeid, reason in self._failures:
                console.print(f"  [red]FAIL[/red]  {rich_escape(nodeid)}")
                console.print(f"         [dim]{rich_escape(reason)}[/dim]")
            console.print()

        # Print summary table
        print_summary(self._passed, self._failed, self._unstable, self._skipped)

    # -- reason extraction ------------------------------------------------

    @staticmethod
    def _extract_reason(report) -> str:
        """Extract a one-line failure reason from report.longrepr."""
        try:
            if hasattr(report.longrepr, "reprcrash") and report.longrepr.reprcrash:
                return report.longrepr.reprcrash.message.split("\n")[0]
        except Exception:
            pass
        # Fallback: stringify and take last non-empty line
        try:
            lines = str(report.longrepr).strip().splitlines()
            for line in reversed(lines):
                stripped = line.strip()
                if stripped:
                    return stripped
        except Exception:
            pass
        return "Unknown error"

    @staticmethod
    def _get_skip_reason(report) -> str:
        """Extract skip reason string."""
        try:
            if isinstance(report.longrepr, tuple) and len(report.longrepr) >= 3:
                return str(report.longrepr[2])
        except Exception:
            pass
        return ""


# -- Module-level hook: register plugins ---------------------------------

def pytest_configure_node(node):
    """xdist controller -> worker: pass session timestamp."""
    node.workerinput["session_ts"] = os.environ["LLMTEST_SESSION_TS"]


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # xdist worker: inherit session timestamp from controller, then skip reporter setup
    if hasattr(config, "workerinput"):
        ts = config.workerinput.get("session_ts")
        if ts:
            os.environ["LLMTEST_SESSION_TS"] = ts
        return

    # Unregister default terminal reporter (registered by _pytest.terminal)
    standard_reporter = config.pluginmanager.getplugin("terminalreporter")
    if standard_reporter is not None:
        config.pluginmanager.unregister(standard_reporter, "terminalreporter")
        # Neutralize the old reporter so any stale references
        # (e.g. xdist DSession.terminal) don't leak output
        standard_reporter.write_line = lambda *a, **k: None
        standard_reporter.write = lambda *a, **k: None
        standard_reporter.rewrite = lambda *a, **k: None
        standard_reporter.line = lambda *a, **k: None

    # Register stub so xdist and other plugins can find "terminalreporter"
    stub = StubTerminalReporter(config)
    config.pluginmanager.register(stub, "terminalreporter")

    # Register our Rich reporter
    rich_reporter = RichReporter(config)
    config.pluginmanager.register(rich_reporter, "rich_reporter")


# =====================================================================
# Original conftest content - fixtures and configuration
# =====================================================================

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
    """根据 --protocol 过滤协议, 根据 config.yaml 跳过未开启的测试层"""
    try:
        cfg = load_config()
    except Exception:
        return

    # --protocol 过滤: deselect 不匹配的协议
    protocol = config.getoption("--protocol", default=None)
    if protocol:
        selected = []
        deselected = []
        for item in items:
            # 参数化 ID 格式: [protocol-model-...], 非参数化测试直接保留
            if f"[{protocol}-" in item.nodeid or "[" not in item.nodeid:
                selected.append(item)
            else:
                deselected.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = selected

    # 测试层级过滤
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
