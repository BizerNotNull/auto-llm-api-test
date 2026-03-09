"""入口文件 - 使用 Rich 控制台运行 pytest"""
import os
import sys
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.config import load_config


console = Console()


def _export_curl(config, protocol_filter: str | None = None):
    """导出所有测试用例为 curl 命令"""
    from src.export import export_all_curls

    console.print()
    console.print(Panel.fit(
        "[bold cyan]LLM API Auto Test — Export Curl[/bold cyan]",
        subtitle="Generating curl commands without sending requests",
    ))
    console.print()

    out_path, total, sections = export_all_curls(
        config, protocol_filter=protocol_filter)

    # 显示导出统计
    for section_title, items in sections:
        console.print(f"  [bold]{section_title}[/bold]: {len(items)} curls")
    console.print()
    console.print(f"  [green]Total: {total} curl commands[/green]")
    console.print(f"  Output: [link=file://{out_path}]{out_path}[/link]")
    console.print()


def main():
    config = load_config()

    # 检查 --export-curl 标志
    argv = sys.argv[1:]
    if "--export-curl" in argv:
        argv_rest = [a for a in argv if a != "--export-curl"]
        # 支持 --protocol 过滤
        protocol_filter = None
        if "--protocol" in argv_rest:
            idx = argv_rest.index("--protocol")
            if idx + 1 < len(argv_rest):
                protocol_filter = argv_rest[idx + 1]
        _export_curl(config, protocol_filter=protocol_filter)
        return

    # 显示配置摘要
    console.print()
    console.print(Panel.fit(
        "[bold cyan]LLM API Auto Test[/bold cyan]",
        subtitle="Powered by pytest + httpx + rich",
    ))
    console.print()

    # 显示协议和模型
    for name, proto in config.protocols.items():
        if not proto.all_models:
            continue
        models_str = ", ".join(proto.all_models)
        console.print(f"  [bold]{name}[/bold]: {models_str}")
    console.print()

    # 显示测试层级
    for level, enabled in config.test_levels.items():
        status = "[green]ON[/green]" if enabled else "[dim]OFF[/dim]"
        console.print(f"  {level}: {status}")
    console.print()

    # 显示重试配置
    if config.retry.enabled:
        console.print(
            f"  Retry: [green]ON[/green] "
            f"(max={config.retry.max_attempts}, "
            f"backoff={config.retry.multiplier}x, "
            f"max_wait={config.retry.max_wait}s)")
    else:
        console.print("  Retry: [dim]OFF[/dim]")

    # 显示 AI 检验配置
    if config.ai_validation.enabled:
        console.print(
            f"  AI Validation: [green]ON[/green] ({config.ai_validation.model})")
    else:
        console.print("  AI Validation: [dim]OFF[/dim]")

    console.print()
    console.rule("[bold]Running Tests")
    console.print()

    # 运行 pytest (使用 UTF-8 避免 Windows GBK 编码问题)
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    args = [
        sys.executable, "-m", "pytest",
        str(root / "tests"),
        "--tb=short",
        "--no-header",
        "-n", "auto",
    ]

    # 将额外命令行参数传递给 pytest
    args.extend(sys.argv[1:])

    result = subprocess.run(args, cwd=str(root), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
