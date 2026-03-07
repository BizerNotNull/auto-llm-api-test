"""Rich 控制台输出模块"""
import io
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.table import Table
from rich.text import Text


# Windows GBK 终端无法显示 Rich 的 Unicode spinner，强制使用 UTF-8
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()

# 测试状态颜色
STATUS_COLORS = {
    "PASS": "green",
    "FAIL": "red",
    "UNSTABLE": "yellow",
    "RUNNING": "cyan",
    "SKIP": "dim",
}


class TestDisplay:
    """测试显示管理器 - 进度条 + 最近5条测试结果"""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.recent_results: list[tuple[str, str]] = []  # (test_name, status)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        )
        self.task_id = self.progress.add_task("Testing", total=total)

    def update(self, test_name: str, status: str):
        """更新测试结果"""
        self.completed += 1
        self.progress.update(self.task_id, advance=1)
        self.recent_results.append((test_name, status))
        # 只保留最近5条
        if len(self.recent_results) > 5:
            self.recent_results = self.recent_results[-5:]

    def print_recent(self):
        """打印最近5条测试结果"""
        if not self.recent_results:
            return
        for name, status in self.recent_results:
            color = STATUS_COLORS.get(status, "white")
            console.print(f"  [{color}]{status:>8}[/{color}]  {name}")

    def start(self):
        self.progress.start()

    def stop(self):
        self.progress.stop()


def print_summary(passed: int, failed: int, unstable: int, skipped: int):
    """打印测试汇总"""
    console.print()
    console.rule("[bold]Test Summary")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[green]PASSED[/green]", str(passed))
    table.add_row("[red]FAILED[/red]", str(failed))
    table.add_row("[yellow]UNSTABLE[/yellow]", str(unstable))
    table.add_row("[dim]SKIPPED[/dim]", str(skipped))
    total = passed + failed + unstable + skipped
    table.add_row("[bold]TOTAL[/bold]", str(total))
    console.print(table)
    console.print()
