"""AgentBricks CLI - Interactive command-line interface for learning journey.

This CLI provides commands to start bricks, ask questions, check progress,
get hints, review code, and submit completed bricks.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from agent.brick_data import BRICK_NAMES, get_brick_info
from agent.hints import get_hint
from agent.progress import check_progress
from agent.rag import answer_question
from agent.review import review_code
from agent.submit import submit_brick

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="agentbricks")
def cli() -> None:
    """🧱 AgentBricks CLI - Your AI mentor for building ML systems."""
    pass


@cli.command()
@click.argument("brick_number", type=int)
def start_brick(brick_number: int) -> None:
    """Start a new brick - display story intro, objectives, and create feature branch.

    Args:
        brick_number: The brick number (1-6 for movie-recommender)
    """
    if brick_number < 1 or brick_number > 6:
        console.print(f"[red]❌ Invalid brick number: {brick_number}[/red]")
        console.print("[yellow]Available bricks: 1-6[/yellow]")
        sys.exit(1)

    brick_name = BRICK_NAMES.get(brick_number)
    if not brick_name:
        console.print(f"[red]❌ Brick {brick_number} not found[/red]")
        sys.exit(1)

    # Get brick information
    brick_info = get_brick_info(brick_number)

    # Display story intro
    console.print("\n")
    console.print(
        Panel.fit(
            f"[bold cyan]🧱 Brick {brick_number:02d}: {brick_info['title']}[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print("\n[bold]📖 Story Context:[/bold]\n")
    console.print(Markdown(brick_info.get("story_intro", "Welcome to this brick!")))

    # Display learning objectives
    console.print("\n[bold]🎯 Learning Objectives:[/bold]\n")
    objectives = brick_info.get("objectives", [])
    for i, obj in enumerate(objectives, 1):
        console.print(f"  {i}. {obj}")

    # Display tasks overview
    tasks = brick_info.get("tasks", [])
    if tasks:
        console.print(f"\n[bold]📋 Tasks ({len(tasks)} total):[/bold]\n")
        task_table = Table(show_header=True, header_style="bold magenta")
        task_table.add_column("Task", style="cyan", no_wrap=True)
        task_table.add_column("Description", style="white")

        for i, task in enumerate(tasks[:10], 1):  # Show first 10
            task_table.add_row(f"Task {i}", task.get("description", "")[:60] + "...")

        if len(tasks) > 10:
            task_table.add_row("...", f"... and {len(tasks) - 10} more tasks")

        console.print(task_table)

    # Create feature branch
    branch_name = f"feature/brick-{brick_number:02d}-{brick_name}"
    console.print(f"\n[bold]🌿 Creating feature branch:[/bold] [cyan]{branch_name}[/cyan]")

    try:
        # Check if we're in a git repo
        result = subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True, check=False)

        if result.returncode != 0:
            console.print("[yellow]⚠️  Not in a git repository. Skipping branch creation.[/yellow]")
        else:
            # Check if branch already exists
            result = subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
                capture_output=True,
                check=False,
            )

            if result.returncode == 0:
                msg = f"[yellow]⚠️  Branch {branch_name} already exists.[/yellow]"
                console.print(msg)
                switch_msg = "[yellow]Switch to existing branch?[/yellow]"
                if Confirm.ask(switch_msg):
                    subprocess.run(["git", "checkout", branch_name], check=True)
            else:
                # Create and checkout new branch
                subprocess.run(["git", "checkout", "-b", branch_name], check=True)
                msg = f"[green]✅ Created and switched to branch: {branch_name}[/green]"
                console.print(msg)
    except subprocess.CalledProcessError as e:
        error_msg = f"[red]❌ Error creating branch: {e}[/red]"
        console.print(error_msg)
    except FileNotFoundError:
        msg = "[yellow]⚠️  Git not found. Skipping branch creation.[/yellow]"
        console.print(msg)

    # Initialize brick directory if needed
    brick_path = Path(f"stories/movie-recommender/brick-{brick_number:02d}-{brick_name}")
    if not brick_path.exists():
        console.print(f"\n[bold]📁 Initializing brick directory:[/bold] [cyan]{brick_path}[/cyan]")
        brick_path.mkdir(parents=True, exist_ok=True)
        (brick_path / "src").mkdir(exist_ok=True)
        (brick_path / "tests").mkdir(exist_ok=True)

        # Create basic README if it doesn't exist
        readme_path = brick_path / "README.md"
        if not readme_path.exists():
            title = brick_info["title"]
            intro = brick_info.get("story_intro", "")
            readme_content = f"# Brick {brick_number:02d}: {title}\n\n{intro}\n"
            readme_path.write_text(readme_content)

        console.print("[green]✅ Brick directory initialized[/green]")
    else:
        msg = f"[green]✅ Brick directory already exists: {brick_path}[/green]"
        console.print(msg)

    # Show next steps
    console.print("\n[bold]💡 Next Steps:[/bold]")
    console.print("  1. Review the tasks above")
    console.print("  2. Start with Task 1")
    console.print("  3. Use [cyan]agentbricks hint 1[/cyan] for help")
    hint_cmd = "agentbricks check-progress"
    console.print(f"  4. Use [cyan]{hint_cmd}[/cyan] to track your progress")
    ask_cmd = "agentbricks ask <question>"
    console.print(f"  5. Use [cyan]{ask_cmd}[/cyan] for guidance\n")


@cli.command()
@click.argument("question", nargs=-1, required=True)
def ask(question: tuple) -> None:
    """Ask the AI mentor a question - RAG over documentation.

    Args:
        question: Your question (can be multiple words)
    """
    question_text = " ".join(question)

    console.print("\n[bold]🤔 Your Question:[/bold]")
    console.print(f"[cyan]{question_text}[/cyan]\n")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Searching documentation...", total=None)
        answer = answer_question(question_text)
        progress.update(task, completed=True)

    console.print("\n[bold]💡 AI Mentor Response:[/bold]\n")
    console.print(Panel(Markdown(answer), border_style="green"))
    console.print("\n")


@cli.command()
def check_progress_cmd() -> None:
    """Check your progress - analyze completed tasks and show recommendations."""
    console.print("\n[bold]📊 Progress Check[/bold]\n")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Analyzing your progress...", total=None)
        progress_data = check_progress()
        progress.update(task, completed=True)

    # Overall progress
    overall = progress_data.get("overall", {})
    completed_bricks = overall.get("completed_bricks", 0)
    total_bricks = overall.get("total_bricks", 6)
    progress_pct = (completed_bricks / total_bricks * 100) if total_bricks > 0 else 0

    console.print(
        f"\n[bold]Overall Progress:[/bold] {completed_bricks}/{total_bricks} bricks completed"
    )
    console.print(f"[bold]Progress:[/bold] {progress_pct:.1f}%")

    # Progress bar
    with Progress(
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), console=console
    ) as progress_bar:
        progress_bar.add_task("Overall Progress", total=100, completed=progress_pct)

    # Current brick progress
    current_brick = progress_data.get("current_brick")
    if current_brick:
        brick_num = current_brick.get("number")
        brick_name = current_brick.get("name")
        completed_tasks = current_brick.get("completed_tasks", 0)
        total_tasks = current_brick.get("total_tasks", 0)

        console.print(f"\n[bold]Current Brick:[/bold] Brick {brick_num:02d} - {brick_name}")
        console.print(f"[bold]Tasks:[/bold] {completed_tasks}/{total_tasks} completed")

        if total_tasks > 0:
            task_pct = completed_tasks / total_tasks * 100
            with Progress(
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress_bar:
                progress_bar.add_task("Brick Progress", total=100, completed=task_pct)

    # Next recommended tasks
    next_tasks = progress_data.get("next_tasks", [])
    if next_tasks:
        console.print("\n[bold]📋 Next Recommended Tasks:[/bold]\n")
        task_table = Table(show_header=True, header_style="bold magenta")
        task_table.add_column("Task", style="cyan", no_wrap=True)
        task_table.add_column("Description", style="white")
        task_table.add_column("Priority", style="yellow")

        for task_item in next_tasks[:5]:
            task_num = task_item.get("number", "?")
            task_desc = task_item.get("description", "")[:50] + "..."
            task_priority = task_item.get("priority", "Medium")
            task_table.add_row(
                f"Task {task_num}",
                task_desc,
                task_priority,
            )

        console.print(task_table)

    # Completed tasks summary
    completed = progress_data.get("completed_tasks_summary", [])
    if completed:
        console.print("\n[green]✅ Recently Completed:[/green]")
        for task in completed[-5:]:
            console.print(f"  • {task}")

    console.print("\n")


@cli.command()
@click.argument("task_number", type=int)
@click.option("--level", "-l", type=int, default=1, help="Hint level (1-3)")
def hint(task_number: int, level: int) -> None:
    """Get a hint for a specific task - graduated hints (gentle to pseudocode).

    Args:
        task_number: The task number
        level: Hint level (1=gentle nudge, 2=specific guidance, 3=pseudocode)
    """
    if level < 1 or level > 3:
        console.print("[red]❌ Hint level must be between 1 and 3[/red]")
        sys.exit(1)

    console.print(f"\n[bold]💡 Hint for Task {task_number} (Level {level})[/bold]\n")

    hint_text = get_hint(task_number, level)

    level_names = {1: "Gentle Nudge", 2: "Specific Guidance", 3: "Pseudocode"}

    console.print(
        Panel(
            Markdown(hint_text),
            title=f"Level {level}: {level_names.get(level, 'Hint')}",
            border_style="yellow",
        )
    )

    if level < 3:
        next_level = level + 1
        hint_cmd = f"agentbricks hint {task_number} --level {next_level}"
        msg = f"\n[yellow]💭 Need more help? Try: [cyan]{hint_cmd}[/cyan][/yellow]"
        console.print(msg)

    console.print("\n")


@cli.command()
@click.option("--path", "-p", default=".", help="Path to code to review")
def review(path: str) -> None:
    """Review your code - analyze against best practices and suggest improvements.

    Args:
        path: Path to the code directory or file to review
    """
    review_path = Path(path)

    if not review_path.exists():
        console.print(f"[red]❌ Path not found: {path}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]🔍 Code Review:[/bold] [cyan]{review_path}[/cyan]\n")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Analyzing code...", total=None)
        review_results = review_code(str(review_path))
        progress.update(task, completed=True)

    # Display review results
    issues = review_results.get("issues", [])
    suggestions = review_results.get("suggestions", [])
    score = review_results.get("score", 0)

    # Overall score
    score_color = "green" if score >= 80 else "yellow" if score >= 60 else "red"
    console.print(f"\n[bold]Overall Score:[/bold] [{score_color}]{score}/100[/{score_color}]")

    # Issues
    if issues:
        console.print("\n[bold red]❌ Issues Found:[/bold red]\n")
        for issue in issues:
            console.print(
                f"  • [red]{issue.get('type', 'Issue')}:[/red] {issue.get('message', '')}"
            )
            if issue.get("file"):
                console.print(f"    [dim]File: {issue['file']}[/dim]")
            if issue.get("line"):
                console.print(f"    [dim]Line: {issue['line']}[/dim]")

    # Suggestions
    if suggestions:
        console.print("\n[bold yellow]💡 Suggestions:[/bold yellow]\n")
        for suggestion in suggestions:
            category = suggestion.get("category", "Improvement")
            message = suggestion.get("message", "")
            console.print(f"  • [yellow]{category}:[/yellow] {message}")

    # Positive feedback
    positives = review_results.get("positives", [])
    if positives:
        console.print("\n[bold green]✅ Good Practices:[/bold green]\n")
        for positive in positives:
            console.print(f"  • [green]{positive}[/green]")

    # Code examples for improvements
    examples = review_results.get("examples", [])
    if examples:
        console.print("\n[bold]📝 Code Examples:[/bold]\n")
        for example in examples:
            console.print(f"[bold]{example.get('title', 'Example')}:[/bold]")
            console.print(
                Syntax(
                    example.get("code", ""),
                    example.get("language", "python"),
                    theme="monokai",
                    line_numbers=True,
                )
            )

    console.print("\n")


@cli.command()
@click.option("--brick", "-b", type=int, help="Specific brick number to submit")
def submit_brick_cmd(brick: Optional[int]) -> None:
    """Submit a completed brick - run tests, check coverage, verify criteria.

    Args:
        brick: Optional brick number to submit (defaults to current brick)
    """
    console.print("\n[bold]🎓 Submitting Brick[/bold]\n")

    # Determine which brick to submit
    if brick:
        brick_num = brick
    else:
        # Try to detect current brick from git branch or directory
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
                if "brick-" in branch:
                    # Extract brick number from branch name
                    parts = branch.split("-")
                    for part in parts:
                        if part.isdigit():
                            brick_num = int(part)
                            break
                    else:
                        brick_num = None
                else:
                    brick_num = None
            else:
                brick_num = None
        except Exception:
            brick_num = None

        if not brick_num:
            msg = (
                "[yellow]⚠️  Could not detect current brick. " "Please specify with --brick[/yellow]"
            )
            console.print(msg)
            brick_num_str = Prompt.ask("Enter brick number")
            brick_num = int(brick_num_str)

    if brick_num is None:
        console.print("[red]❌ Brick number is required[/red]")
        sys.exit(1)

    console.print(f"[bold]Submitting Brick {brick_num:02d}...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running submission checks...", total=None)
        results = submit_brick(brick_num)
        progress.update(task, completed=True)

    # Display results
    tests_passed = results.get("tests_passed", False)
    coverage_met = results.get("coverage_met", False)
    criteria_met = results.get("criteria_met", False)
    all_passed = tests_passed and coverage_met and criteria_met

    # Test results
    console.print("\n[bold]🧪 Test Results:[/bold]")
    if tests_passed:
        console.print("[green]✅ All tests passed[/green]")
    else:
        console.print("[red]❌ Some tests failed[/red]")
        failures = results.get("test_failures", [])
        for failure in failures:
            console.print(f"  • [red]{failure}[/red]")

    # Coverage
    coverage = results.get("coverage", 0)
    console.print(f"\n[bold]📊 Coverage:[/bold] {coverage:.1f}%")
    if coverage_met:
        console.print("[green]✅ Coverage requirement met (≥80%)[/green]")
    else:
        console.print("[red]❌ Coverage below requirement (need ≥80%)[/red]")

    # Acceptance criteria
    console.print("\n[bold]✅ Acceptance Criteria:[/bold]")
    criteria = results.get("criteria", [])
    for criterion in criteria:
        status = "✅" if criterion.get("met", False) else "❌"
        color = "green" if criterion.get("met", False) else "red"
        console.print(f"  [{color}]{status}[/{color}] {criterion.get('description', '')}")

    # Final result
    console.print("\n")
    if all_passed:
        console.print(
            Panel.fit(
                "[bold green]🎉 Congratulations! Brick submission successful![/bold green]\n\n"
                "You've completed this brick and met all requirements.\n"
                "Your code is production-ready!",
                border_style="green",
            )
        )

        # Generate certificate
        certificate = results.get("certificate")
        if certificate:
            console.print(f"\n[bold]📜 Certificate:[/bold] {certificate}")
    else:
        console.print(
            Panel.fit(
                "[bold red]❌ Submission incomplete[/bold red]\n\n"
                "Please address the issues above before submitting.",
                border_style="red",
            )
        )

    console.print("\n")


if __name__ == "__main__":
    cli()
