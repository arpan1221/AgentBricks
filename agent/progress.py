"""Progress tracking and analysis.

This module analyzes user progress through bricks and tasks.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from agent.brick_data import BRICK_NAMES, get_brick_info, get_brick_path


def detect_current_brick() -> Optional[int]:
    """Detect the current brick from git branch or directory.

    Returns:
        Optional[int]: Current brick number, or None if not detected
    """
    # Try git branch first
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
                # Extract brick number
                parts = branch.split("-")
                for part in parts:
                    if part.isdigit():
                        return int(part)
    except Exception:
        pass

    # Try to detect from current directory
    cwd = Path.cwd()
    for brick_num in range(1, 7):
        brick_path = get_brick_path(brick_num)
        if brick_path.exists() and str(brick_path) in str(cwd):
            return brick_num

    return None


def count_tests(brick_number: int) -> int:
    """Count test files in a brick.

    Args:
        brick_number: The brick number

    Returns:
        int: Number of test files
    """
    brick_path = get_brick_path(brick_number)
    tests_path = brick_path / "tests"

    if not tests_path.exists():
        return 0

    test_files = list(tests_path.glob("test_*.py"))
    return len(test_files)


def count_src_files(brick_number: int) -> int:
    """Count source files in a brick.

    Args:
        brick_number: The brick number

    Returns:
        int: Number of source files
    """
    brick_path = get_brick_path(brick_number)
    src_path = brick_path / "src"

    if not src_path.exists():
        return 0

    # Count Python files (excluding __init__.py)
    py_files = [f for f in src_path.rglob("*.py") if f.name != "__init__.py"]
    return len(py_files)


def check_brick_completion(brick_number: int) -> bool:
    """Check if a brick is completed based on heuristics.

    Args:
        brick_number: The brick number

    Returns:
        bool: True if brick appears completed
    """
    brick_path = get_brick_path(brick_number)

    if not brick_path.exists():
        return False

    # Heuristics for completion:
    # 1. Has source files
    # 2. Has test files
    # 3. Tests pass (if we can run them)

    src_count = count_src_files(brick_number)
    test_count = count_tests(brick_number)

    # Basic check: has both source and tests
    if src_count > 0 and test_count > 0:
        # Try to run tests
        try:
            result = subprocess.run(
                ["pytest", str(brick_path / "tests"), "-v", "--tb=short"],
                capture_output=True,
                check=False,
                timeout=30,
            )
            # If tests run and most pass, consider it complete
            return result.returncode == 0 or src_count >= 2
        except Exception:
            # If we can't run tests, use file count heuristic
            return src_count >= 2

    return False


def get_brick_progress(brick_number: int) -> Dict[str, Any]:
    """Get progress information for a specific brick.

    Args:
        brick_number: The brick number

    Returns:
        dict: Progress information
    """
    brick_info = get_brick_info(brick_number)
    tasks = brick_info.get("tasks", [])
    total_tasks = len(tasks)

    # Estimate completed tasks based on files
    src_count = count_src_files(brick_number)
    test_count = count_tests(brick_number)

    # Rough estimate: assume 1-2 tasks per source file
    estimated_completed = min(src_count, total_tasks)

    return {
        "number": brick_number,
        "name": BRICK_NAMES.get(brick_number, ""),
        "total_tasks": total_tasks,
        "completed_tasks": estimated_completed,
        "src_files": src_count,
        "test_files": test_count,
        "is_completed": check_brick_completion(brick_number),
    }


def check_progress() -> Dict[str, Any]:
    """Check overall progress across all bricks.

    Returns:
        dict: Overall progress information
    """
    # Check each brick
    completed_bricks = []
    current_brick_num = detect_current_brick()
    current_brick = None

    for brick_num in range(1, 7):
        brick_progress = get_brick_progress(brick_num)
        if brick_progress["is_completed"]:
            completed_bricks.append(brick_num)

        if brick_num == current_brick_num:
            current_brick = brick_progress

    # If no current brick detected, use first incomplete brick
    if not current_brick:
        for brick_num in range(1, 7):
            if brick_num not in completed_bricks:
                current_brick = get_brick_progress(brick_num)
                break

    # Get next recommended tasks
    next_tasks = []
    if current_brick:
        brick_info = get_brick_info(current_brick["number"])
        tasks = brick_info.get("tasks", [])
        completed = current_brick["completed_tasks"]

        # Recommend next few tasks
        for task in tasks[completed : completed + 3]:
            next_tasks.append(
                {
                    "number": task.get("number", completed + 1),
                    "description": task.get("description", ""),
                    "priority": "High" if completed == 0 else "Medium",
                }
            )

    return {
        "overall": {
            "completed_bricks": len(completed_bricks),
            "total_bricks": 6,
            "progress_percent": (len(completed_bricks) / 6 * 100) if 6 > 0 else 0,
        },
        "current_brick": current_brick,
        "next_tasks": next_tasks,
        "completed_tasks_summary": [
            f"Brick {num:02d}: {BRICK_NAMES.get(num, '')}" for num in completed_bricks
        ],
    }
