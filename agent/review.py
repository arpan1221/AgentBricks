"""Code review and best practices checking.

This module analyzes code against AgentBricks best practices.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List


def analyze_python_file(file_path: Path) -> List[Dict[str, Any]]:
    """Analyze a Python file for best practices.

    Args:
        file_path: Path to the Python file

    Returns:
        List of issues found
    """
    issues = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        issues.append(
            {
                "type": "Syntax Error",
                "message": f"Syntax error: {e.msg}",
                "file": str(file_path),
                "line": e.lineno,
            }
        )
        return issues
    except Exception as e:
        issues.append(
            {
                "type": "Error",
                "message": f"Could not parse file: {e}",
                "file": str(file_path),
            }
        )
        return issues

    # Check for type hints
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if function has type hints
            if not node.returns and node.name not in ["__init__", "__str__", "__repr__"]:
                # Check if it's a private method (starts with _)
                if not node.name.startswith("_"):
                    issues.append(
                        {
                            "type": "Missing Type Hint",
                            "message": f"Function '{node.name}' missing return type hint",
                            "file": str(file_path),
                            "line": node.lineno,
                        }
                    )

            # Check parameters
            for arg in node.args.args:
                if arg.annotation is None and arg.arg != "self" and not arg.arg.startswith("_"):
                    issues.append(
                        {
                            "type": "Missing Type Hint",
                            "message": f"Parameter '{arg.arg}' in '{node.name}' missing type hint",
                            "file": str(file_path),
                            "line": node.lineno,
                        }
                    )

    # Check for docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not ast.get_docstring(node) and node.name not in ["__init__"]:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    issues.append(
                        {
                            "type": "Missing Docstring",
                            "message": f"Function '{node.name}' missing docstring",
                            "file": str(file_path),
                            "line": node.lineno,
                        }
                    )
                elif isinstance(node, ast.ClassDef):
                    issues.append(
                        {
                            "type": "Missing Docstring",
                            "message": f"Class '{node.name}' missing docstring",
                            "file": str(file_path),
                            "line": node.lineno,
                        }
                    )

    # Check for bare except
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                issues.append(
                    {
                        "type": "Bare Except",
                        "message": "Bare except clause found - should specify exception type",
                        "file": str(file_path),
                        "line": node.lineno,
                    }
                )

    # Check for print statements (should use logging)
    content_lines = content.split("\n")
    for i, line in enumerate(content_lines, 1):
        stripped = line.strip()
        if stripped.startswith("print(") and "logger" not in line.lower():
            issues.append(
                {
                    "type": "Use Logging",
                    "message": "Use logging instead of print statements",
                    "file": str(file_path),
                    "line": i,
                }
            )

    return issues


def review_code(path: str) -> Dict[str, Any]:
    """Review code at the given path.

    Args:
        path: Path to code directory or file

    Returns:
        dict: Review results with issues, suggestions, and score
    """
    review_path = Path(path)

    if not review_path.exists():
        return {
            "score": 0,
            "issues": [{"type": "Error", "message": f"Path not found: {path}"}],
            "suggestions": [],
            "positives": [],
        }

    # Collect all Python files
    if review_path.is_file():
        python_files = [review_path] if review_path.suffix == ".py" else []
    else:
        python_files = list(review_path.rglob("*.py"))
        # Exclude test files and __pycache__
        python_files = [
            f for f in python_files if "__pycache__" not in str(f) and "test_" not in f.name
        ]

    if not python_files:
        return {
            "score": 0,
            "issues": [{"type": "Info", "message": "No Python files found to review"}],
            "suggestions": [],
            "positives": [],
        }

    # Analyze each file
    all_issues = []
    for py_file in python_files:
        issues = analyze_python_file(py_file)
        all_issues.extend(issues)

    # Calculate score (100 - issues * penalty)
    issue_penalty = 5  # 5 points per issue
    base_score = 100
    score = max(0, base_score - len(all_issues) * issue_penalty)

    # Categorize issues
    issues_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for issue in all_issues:
        issue_type = issue.get("type", "Other")
        if issue_type not in issues_by_type:
            issues_by_type[issue_type] = []
        issues_by_type[issue_type].append(issue)

    # Generate suggestions
    suggestions = []
    if "Missing Type Hint" in issues_by_type:
        msg = (
            "Add type hints to all functions and parameters. "
            "This improves code readability and enables static type checking."
        )
        suggestions.append({"category": "Type Hints", "message": msg})

    if "Missing Docstring" in issues_by_type:
        msg = (
            "Add Google-style docstrings to all public functions and classes. "
            "Include Args, Returns, and Raises sections."
        )
        suggestions.append({"category": "Documentation", "message": msg})

    if "Bare Except" in issues_by_type:
        msg = (
            "Specify exception types in except clauses. "
            "Use 'except ValueError as e:' instead of 'except:'."
        )
        suggestions.append({"category": "Error Handling", "message": msg})

    if "Use Logging" in issues_by_type:
        msg = (
            "Use structured logging instead of print statements. "
            "Import logging and use logger.info(), logger.error(), etc."
        )
        suggestions.append({"category": "Logging", "message": msg})

    # Positive feedback
    positives = []
    if len(python_files) > 0:
        positives.append(f"Found {len(python_files)} Python file(s) to review")

    if score >= 80:
        positives.append("Great job! Your code follows most best practices.")
    elif score >= 60:
        positives.append("Good progress! A few improvements will make your code production-ready.")

    # Code examples
    examples = []
    if "Missing Type Hint" in issues_by_type:
        examples.append(
            {
                "title": "Adding Type Hints",
                "language": "python",
                "code": """# Before:
def process_event(event):
    return event.id

# After:
def process_event(event: Event) -> str:
    return event.id
""",
            }
        )

    if "Missing Docstring" in issues_by_type:
        examples.append(
            {
                "title": "Adding Docstrings",
                "language": "python",
                "code": '''def process_event(event: Event) -> str:
    """
    Process an event and return its ID.

    Args:
        event: The event to process

    Returns:
        str: The event ID

    Raises:
        ValueError: If event is invalid
    """
    return event.id
''',
            }
        )

    return {
        "score": score,
        "issues": all_issues,
        "suggestions": suggestions,
        "positives": positives,
        "examples": examples,
    }
