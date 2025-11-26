"""Brick submission and validation.

This module handles brick submission, running tests, checking coverage,
and verifying acceptance criteria.
"""

import json
import subprocess
import uuid
from datetime import datetime
from typing import Any, Dict, List

from agent.brick_data import get_brick_info, get_brick_path


def run_tests(brick_number: int) -> Dict[str, Any]:
    """Run tests for a brick.

    Args:
        brick_number: The brick number

    Returns:
        dict: Test results
    """
    brick_path = get_brick_path(brick_number)
    tests_path = brick_path / "tests"

    if not tests_path.exists():
        return {
            "passed": False,
            "failures": ["No tests directory found"],
            "error": "Tests directory not found",
        }

    # Check if pytest is available
    try:
        result = subprocess.run(
            ["pytest", str(tests_path), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        passed = result.returncode == 0
        output = result.stdout + result.stderr

        # Parse failures from output
        failures = []
        if not passed:
            # Try to extract failure messages
            lines = output.split("\n")
            for i, line in enumerate(lines):
                if "FAILED" in line or "ERROR" in line:
                    # Get context around failure
                    context = "\n".join(lines[max(0, i - 2) : min(len(lines), i + 5)])
                    failures.append(context[:200])  # Limit length

        return {
            "passed": passed,
            "failures": failures if failures else ["Tests failed - check output"],
            "output": output,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "failures": ["Tests timed out after 2 minutes"],
            "error": "Timeout",
        }
    except FileNotFoundError:
        return {
            "passed": False,
            "failures": ["pytest not found - install with: pip install pytest"],
            "error": "pytest not installed",
        }
    except Exception as e:
        return {
            "passed": False,
            "failures": [f"Error running tests: {str(e)}"],
            "error": str(e),
        }


def check_coverage(brick_number: int) -> Dict[str, Any]:
    """Check test coverage for a brick.

    Args:
        brick_number: The brick number

    Returns:
        dict: Coverage information
    """
    brick_path = get_brick_path(brick_number)
    src_path = brick_path / "src"
    tests_path = brick_path / "tests"

    if not src_path.exists() or not tests_path.exists():
        return {
            "coverage": 0.0,
            "met": False,
            "error": "Source or tests directory not found",
        }

    # Try to run coverage
    try:
        result = subprocess.run(
            [
                "pytest",
                str(tests_path),
                "--cov",
                str(src_path),
                "--cov-report",
                "term-missing",
                "--cov-report",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        # Try to read coverage JSON
        coverage_json = brick_path / "coverage.json"
        if coverage_json.exists():
            try:
                with open(coverage_json, "r") as f:
                    coverage_data = json.load(f)
                    total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                    return {
                        "coverage": total_coverage,
                        "met": total_coverage >= 80.0,
                    }
            except Exception:
                pass

        # Fallback: parse from output
        output = result.stdout
        for line in output.split("\n"):
            if "TOTAL" in line and "%" in line:
                # Try to extract percentage
                parts = line.split()
                for part in parts:
                    if "%" in part:
                        try:
                            coverage = float(part.replace("%", ""))
                            return {
                                "coverage": coverage,
                                "met": coverage >= 80.0,
                            }
                        except ValueError:
                            pass

        return {
            "coverage": 0.0,
            "met": False,
            "error": "Could not parse coverage",
        }
    except subprocess.TimeoutExpired:
        return {
            "coverage": 0.0,
            "met": False,
            "error": "Coverage check timed out",
        }
    except FileNotFoundError:
        return {
            "coverage": 0.0,
            "met": False,
            "error": "pytest-cov not found - install with: pip install pytest-cov",
        }
    except Exception as e:
        return {
            "coverage": 0.0,
            "met": False,
            "error": f"Error checking coverage: {str(e)}",
        }


def verify_acceptance_criteria(brick_number: int) -> List[Dict[str, Any]]:
    """Verify acceptance criteria for a brick.

    Args:
        brick_number: The brick number

    Returns:
        List of criteria with met status
    """
    brick_info = get_brick_info(brick_number)
    tasks = brick_info.get("tasks", [])

    criteria_list = []
    for task in tasks:
        acceptance_criteria = task.get("acceptance_criteria", [])
        for criterion in acceptance_criteria:
            # Simple heuristic: check if relevant files exist
            # In production, this would be more sophisticated
            brick_path = get_brick_path(brick_number)
            src_path = brick_path / "src"

            # Check if source files exist (basic check)
            met = src_path.exists() and len(list(src_path.rglob("*.py"))) > 0

            criteria_list.append(
                {
                    "description": criterion,
                    "met": met,
                }
            )

    return criteria_list


def generate_certificate(brick_number: int) -> str:
    """Generate a completion certificate for a brick.

    Args:
        brick_number: The brick number

    Returns:
        str: Path to certificate file
    """
    brick_info = get_brick_info(brick_number)
    brick_name = brick_info.get("title", f"Brick {brick_number}")

    certificate_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d")

    certificate = {
        "certificate_id": certificate_id,
        "brick_number": brick_number,
        "brick_name": brick_name,
        "completion_date": timestamp,
        "issued_by": "AgentBricks",
    }

    # Save certificate
    brick_path = get_brick_path(brick_number)
    cert_path = brick_path / f"certificate_{brick_number:02d}.json"

    with open(cert_path, "w") as f:
        json.dump(certificate, f, indent=2)

    return str(cert_path)


def submit_brick(brick_number: int) -> Dict[str, Any]:
    """Submit a brick for validation.

    Args:
        brick_number: The brick number

    Returns:
        dict: Submission results
    """
    # Run tests
    test_results = run_tests(brick_number)
    tests_passed = test_results.get("passed", False)
    test_failures = test_results.get("failures", [])

    # Check coverage
    coverage_results = check_coverage(brick_number)
    coverage = coverage_results.get("coverage", 0.0)
    coverage_met = coverage_results.get("met", False)

    # Verify acceptance criteria
    criteria = verify_acceptance_criteria(brick_number)
    criteria_met = all(c.get("met", False) for c in criteria)

    # All checks passed?
    all_passed = tests_passed and coverage_met and criteria_met

    # Generate certificate if passed
    certificate = None
    if all_passed:
        certificate = generate_certificate(brick_number)

    return {
        "tests_passed": tests_passed,
        "test_failures": test_failures,
        "coverage": coverage,
        "coverage_met": coverage_met,
        "criteria": criteria,
        "criteria_met": criteria_met,
        "all_passed": all_passed,
        "certificate": certificate,
    }
