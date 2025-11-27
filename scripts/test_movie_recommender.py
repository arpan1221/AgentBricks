#!/usr/bin/env python3
"""
Test and run script for the Movie Recommender story arc.

This script automates the setup, testing, and running of the entire
Movie Recommender story arc end-to-end.

Usage:
    python scripts/test_movie_recommender.py [options]

Options:
    --setup-only      Only set up infrastructure, don't run tests
    --test-only       Only run tests, assume infrastructure is running
    --e2e-only        Only run end-to-end tests
    --unit-only       Only run unit tests
    --integration-only Only run integration tests
    --no-cleanup      Don't clean up services after tests
    --verbose         Verbose output
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_section(title: str) -> None:
    """Print a formatted section title."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def check_prerequisites() -> Tuple[bool, List[str]]:
    """
    Check if all prerequisites are met.

    Returns:
        Tuple of (all_met, missing_items)
    """
    print_section("Checking Prerequisites")
    missing = []

    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 11):
        missing.append(f"Python 3.11+ (found {python_version.major}.{python_version.minor})")
    else:
        print_success(f"Python {python_version.major}.{python_version.minor}")

    # Check Docker
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print_success(f"Docker: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("Docker not found or not running")
        print_error("Docker not found or not running")

    # Check Docker Compose
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print_success(f"Docker Compose: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("Docker Compose not found")
        print_error("Docker Compose not found")

    # Check Docker is running
    try:
        subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            check=True,
        )
        print_success("Docker daemon is running")
    except subprocess.CalledProcessError:
        missing.append("Docker daemon is not running")
        print_error("Docker daemon is not running")

    # Check required Python packages
    # Map package names to their import names (some differ)
    package_import_map = {
        "fastapi": "fastapi",
        "kafka-python": "kafka",  # kafka-python imports as 'kafka'
        "pymongo": "pymongo",
        "pytest": "pytest",
        "tqdm": "tqdm",  # Required for data generation
        "duckdb": "duckdb",  # Required for feature store (brick-02)
    }

    for package_name, import_name in package_import_map.items():
        try:
            __import__(import_name)
            print_success(f"Package {package_name} installed")
        except ImportError:
            missing.append(f"Python package: {package_name}")
            print_error(f"Package {package_name} not found")

    if missing:
        print_warning(f"Missing prerequisites: {', '.join(missing)}")
        return False, missing

    print_success("All prerequisites met!")
    return True, []


def get_project_root() -> Path:
    """Get the project root directory."""
    # This script should be in scripts/ directory
    script_dir = Path(__file__).parent
    return script_dir.parent


def run_command(
    command: List[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_output: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """
    Run a shell command.

    Args:
        command: Command to run as list of strings
        cwd: Working directory
        check: Raise exception on non-zero exit
        capture_output: Capture stdout/stderr
        env: Environment variables to set (merged with current env)

    Returns:
        CompletedProcess result
    """
    if cwd:
        cwd = Path(cwd).resolve()

    logger.debug(f"Running: {' '.join(command)} in {cwd}")

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=check,
            capture_output=capture_output,
            text=True,
            env=env,
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}")
        if capture_output:
            logger.error(f"Error output: {e.stderr}")
        raise


def generate_synthetic_data(users: int = 1000, movies: int = 500, days: int = 7) -> bool:
    """
    Generate synthetic data for testing.

    Args:
        users: Number of users to generate
        movies: Number of movies to generate
        days: Number of days of interactions

    Returns:
        True if successful
    """
    print_section("Generating Synthetic Data")
    project_root = get_project_root()
    sim_dir = project_root / "sim"

    if not sim_dir.exists():
        print_error("sim/ directory not found")
        return False

    try:
        # Set PYTHONPATH to project root so Python can find the sim package
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = str(project_root)

        run_command(
            [
                "python",
                "generate.py",
                "generate-all",
                "--user-count",
                str(users),
                "--movie-count",
                str(movies),
                "--days",
                str(days),
            ],
            cwd=sim_dir,
            env=env,
        )
        print_success(f"Generated {users} users, {movies} movies, {days} days of data")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to generate data: {e}")
        return False


def start_brick01_services() -> bool:
    """
    Start Brick 01 infrastructure services.

    Returns:
        True if successful
    """
    print_section("Starting Brick 01 Services")
    project_root = get_project_root()
    brick01_dir = project_root / "stories" / "movie-recommender" / "brick-01-data-collection"

    if not brick01_dir.exists():
        print_error("Brick 01 directory not found")
        return False

    try:
        # Start services
        run_command(["docker", "compose", "up", "-d"], cwd=brick01_dir)
        print_success("Docker Compose services started")

        # Wait for services to be ready
        print("Waiting for services to be ready...")
        max_wait = 120  # 2 minutes
        wait_interval = 5
        waited = 0

        # Wait for API
        api_ready = False
        while waited < max_wait and not api_ready:
            try:
                result = subprocess.run(
                    ["curl", "-f", "http://localhost:8000/health"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    api_ready = True
                    print_success("API is ready")
                else:
                    time.sleep(wait_interval)
                    waited += wait_interval
            except Exception:
                time.sleep(wait_interval)
                waited += wait_interval

        if not api_ready:
            print_warning("API did not become ready in time, but continuing...")

        # Check service status
        result = run_command(
            ["docker", "compose", "ps"],
            cwd=brick01_dir,
            capture_output=True,
        )
        print(result.stdout)

        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to start services: {e}")
        return False


def stop_brick01_services() -> bool:
    """
    Stop Brick 01 infrastructure services.

    Returns:
        True if successful
    """
    print_section("Stopping Brick 01 Services")
    project_root = get_project_root()
    brick01_dir = project_root / "stories" / "movie-recommender" / "brick-01-data-collection"

    if not brick01_dir.exists():
        return False

    try:
        run_command(["docker", "compose", "down", "-v"], cwd=brick01_dir)
        print_success("Services stopped")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to stop services: {e}")
        return False


def run_unit_tests() -> Tuple[bool, int, int]:
    """
    Run unit tests for all bricks.

    Returns:
        Tuple of (success, passed, failed)
    """
    print_section("Running Unit Tests")
    project_root = get_project_root()
    test_dirs = [
        project_root / "stories" / "movie-recommender" / "brick-01-data-collection" / "tests",
        project_root / "stories" / "movie-recommender" / "brick-02-feature-engineering" / "tests",
    ]

    passed = 0
    failed = 0

    for test_dir in test_dirs:
        if not test_dir.exists():
            print_warning(f"Test directory not found: {test_dir}")
            continue

        brick_name = test_dir.parent.name
        print(f"\nRunning tests for {brick_name}...")

        try:
            result = run_command(
                ["pytest", "-v", "--tb=short"],
                cwd=test_dir.parent,
                check=False,
                capture_output=True,
            )

            if result.returncode == 0:
                print_success(f"All tests passed for {brick_name}")
                passed += 1
            else:
                print_error(f"Some tests failed for {brick_name}")
                print(result.stdout)
                print(result.stderr)
                failed += 1
        except Exception as e:
            print_error(f"Failed to run tests for {brick_name}: {e}")
            failed += 1

    return failed == 0, passed, failed


def run_integration_tests() -> Tuple[bool, int, int]:
    """
    Run integration tests.

    Returns:
        Tuple of (success, passed, failed)
    """
    print_section("Running Integration Tests")
    project_root = get_project_root()
    integration_test_dir = project_root / "tests" / "integration"

    if not integration_test_dir.exists():
        print_warning("Integration test directory not found")
        return True, 0, 0

    try:
        result = run_command(
            ["pytest", "-v", "--tb=short"],
            cwd=project_root,
            check=False,
            capture_output=True,
        )

        if result.returncode == 0:
            print_success("All integration tests passed")
            return True, 1, 0
        elif result.returncode == 5:
            # Exit code 5 means no tests were collected
            print_warning("No integration tests found")
            return True, 0, 0
        else:
            print_error("Some integration tests failed")
            print(result.stdout)
            print(result.stderr)
            return False, 0, 1
    except Exception as e:
        print_error(f"Failed to run integration tests: {e}")
        return False, 0, 1


def run_e2e_tests() -> Tuple[bool, int, int]:
    """
    Run end-to-end tests.

    Returns:
        Tuple of (success, passed, failed)
    """
    print_section("Running End-to-End Tests")
    project_root = get_project_root()
    e2e_test_dir = project_root / "tests" / "e2e"

    if not e2e_test_dir.exists():
        print_warning("E2E test directory not found")
        return True, 0, 0

    try:
        result = run_command(
            ["pytest", "-v", "--tb=short", "-m", "e2e"],
            cwd=project_root,
            check=False,
            capture_output=True,
        )

        if result.returncode == 0:
            print_success("All E2E tests passed")
            return True, 1, 0
        elif result.returncode == 5:
            # Exit code 5 means no tests were collected
            print_warning("No E2E tests found")
            return True, 0, 0
        else:
            print_error("Some E2E tests failed")
            print(result.stdout)
            print(result.stderr)
            return False, 0, 1
    except Exception as e:
        print_error(f"Failed to run E2E tests: {e}")
        return False, 0, 1


def test_api_endpoints() -> bool:
    """
    Test API endpoints manually.

    Returns:
        True if successful
    """
    print_section("Testing API Endpoints")

    # Test health endpoint
    try:
        result = subprocess.run(
            ["curl", "-f", "http://localhost:8000/health"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            print_success("Health endpoint is working")
        else:
            print_error("Health endpoint failed")
            return False
    except Exception as e:
        print_error(f"Failed to test health endpoint: {e}")
        return False

    # Test view event endpoint
    try:
        result = subprocess.run(
            [
                "curl",
                "-X",
                "POST",
                "http://localhost:8000/events/view",
                "-H",
                "Content-Type: application/json",
                "-d",
                (
                    '{"user_id": "test_user", "movie_id": "test_movie", '
                    '"timestamp": "2025-01-15T10:00:00Z"}'
                ),
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            print_success("View event endpoint is working")
        else:
            print_warning("View event endpoint may have issues")
            print(result.stderr.decode())
    except Exception as e:
        print_warning(f"Failed to test view event endpoint: {e}")

    return True


def print_summary(
    setup_success: bool,
    unit_success: bool,
    integration_success: bool,
    e2e_success: bool,
    unit_passed: int,
    unit_failed: int,
) -> None:
    """Print test summary."""
    print_section("Test Summary")

    if setup_success:
        print_success("Setup: ✓")
    else:
        print_error("Setup: ✗")

    if unit_success:
        print_success(f"Unit Tests: ✓ ({unit_passed} passed, {unit_failed} failed)")
    else:
        print_error(f"Unit Tests: ✗ ({unit_passed} passed, {unit_failed} failed)")

    if integration_success:
        print_success("Integration Tests: ✓")
    else:
        print_error("Integration Tests: ✗")

    if e2e_success:
        print_success("E2E Tests: ✓")
    else:
        print_error("E2E Tests: ✗")

    overall_success = all([setup_success, unit_success, integration_success, e2e_success])

    print("\n")
    if overall_success:
        print_success("Overall: All tests passed! 🎉")
    else:
        print_error("Overall: Some tests failed. Please review the output above.")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test and run Movie Recommender story arc")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only set up infrastructure, don't run tests",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run tests, assume infrastructure is running",
    )
    parser.add_argument(
        "--e2e-only",
        action="store_true",
        help="Only run end-to-end tests",
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Only run unit tests",
    )
    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="Only run integration tests",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up services after tests",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=1000,
        help="Number of users to generate (default: 1000)",
    )
    parser.add_argument(
        "--movies",
        type=int,
        default=500,
        help="Number of movies to generate (default: 500)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of interactions (default: 7)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check prerequisites
    all_met, missing = check_prerequisites()
    if not all_met:
        print_error("Prerequisites not met. Please install missing items:")
        for item in missing:
            print(f"  - {item}")
        return 1

    setup_success = True
    unit_success = True
    integration_success = True
    e2e_success = True
    unit_passed = 0
    unit_failed = 0

    try:
        # Setup phase
        if not args.test_only:
            if not generate_synthetic_data(args.users, args.movies, args.days):
                setup_success = False

            if not start_brick01_services():
                setup_success = False

            # Test API endpoints
            if not test_api_endpoints():
                setup_success = False

            if args.setup_only:
                print_success("Setup complete!")
                return 0

        # Testing phase
        if not args.setup_only:
            if args.unit_only or not args.e2e_only:
                unit_success, unit_passed, unit_failed = run_unit_tests()

            if args.integration_only or (not args.unit_only and not args.e2e_only):
                integration_success, _, _ = run_integration_tests()

            if args.e2e_only or (not args.unit_only and not args.integration_only):
                e2e_success, _, _ = run_e2e_tests()

        # Print summary
        print_summary(
            setup_success,
            unit_success,
            integration_success,
            e2e_success,
            unit_passed,
            unit_failed,
        )

        # Cleanup
        if not args.no_cleanup and not args.test_only:
            stop_brick01_services()

        return 0 if all([setup_success, unit_success, integration_success, e2e_success]) else 1

    except KeyboardInterrupt:
        print_warning("\nInterrupted by user")
        if not args.no_cleanup:
            stop_brick01_services()
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        logger.exception("Unexpected error")
        if not args.no_cleanup:
            stop_brick01_services()
        return 1


if __name__ == "__main__":
    sys.exit(main())
