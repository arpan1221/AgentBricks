# Contributing to AgentBricks

Thank you for your interest in contributing to AgentBricks! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Commit Message Format](#commit-message-format)
- [Pull Request Process and Review Guidelines](#pull-request-process-and-review-guidelines)
- [Code Style Requirements](#code-style-requirements)
- [Testing Requirements](#testing-requirements)

## Setting Up the Development Environment

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/agentbricks.git
   cd agentbricks
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3.11+ -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install development dependencies
   ```

4. **Install Pre-commit Hooks**:
   ```bash
   pre-commit install
   ```

5. **Verify Installation**:
   ```bash
   pytest --version
   black --version
   ruff --version
   mypy --version
   ```

## Branch Naming Conventions

We use a prefix-based naming convention for branches:

- **Feature branches**: `feature/<short-description>`
  - Example: `feature/add-kafka-consumer`

- **Bug fix branches**: `bugfix/<short-description>`
  - Example: `bugfix/fix-feature-computation`

- **Documentation branches**: `docs/<short-description>`
  - Example: `docs/update-readme`

- **Hotfix branches**: `hotfix/<short-description>`
  - Example: `hotfix/fix-security-issue`

- **Refactor branches**: `refactor/<short-description>`
  - Example: `refactor/optimize-feature-store`

**Guidelines**:
- Use lowercase letters and hyphens
- Keep descriptions concise but descriptive
- No spaces or special characters (except hyphens)

## Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring without changing functionality
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates
- `ci`: CI/CD configuration changes
- `build`: Build system or external dependency changes

### Scope

The scope should indicate the area of the codebase affected:
- Module name (e.g., `sim`, `bricks`, `api`)
- Component name (e.g., `ingestion`, `features`, `training`)
- Brick name (e.g., `brick-01`, `brick-02`)

### Examples

```
feat(brick-01): add event validation middleware

Implement Pydantic validation for all event types.
Includes comprehensive error messages and logging.

Closes #42
```

```
fix(sim): correct user preference generation

Users were getting identical preference vectors.
Now using proper random seed per user.

Fixes #23
```

```
docs(readme): update installation instructions

Add troubleshooting section for Docker issues.
```

```
test(bricks): add unit tests for feature engineering

Achieve 85% coverage for feature computation module.
```

## Pull Request Process and Review Guidelines

### Before Submitting a PR

1. **Update Documentation**: Ensure all documentation is up to date
2. **Add Tests**: New features should include tests
3. **Run Tests**: Ensure all tests pass locally
4. **Check Code Style**: Run `black`, `ruff`, and `mypy`
5. **Update CHANGELOG.md**: Add entry for your changes

### PR Guidelines

1. **Create a Descriptive Title**: Use conventional commit format
2. **Write a Clear Description**:
   - What changes are made and why
   - Reference related issues (e.g., "Closes #123")
   - Include screenshots for UI changes
   - List breaking changes if any

3. **Keep PRs Focused**:
   - One feature or fix per PR
   - Keep changes small and reviewable
   - Split large features into multiple PRs

4. **Ensure CI Passes**: All checks must pass before review

### Review Guidelines

**For Reviewers**:
- Be respectful and constructive in feedback
- Focus on code quality, not personal preferences
- Approve PRs that meet project standards
- Request changes with clear explanations
- Respond promptly to review requests

**For Authors**:
- Address all review comments
- Be open to feedback and suggestions
- Keep discussions professional
- Re-request review after addressing comments

### PR Template

When opening a PR, include:

- **Type**: Feature, Bug Fix, Documentation, etc.
- **Description**: What and why
- **Related Issues**: Link to relevant issues
- **Testing**: How to test the changes
- **Checklist**: Confirm all items are completed

## Code Style Requirements

### Python Code Style

We enforce code style using automated tools:

1. **Black** (Line length: 100):
   ```bash
   black .
   ```

2. **Ruff** (Selected rules: E, F, I):
   ```bash
   ruff check .
   ruff format .  # Optional: auto-fix formatting
   ```

3. **mypy** (Strict mode):
   ```bash
   mypy .
   ```

### Code Style Guidelines

- **Type Hints**: Required on all functions and class methods
- **Docstrings**: Google style for all public functions/classes
- **Line Length**: Maximum 100 characters
- **Naming Conventions**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`
  - Private: `_leading_underscore`

- **Imports**: Organized with standard library, third-party, then local imports
- **Error Handling**: Never use bare `except:` statements
- **Logging**: Use structured logging with appropriate levels

### Example

```python
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def compute_features(
    user_id: str,
    as_of_date: datetime,
    interactions: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, float]:
    """
    Compute user features with point-in-time correctness.

    Args:
        user_id: Unique user identifier
        as_of_date: Compute features as of this timestamp
        interactions: Optional list of user interactions

    Returns:
        Dictionary of computed features

    Raises:
        ValueError: If user_id is invalid

    Example:
        >>> features = compute_features("user_123", datetime.now())
        >>> print(features['total_watch_time'])
        3600
    """
    if not user_id:
        raise ValueError("user_id cannot be empty")

    try:
        # Feature computation logic
        logger.info(f"Computing features for user {user_id}")
        return {"total_watch_time": 3600.0}
    except Exception as e:
        logger.error(f"Failed to compute features: {e}", exc_info=True)
        raise
```

## Testing Requirements

### Test Coverage

- **Minimum Coverage**: 80% for all modules
- **New Code**: Must include tests
- **Critical Paths**: 100% coverage required

### Test Organization

- **Unit Tests**: `tests/unit/` - Test individual components
- **Integration Tests**: `tests/integration/` - Test component interactions
- **E2E Tests**: `tests/e2e/` - Test complete system flows

### Test Naming

Follow the pattern: `test_<function>_with_<condition>_returns_<expected>`

Examples:
- `test_compute_features_with_new_user_returns_defaults`
- `test_process_event_with_invalid_data_raises_error`
- `test_train_model_with_small_dataset_converges`

### Writing Tests

```python
import pytest
from datetime import datetime

def test_compute_features_with_new_user_returns_defaults():
    """Test that new users get default feature values."""
    # Arrange
    user_id = "new_user_123"
    as_of_date = datetime.now()

    # Act
    features = compute_features(user_id, as_of_date)

    # Assert
    assert features["total_watch_time"] == 0.0
    assert features["avg_watch_time"] == 0.0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_features.py

# Run specific test
pytest tests/unit/test_features.py::test_compute_features_with_new_user

# Run with markers
pytest -m "not slow"  # Skip slow tests
```

### Test Best Practices

1. **Use Fixtures**: Reuse test data with pytest fixtures
2. **Mock External Dependencies**: Mock Kafka, Redis, databases, etc.
3. **Test Edge Cases**: Empty inputs, None values, boundary conditions
4. **Test Error Conditions**: Verify proper error handling
5. **Keep Tests Fast**: Unit tests should run quickly
6. **Test Behavior, Not Implementation**: Focus on what, not how

## Additional Resources

- [Project README](../README.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Architecture Documentation](docs/architecture/README.md)
- [Best Practices](docs/best-practices/README.md)

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search closed/open issues
3. Open a new issue with the `question` label

Thank you for contributing to AgentBricks! 🎉
