# Contributing to AgentBricks

Thank you for your interest in contributing to AgentBricks! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Development Workflow](#development-workflow)
- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Commit Message Format](#commit-message-format)
- [Pull Request Process and Review Guidelines](#pull-request-process-and-review-guidelines)
- [Code Style Requirements](#code-style-requirements)
- [Testing Requirements](#testing-requirements)

## Ways to Contribute

There are many ways to contribute to AgentBricks, regardless of your experience level. We welcome contributions from everyone!

### For Beginners

Perfect for getting started with open source! These contributions help make AgentBricks more accessible and user-friendly.

- **📝 Fix typos in documentation**
  - Improve clarity and correctness
  - Fix grammar and spelling errors
  - Examples: README files, docstrings, comments

- **💬 Improve error messages**
  - Make error messages more helpful and actionable
  - Add context to error handling
  - Examples: Validation errors, connection failures

- **📚 Add examples**
  - Add code examples to documentation
  - Create tutorial snippets
  - Add use case examples
  - Examples: API usage, configuration examples

- **✅ Write tests**
  - Add unit tests for existing code
  - Improve test coverage
  - Add test cases for edge conditions
  - Examples: Test utility functions, test error handling

**Getting Started:**
1. Look for issues labeled `good first issue` or `beginner-friendly`
2. Comment on the issue to let others know you're working on it
3. Ask questions if you need help!

### For Intermediate

Great for developers with some experience who want to make meaningful contributions.

- **🧱 Add new bricks to existing story arcs**
  - Design and implement new learning modules
  - Follow existing brick patterns
  - Include tests and documentation
  - Examples: Add a new brick to Movie Recommender arc

- **⚡ Improve performance**
  - Optimize slow functions
  - Add caching where appropriate
  - Profile and improve bottlenecks
  - Examples: Feature computation, API response times

- **📊 Add visualizations**
  - Create diagrams for documentation
  - Add visualization tools for data exploration
  - Build dashboards for monitoring
  - Examples: Architecture diagrams, data flow charts

- **📖 Create tutorials**
  - Write step-by-step guides
  - Create video tutorials
  - Build interactive examples
  - Examples: "How to add a new feature", "Deploying to AWS"

**Getting Started:**
1. Look for issues labeled `help wanted` or `intermediate`
2. Discuss your approach in the issue before starting
3. Review similar implementations for consistency

### For Advanced

For experienced developers who want to make significant architectural contributions.

- **🎬 Design new story arcs**
  - Create complete learning paths
  - Design system architecture
  - Plan brick progression
  - Examples: TikTok For You Page, Amazon Product Ranking

- **🤖 Optimize ML models**
  - Improve model architectures
  - Optimize training pipelines
  - Add new model types
  - Examples: Better NCF variants, transformer models

- **☁️ Add cloud deployment options**
  - Create Terraform modules
  - Add Kubernetes manifests
  - Build deployment scripts
  - Examples: AWS deployment, GCP setup, Azure configs

- **🛠️ Build community tools**
  - Create CLI enhancements
  - Build web interfaces
  - Develop developer tools
  - Examples: Better mentor CLI, web dashboard, VS Code extension

**Getting Started:**
1. Open a discussion to propose your idea
2. Create a design document
3. Get feedback from maintainers before implementation
4. Break large features into smaller PRs

## Development Workflow

Follow this workflow to ensure smooth collaboration and code quality.

### Setting Up Development Environment

#### 1. Fork the Repository

1. Go to [AgentBricks on GitHub](https://github.com/yourusername/agentbricks)
2. Click the "Fork" button in the top right
3. This creates your own copy of the repository

#### 2. Clone Your Fork

```bash
# Replace YOUR_USERNAME with your GitHub username
git clone https://github.com/YOUR_USERNAME/agentbricks.git
cd agentbricks
```

#### 3. Add Upstream Remote

This allows you to sync with the main repository:

```bash
# Add the original repository as upstream
git remote add upstream https://github.com/yourusername/agentbricks.git

# Verify remotes
git remote -v
# Should show:
# origin    https://github.com/YOUR_USERNAME/agentbricks.git (fetch)
# origin    https://github.com/YOUR_USERNAME/agentbricks.git (push)
# upstream  https://github.com/yourusername/agentbricks.git (fetch)
# upstream  https://github.com/yourusername/agentbricks.git (push)
```

#### 4. Create Feature Branch

Always create a new branch for your changes:

```bash
# Sync with upstream first
git fetch upstream
git checkout main
git merge upstream/main

# Create and switch to your feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes:
git checkout -b bugfix/your-bugfix-name
```

**Branch Naming:**
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

#### 5. Make Changes

- Write your code following our [Code Style Requirements](#code-style-requirements)
- Add tests for new functionality
- Update documentation as needed
- Keep changes focused and reviewable

#### 6. Run Tests

Before committing, ensure all tests pass:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_your_module.py

# Run linting
black --check .
ruff check .
mypy .
```

#### 7. Commit with Conventional Commits

Follow our [Commit Message Format](#commit-message-format):

```bash
# Stage your changes
git add .

# Commit with proper format
git commit -m "feat(brick-01): add event validation middleware

Implement Pydantic validation for all event types.
Includes comprehensive error messages and logging.

Closes #42"
```

#### 8. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Then go to GitHub and create a Pull Request
# GitHub will show a "Compare & pull request" button
```

**Before Creating PR:**
- [ ] All tests pass locally
- [ ] Code follows style guide (Black, Ruff, mypy)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)
- [ ] Commit messages follow conventional format
- [ ] Branch is up to date with upstream/main

### Keeping Your Fork Updated

Regularly sync your fork with the main repository:

```bash
# Fetch latest changes from upstream
git fetch upstream

# Switch to main branch
git checkout main

# Merge upstream changes
git merge upstream/main

# Push to your fork
git push origin main

# Update your feature branch
git checkout feature/your-feature-name
git merge main
```

## Setting Up the Development Environment

This section provides detailed instructions for setting up your local development environment. If you've already followed the [Development Workflow](#development-workflow), you can skip to step 2.

### Step 1: Fork and Clone (if not done already)

```bash
# Clone your fork (replace YOUR_USERNAME)
git clone https://github.com/YOUR_USERNAME/agentbricks.git
cd agentbricks

# Add upstream remote
git remote add upstream https://github.com/yourusername/agentbricks.git
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment with Python 3.11+
python3.11 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (should show venv path)
which python  # macOS/Linux
where python  # Windows
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install project dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Verify installation
pip list | grep -E "(pytest|black|ruff|mypy)"
```

### Step 4: Install Pre-commit Hooks

Pre-commit hooks automatically check your code before commits:

```bash
# Install hooks
pre-commit install

# Test hooks (optional)
pre-commit run --all-files
```

**What pre-commit checks:**
- Code formatting (Black)
- Linting (Ruff)
- Type checking (mypy)
- Commit message format
- File permissions

### Step 5: Verify Installation

```bash
# Check all tools are installed
pytest --version
black --version
ruff --version
mypy --version

# Run a quick test
pytest tests/unit/ -v --tb=short

# Check code style
black --check .
ruff check .
```

### Step 6: Generate Test Data (Optional)

```bash
# Generate synthetic data for testing
cd sim
python generate.py generate-all --users 100 --movies 50 --days 1
cd ..
```

### Step 7: Start Infrastructure (Optional)

If you're working on bricks that require infrastructure:

```bash
# Start Docker services
cd stories/movie-recommender/brick-01-data-collection
docker compose up -d

# Verify services are running
docker compose ps

# Check API health
curl http://localhost:8000/health
```

### Troubleshooting Setup

**Issue: Python version not found**
```bash
# Check available Python versions
python3.11 --version
python3.12 --version

# Or install Python 3.11 using pyenv
pyenv install 3.11.5
pyenv local 3.11.5
```

**Issue: Dependencies fail to install**
```bash
# Clear pip cache
pip cache purge

# Upgrade pip and try again
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Issue: Pre-commit hooks not running**
```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Test manually
pre-commit run --all-files
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

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. This ensures clear, consistent commit messages that make it easy to understand what changed and why.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | A new feature | `feat(brick-01): add event validation` |
| `fix` | A bug fix | `fix(sim): correct user preference generation` |
| `docs` | Documentation only changes | `docs(readme): update installation` |
| `style` | Code style changes (formatting, etc.) | `style(api): format with black` |
| `refactor` | Code refactoring without changing functionality | `refactor(features): extract common logic` |
| `perf` | Performance improvements | `perf(serving): optimize retrieval latency` |
| `test` | Adding or updating tests | `test(bricks): add unit tests` |
| `chore` | Maintenance tasks, dependency updates | `chore(deps): update pytest to 7.4.3` |
| `ci` | CI/CD configuration changes | `ci: add security scanning job` |
| `build` | Build system or external dependency changes | `build: update Docker base image` |

### Scope

The scope should indicate the area of the codebase affected:

- **Module name**: `sim`, `bricks`, `api`, `agent`
- **Component name**: `ingestion`, `features`, `training`, `serving`
- **Brick name**: `brick-01`, `brick-02`, etc.
- **Tool name**: `cli`, `docker`, `terraform`

**Scope is optional** but recommended for clarity.

### Subject

- Use imperative mood ("add" not "added" or "adds")
- First letter lowercase
- No period at the end
- Maximum 50 characters (aim for clarity)
- Describe what the commit does, not why

### Body (Optional)

- Explain **what** and **why** vs. **how**
- Wrap at 72 characters
- Can include multiple paragraphs
- Use bullet points for multiple changes
- Reference issues and PRs

### Footer (Optional)

- Reference issues: `Closes #42`, `Fixes #23`, `Related to #10`
- Breaking changes: `BREAKING CHANGE: description`
- Co-authors: `Co-authored-by: Name <email>`

### Examples

#### Feature Addition

```
feat(brick-01): add event validation middleware

Implement Pydantic validation for all event types.
Includes comprehensive error messages and logging.

- Add ViewEvent schema validation
- Add RatingEvent schema validation
- Add error handling middleware
- Update API documentation

Closes #42
```

#### Bug Fix

```
fix(sim): correct user preference generation

Users were getting identical preference vectors due to
improper random seed initialization. Now using proper
random seed per user to ensure unique preferences.

Fixes #23
```

#### Documentation

```
docs(readme): update installation instructions

Add troubleshooting section for Docker issues.
Include common port conflicts and solutions.

Related to #15
```

#### Test Addition

```
test(bricks): add unit tests for feature engineering

Achieve 85% coverage for feature computation module.
Add tests for edge cases including empty data and
temporal boundary conditions.

- Test compute_user_features with new users
- Test compute_movie_features with no interactions
- Test point-in-time correctness
```

#### Breaking Change

```
feat(api): refactor event schema structure

BREAKING CHANGE: Event schemas now use nested structure.
Migration guide available in docs/migration/v2.md.

Old format:
{
  "user_id": "...",
  "movie_id": "..."
}

New format:
{
  "user": {"id": "..."},
  "movie": {"id": "..."}
}
```

#### Multiple Changes

```
feat(brick-02): improve feature store performance

- Add connection pooling for DuckDB
- Implement batch feature updates
- Add feature caching layer
- Optimize temporal queries

Performance improvements:
- 3x faster batch updates
- 50% reduction in query latency

Closes #45
Closes #46
```

### Commit Message Best Practices

✅ **Do:**
- Write clear, descriptive subjects
- Use imperative mood
- Include scope when helpful
- Reference related issues
- Explain why in the body (if not obvious)
- Keep commits focused (one logical change)

❌ **Don't:**
- Write vague messages like "fix bug" or "update code"
- Include unnecessary details about implementation
- Mix multiple unrelated changes
- Forget to reference related issues
- Use past tense ("fixed", "added")
- Write messages longer than 72 characters for subject

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
