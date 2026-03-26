# Contributing to space-ml-sim

Thanks for your interest in contributing. This document covers the development workflow, standards, and policies.

## Getting Started

```bash
git clone https://github.com/yaitsmesj/space-ml-sim.git
cd space-ml-sim
pip install -e ".[dev]"
pre-commit install
```

## Development Workflow

1. **Fork and branch** -- Create a feature branch from `main`
2. **Write tests first** -- We follow TDD. Write a failing test, then implement
3. **Run checks locally** before pushing:

```bash
pytest tests/ -v --cov=space_ml_sim --cov-fail-under=80   # Tests + coverage
ruff check src/ tests/                                      # Linting
ruff format src/ tests/                                     # Formatting
bandit -r src/ -c pyproject.toml -ll                        # Security
```

4. **Commit with conventional messages** -- `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
5. **Open a PR** -- Fill out the PR template. CI must pass before merge

## Code Standards

- **Type hints** on all public methods
- **Docstrings** on all classes and public methods
- **Immutability** -- Return new objects, don't mutate inputs
- **Small files** -- Under 400 lines typical, 800 max
- **Small functions** -- Under 50 lines
- **80%+ test coverage** -- Enforced in CI

## What Makes a Good PR

- Solves one problem (don't bundle unrelated changes)
- Tests cover the new behavior
- No regressions (all existing tests pass)
- Follows existing code patterns and naming conventions
- Performance-sensitive code includes benchmark evidence

## Security

- Never commit secrets, API keys, or credentials
- Run `bandit` before submitting security-sensitive changes
- Report vulnerabilities privately -- see [SECURITY.md](SECURITY.md)

## Licensing

This project is dual-licensed under AGPL-3.0 and a commercial license. By contributing, you agree that your contributions may be distributed under both licenses. See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md#contributor-license-agreement) for details.

## Review Process

- All PRs require at least one review from a maintainer
- CI must pass (tests, lint, security, benchmarks)
- Changes to core modules (fault injection, radiation, orbit) require maintainer review (enforced by CODEOWNERS)
