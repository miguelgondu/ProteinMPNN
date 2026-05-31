.PHONY: install lint format test typecheck build clean release-dry-run release release-test

# Install all dependencies including dev
install:
	uv sync --all-groups

# Run linter
lint:
	uv run ruff check

# Format code
format:
	uv run ruff format

# Run tests
test:
	uv run pytest

# Run type checker
typecheck:
	uv run pyright

# Build package
build:
	uv build

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Test release process without publishing
release-dry-run:
	./scripts/release.sh --dry-run

# Full release to PyPI
release:
	./scripts/release.sh

# Release to TestPyPI
release-test:
	./scripts/release.sh --test-pypi
