#!/usr/bin/env bash
set -euo pipefail

# Release script for proteinmpnn-cli
# Usage: ./scripts/release.sh [--dry-run] [--test-pypi]

DRY_RUN=false
TEST_PYPI=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --test-pypi)
            TEST_PYPI=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./scripts/release.sh [--dry-run] [--test-pypi]"
            exit 1
            ;;
    esac
done

echo "=== ProteinMPNN Release Script ==="
echo ""

# Get version from pyproject.toml
VERSION=$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2)
echo "Version: $VERSION"

# Pre-flight checks
echo ""
echo "=== Pre-flight Checks ==="

# Check for clean working directory
if [[ -n $(git status --porcelain) ]]; then
    echo "ERROR: Working directory is not clean. Commit or stash changes first."
    exit 1
fi
echo "✓ Working directory is clean"

# Check we're on main branch
BRANCH=$(git branch --show-current)
if [[ "$BRANCH" != "main" ]]; then
    echo "ERROR: Not on main branch (currently on: $BRANCH)"
    exit 1
fi
echo "✓ On main branch"

# Check up to date with remote
git fetch origin main
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
if [[ "$LOCAL" != "$REMOTE" ]]; then
    echo "ERROR: Local branch is not up to date with origin/main"
    exit 1
fi
echo "✓ Up to date with remote"

# Check if tag already exists
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo "ERROR: Tag v$VERSION already exists"
    exit 1
fi
echo "✓ Tag v$VERSION does not exist"

# Run quality checks
echo ""
echo "=== Running Quality Checks ==="

echo "Running linter..."
uv run ruff check
echo "✓ Lint passed"

echo "Running type checker..."
uv run pyright
echo "✓ Type check passed"

echo "Running tests..."
uv run pytest
echo "✓ Tests passed"

# Build package
echo ""
echo "=== Building Package ==="
rm -rf dist/
uv build
echo "✓ Package built"

# Show what would be published
echo ""
echo "=== Package Contents ==="
ls -la dist/

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "=== DRY RUN MODE ==="
    echo "Would create tag: v$VERSION"
    echo "Would publish to: $([ "$TEST_PYPI" == true ] && echo 'TestPyPI' || echo 'PyPI')"
    echo ""
    echo "To perform the actual release:"
    echo "  1. Create tag: git tag -a v$VERSION -m 'Release v$VERSION'"
    echo "  2. Push tag: git push origin v$VERSION"
    echo "  3. Run: ./scripts/release.sh $([ "$TEST_PYPI" == true ] && echo '--test-pypi')"
    exit 0
fi

# Publish
echo ""
echo "=== Publishing ==="

if [[ "$TEST_PYPI" == true ]]; then
    echo "Publishing to TestPyPI..."
    uv publish --index https://test.pypi.org/simple/
    echo ""
    echo "✓ Published to TestPyPI"
    echo ""
    echo "To verify:"
    echo "  pip install -i https://test.pypi.org/simple/ proteinmpnn-cli"
    echo "  proteinmpnn --help"
else
    echo "Publishing to PyPI..."
    uv publish
    echo ""
    echo "✓ Published to PyPI"
    echo ""
    echo "To verify:"
    echo "  pip install proteinmpnn-cli"
    echo "  proteinmpnn --help"
fi

echo ""
echo "=== Release Complete ==="
