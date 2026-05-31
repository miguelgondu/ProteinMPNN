"""Model weights management with on-demand downloading."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Literal

from proteinmpnn.utils.logging import get_logger

logger = get_logger("utils.weights")

# GitHub raw URL for weights
GITHUB_WEIGHTS_URL = (
    "https://github.com/miguelgondu/ProteinMPNN/raw/main/run/model_weights"
)

# All available model weights with their SHA256 checksums
# Checksums ensure download integrity
MODEL_CHECKSUMS: dict[str, str] = {
    "v_48_002.pt": "925f2ca1007bf9b02e0e7f420ff00eb91f50fcc2722f64b42e644ae95adaa131",
    "v_48_010.pt": "db866fae956a28661f926053d630610c55e9fc4bc03922f2aeeb98a37435ccce",
    "v_48_020.pt": "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd",
    "v_48_030.pt": "c34b7bfb38418ea30989fda3314f4781ac4e3920f9825731cf555f1fed44ac66",
    "ca_48_002.pt": "ec038b44a987d7c8351b6ed887c82a2370d54e45e55a6bdaf508a729cef0340e",
    "ca_48_010.pt": "cdb50498d45578d20b271fa7817b8cd8bfde3875ad69dbd3f5e4b5dd3e588301",
    "ca_48_020.pt": "f28f40170e21858c5ff31ef50b6e63414ff76dc331b19f85aa8586a12031744a",
    "s_48_002.pt": "0877f840978fe770be6fcec025784d8f50c438571db3260c05e41aa207a7c448",
    "s_48_010.pt": "79562f7444f72c84595a1c96010713864865a616f4f3967633493041e169fa6e",
    "s_48_020.pt": "7af52d090172c230c7f0e9d21e02203f6b3a38b16db58d3c7a3960e0a9a6e31a",
    "s_48_030.pt": "1dd63f1e9fc68a133cc9ef859edf43b489e5ac581cb5624e0b9ec848ff062421",
}

# Type alias for model names
ModelName = Literal[
    "v_48_002",
    "v_48_010",
    "v_48_020",
    "v_48_030",
    "ca_48_002",
    "ca_48_010",
    "ca_48_020",
    "s_48_002",
    "s_48_010",
    "s_48_020",
    "s_48_030",
]


def get_cache_dir() -> Path:
    """Get the cache directory for model weights.

    Uses ~/.cache/proteinmpnn by default. Can be overridden by
    setting the PROTEINMPNN_CACHE environment variable.

    Returns:
        Path to the cache directory.
    """
    import os

    cache_dir = os.environ.get("PROTEINMPNN_CACHE")
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / ".cache" / "proteinmpnn" / "weights"


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def _download_with_progress(url: str, dest: Path, desc: str) -> None:
    """Download a file with progress logging.

    Args:
        url: URL to download from.
        dest: Destination path.
        desc: Description for logging.
    """
    logger.info("Downloading %s...", desc)
    logger.debug("URL: %s", url)
    logger.debug("Destination: %s", dest)

    # Create parent directory if needed
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temporary file first
    temp_dest = dest.with_suffix(".tmp")

    try:
        # Open URL and get content length
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            last_percent = -1

            with temp_dest.open("wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Log progress at 10% intervals
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        if percent >= last_percent + 10:
                            logger.info(
                                "  %s: %d%% (%d / %d bytes)",
                                desc,
                                percent,
                                downloaded,
                                total_size,
                            )
                            last_percent = percent

        # Move to final destination
        shutil.move(temp_dest, dest)
        logger.info("Downloaded %s successfully (%.2f MB)", desc, downloaded / 1e6)

    except Exception:
        # Clean up temp file on failure
        if temp_dest.exists():
            temp_dest.unlink()
        raise


def get_weights_path(model_name: ModelName) -> Path:
    """Get the expected cache path for model weights (pure function).

    This does NOT download anything. Use `download_weights` to download,
    or `ensure_weights` to download if needed and get the path.

    Args:
        model_name: Name of the model (e.g., "v_48_020").

    Returns:
        Path where the model weights would be cached.

    Raises:
        ValueError: If the model name is not recognized.
    """
    filename = f"{model_name}.pt"

    if filename not in MODEL_CHECKSUMS:
        available = ", ".join(n.replace(".pt", "") for n in MODEL_CHECKSUMS)
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")

    return get_cache_dir() / filename


def download_weights(model_name: ModelName, verify_checksum: bool = True) -> Path:
    """Download model weights from GitHub.

    Downloads to the cache directory. If weights already exist, this is a no-op.

    Args:
        model_name: Name of the model (e.g., "v_48_020").
        verify_checksum: Whether to verify the checksum after download.

    Returns:
        Path to the downloaded weights file.

    Raises:
        ValueError: If the model name is not recognized.
        RuntimeError: If download fails or checksum doesn't match.
    """
    weights_path = get_weights_path(model_name)

    # Already downloaded
    if weights_path.exists():
        logger.debug("Weights already cached at %s", weights_path)
        return weights_path

    filename = f"{model_name}.pt"
    url = f"{GITHUB_WEIGHTS_URL}/{filename}"

    try:
        _download_with_progress(url, weights_path, f"model weights ({model_name})")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to download model weights from {url}. "
            f"Check your internet connection. Error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to download model weights: {e}") from e

    # Verify checksum
    if verify_checksum:
        logger.debug("Verifying checksum...")
        computed = _compute_sha256(weights_path)
        expected = MODEL_CHECKSUMS[filename]

        if computed != expected:
            # Remove corrupted file
            weights_path.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {filename}. "
                f"Expected {expected[:16]}..., got {computed[:16]}... "
                "The file may be corrupted. Please try again."
            )
        logger.debug("Checksum verified successfully")

    return weights_path


def _get_repo_weights_path(model_name: str) -> Path | None:
    """Check if weights exist in the repository (for development use).

    Args:
        model_name: Name of the model.

    Returns:
        Path to weights in repo if they exist, None otherwise.
    """
    # Import here to avoid circular imports
    from proteinmpnn.utils.constants import ROOT_DIR

    repo_path = ROOT_DIR / "run" / "model_weights" / f"{model_name}.pt"
    if repo_path.exists():
        return repo_path
    return None


def ensure_weights(model_name: ModelName) -> Path:
    """Ensure weights exist, downloading if necessary, and return the path.

    Checks in order:
    1. Cache directory (~/.cache/proteinmpnn/weights/)
    2. Repository directory (for development)
    3. Downloads from GitHub if not found

    Args:
        model_name: Name of the model to ensure.

    Returns:
        Path to the model weights file.
    """
    # First check cache
    cache_path = get_weights_path(model_name)
    if cache_path.exists():
        logger.debug("Found weights in cache: %s", cache_path)
        return cache_path

    # Check repo path (for development)
    repo_path = _get_repo_weights_path(model_name)
    if repo_path is not None:
        logger.debug("Found weights in repository: %s", repo_path)
        return repo_path

    # Download to cache
    download_weights(model_name)
    return cache_path


def list_cached_weights() -> list[str]:
    """List all model weights currently in the cache.

    Returns:
        List of model names that are cached.
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return []

    cached = []
    for filename in MODEL_CHECKSUMS:
        if (cache_dir / filename).exists():
            cached.append(filename.replace(".pt", ""))
    return cached


def clear_cache() -> None:
    """Remove all cached model weights."""
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("Cleared weights cache at %s", cache_dir)
    else:
        logger.info("Cache directory does not exist: %s", cache_dir)
