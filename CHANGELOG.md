# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-05-31

### Added

- **CLI commands**:
  - `proteinmpnn run-single` - Run inference for a single PDB file with customizable parameters
  - `proteinmpnn compute-probs` - Compute conditional and unconditional amino acid probabilities per position
- **Python API**:
  - `InferenceRunner` - Main class for running ProteinMPNN inference
  - `DesignResult` and `SequenceResult` - Structured result types
- **Developer tooling**:
  - `uv` for dependency and package management
  - `typer` for CLI with rich help text
  - `pytest` for unit testing with backwards compatibility tests
  - `ruff` for linting and formatting
  - `pyright` for type checking
  - PEP 561 type marker (`py.typed`)

### Notes

- Requires Python 3.13+
- Install with `pip install proteinmpnn-cli`, run with `proteinmpnn` command
- Based on [Kuhlman Lab's fork](https://github.com/Kuhlman-Lab/proteinmpnn) of the original [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
