# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `StructureDatasetPDB.from_pdb_file(path, chains=None)`, which parses one file instead of
  every `*.pdb` beside it, optionally restricted to some of its chains.
- `compute_probs(..., chain="A")` restricts parsing and output to one chain. Previously the
  result always covered every chain in the file.
- `compute_probs(..., n_draws=N)` draws N decoding orders **after a single seeding** and
  averages the log probabilities, which is what upstream's `--num_seq_per_target N` does.
  `ConditionalProbsResult.n_draws` records it. Note that draw *k* depends on the RNG state
  left by draw *k-1*, so N draws under one seed are not reproducible by N separately
  seeded single draws; `(seed, n_draws)` together identify the sample. Rejected for
  unconditional mode with `backbone_noise=0`, where every draw would be identical.

### Fixed

- **`compute_probs` no longer drops rows, and no longer mislabels the rest.** Residues whose
  backbone is incomplete (any of N, CA, C, O missing), including the placeholder rows
  `parse_PDB` inserts across gaps in author numbering, were removed from `log_probs` while
  `residue_info` was rebuilt as a running `1..n` count. Every row after the first such
  residue was therefore attributed to the wrong residue, and the result carried no way to
  detect it. Affects any structure with a numbering gap or an incomplete residue;
  single-chain predicted structures with complete backbones were unaffected. Upstream
  ProteinMPNN's `--conditional_probs_only` never had this problem: it emits every row and
  ships the mask alongside. See
  `notes/2026_11_17__0001_conditional_probs_row_dropping.md`.

### Changed

- **Breaking:** `ConditionalProbsResult` now has one row per *parsed* residue and gains two
  required fields: `observed` (`bool[L]`, the mask the model used) and `native_seq` (the
  parsed sequence, `X` on unobserved rows). Log probabilities on unobserved rows are passed
  through rather than NaN-ed, matching upstream, so **mask with `observed` before use**.
- **Breaking:** `to_csv` gains an `observed` column (third, before the alphabet), and
  `to_npz_dict` gains `observed` and `native_seq` keys.
- `parse_PDB_biounits` now parses with gemmi, with the line-based and biopython
  implementations kept alongside it for comparison.
- `compute_probs` parses only the PDB it was given. It previously called
  `from_pdb_dir(pdb_path.parent)` and parsed every structure in that directory, which is
  slow and surprising when the directory is a shared or content-addressed store.

### Known limitations

- `ResidueInfo.residue_idx` is still a per-chain running index rather than author numbering
  (`resseq`/`icode`), so a row cannot be mapped back to the PDB without re-parsing. Fixing
  that means threading numbering out of `parse_PDB`, and is tracked separately.

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
