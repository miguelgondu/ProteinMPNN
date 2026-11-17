---
author: Claude Opus 5
date: 2026-11-17
---

# `compute_probs` drops rows and then mislabels the rest

Found while building the structure/alignment layer in `kermut-prefect`, which needs
`compute_probs` output to line up row-for-row with its own parse of the same PDB.

**This one is ours, not upstream's.** That is the main thing this note records, because the
first guess was the opposite.

## The bug

`InferenceRunner._compute_probs_internal` (`src/proteinmpnn/inference/runner.py`):

```python
mask_np = mask[0].cpu().numpy()          # 385
valid_indices = mask_np > 0              # 388
log_probs_valid = log_probs_np[valid_indices]   # 389  <- rows disappear here
```

`mask` comes from `tied_featurize` (`src/proteinmpnn/model/featurize.py:359`), which is
`isfinite(sum(X))` over the four backbone atoms. So a row is dropped when the residue is
missing **any** of N, CA, C, O — which includes:

- gap rows invented by `parse_PDB` to fill a break in author numbering (NaN coords), and
- any real residue with an incomplete backbone.

Dropping them would be defensible on its own. The problem is what happens next
(`runner.py:397-406`): `residue_info` is rebuilt by walking the *parsed* chain sequence and
counting `1..n`, while `log_probs_valid` has already had rows removed. The two are then
zipped together. **From the first dropped row onward, every log-prob row is attributed to
the wrong residue**, and the error grows with each further gap. The returned
`ResidueInfo.residue_idx` is a running index, not author numbering, so a caller cannot even
detect the slip from the result alone.

Introduced in `a360448` (2026-05-30, "feat: refactors inference for single-protein CLI"),
i.e. by our own single-protein refactor.

## Reproduction

A synthetic 12-residue chain with author numbering `1-6, 9-14` (a two-residue gap) and one
residue missing its N atom:

| | rows |
|---|---|
| `parse_PDB` / our `COORDS` | 14 (12 residues + 2 gap rows) |
| `compute_probs().log_probs` | **11**, labelled `residue_idx = 1..11` |

14 - 2 gap rows - 1 incomplete residue = 11. Every ProteinGym AF2 structure is a single
chain with contiguous numbering and complete backbones, so nothing is ever dropped there.
That is why this has not bitten us yet, and it is luck rather than design: a crystal
structure with a disordered loop hits it immediately.

## Upstream is not affected

Checked against the vendored copy in `run/protein_mpnn/protein_mpnn_run.py:237`:

```python
np.savez(conditional_probs_only_file, log_p=concat_log_p, S=S[0,].cpu().numpy(),
         mask=mask[0,].cpu().numpy(), design_mask=mask_out)
```

Upstream writes `log_p` for **all** L rows and ships `mask` alongside it. Row `i` of
upstream's output is row `i` of `parse_PDB`'s output, unconditionally, and the consumer is
handed the mask to decide what to do with unusable rows. There is no misattribution and
nothing to report: the `--conditional_probs_only` path is correct as written.

So **no upstream issue is warranted for this finding.** If we raise anything upstream later,
it would be the much weaker documentation point below, and not this.

## Deferred: the one upstream-level candidate

Worth an issue only if it costs someone else real time:

- `parse_PDB` silently invents rows to fill author-numbering gaps (`-`, NaN coords), so the
  row count is `max - min + 1` of author numbering, not the number of residues in the file.
  The npz carries `S` and `mask` but **no author numbering**, so mapping a row back to a
  residue number requires re-parsing the PDB with exactly the same conventions. Documenting
  "row index is not a residue number, and some rows are synthetic" would be a kindness.
- A consumer that ignores `mask` gets numbers at gap rows that mean nothing. `kermut-og`'s
  `extract_ProteinMPNN_probs.py` does exactly that. Again: upstream handed it the mask, so
  this is the consumer's bug.

Not urgent, and not blocking anything of ours. Revisit if we end up filing something upstream
anyway.

`LigandMPNN` has a separate parser and output path (`torch.save` of a stats dict, no
`isfinite` backbone masking in its top-level modules) and was **not** audited for an
equivalent problem.

## The fix

Tracked as **F2** in `kermut-prefect`'s
`notes/2026_11_17__0005_phase_3_primitives_plan.md`, together with F1 (commit the gemmi
parser, which currently exists only in this repo's working tree), F3 (single-file input) and
F4 (chain selection). F2 is:

- `log_probs` keeps **one row per parsed row**, `[L_parse, 21]`, NaN where masked;
- a `bool[L_parse]` `observed` array carrying the mask the model actually used;
- `native_seq`, the parsed sequence with `X` at gap rows;
- `ResidueInfo` carrying `chain` plus author `resseq` and `icode`, replacing the running
  index.

That restores upstream's "all rows plus a mask" contract while keeping the typed result, and
it makes the row space verifiable against an independent parse — which is what
`kermut-prefect` asserts before it stores anything derived from this output.
