# Provenance and differences from the original

Extracted on 2026-11-13 from the *mmpa* project (`softwares/mmpa`):

| Here | Original | Change |
|---|---|---|
| `mmpkit/attachment_order.py` | `mmp_attachment_order.py` | none |
| `mmpkit/stats.py` | `mmp_stats.py` | none |
| `mmpkit/frag_index.py` | `mmp_frag_and_index.py` | import path; one regex literal made raw (`r'\[[0-9]*\*\]'`, same pattern, silences a SyntaxWarning) |
| `mmpkit/deposition.py` | `mmp_deposition.py` | import path |
| `mmpkit/mmp_core.py` | `main.py` (class `MMP`) | only `read_sdf`, `_filter_molecule`, `process_core_structures` and its helpers, `save_molecules` kept; LLM/suggestion/network parts dropped; `_move_to_history` is a no-op (the intermediate MMP SDF stays in the output dir) |
| `mmpkit/report.py` | `mmp_report.py` | `build_compound_report` and its helpers only (`build_core_report`, which needs the Free-Wilson engine, is not included); two import paths |
| `sar_report.py` | `server.py` / `mmp_tools.py` pipeline glue | new: CSV→SDF, property-name sanitising (`subtract_<name>_mmp`), spec building, CLI |

## Verification done

250 random molecules of the AZ microsomal-clearance set, property `Y`
(lower is better):

- all-levels dataset from this skill vs the original pipeline: same columns,
  same 234 rows (order-insensitive);
- HTML report for the most-paired compound, from the same dataset: **byte-identical**
  to the original `build_compound_report` output;
- `run` from an SDF with `--lang en --max-pool-level 0` also completes.

Not tested: datasets above a few thousand molecules (runtime), SDFs with
disconnected salts, multi-property datasets in `report` beyond reading the columns.
