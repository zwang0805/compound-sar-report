# Provenance and differences from the original

Extracted on 2026-11-13 from the *mmpa* project (`softwares/mmpa`):

| Here | Original | Change |
|---|---|---|
| `mmpkit/attachment_order.py` | `mmp_attachment_order.py` | none |
| `mmpkit/stats.py` | `mmp_stats.py` | none |
| `mmpkit/frag_index.py` | `mmp_frag_and_index.py` | import path; one regex literal made raw (`r'\[[0-9]*\*\]'`, same pattern, silences a SyntaxWarning) |
| `mmpkit/deposition.py` | `mmp_deposition.py` | import path |
| `mmpkit/sdf.py` | `mmp_sdf.py` | none (added later: the robust SDF reader, see below) |
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

## Later change: SDF reading must not lose records (2026-11-23)

`mmpkit/mmp_core.read_sdf` used `Chem.SDMolSupplier`, which silently drops every
OTHER record of an SDF whose last data-field value is followed directly by `$$$$`
(Actelion/Glide exports write exactly that: `...Yes\n$$$$\n<next title>`). A
3344-record file loaded as 1672 molecules and printed that as if it were the whole
file. `mmpkit/sdf.py` (shared with *mmpa*'s `mmp_sdf.py`) splits the text itself and
parses each record with `MolFromMolBlock`, keeping the SDF properties on the mol so
fragmentation/deposition are unchanged; a record that really cannot be parsed is
counted and reported. Verified: on files that already loaded correctly the molecule
list, order, names and every property value are identical, and the all-levels
dataset built from the 97-molecule test SDF is byte-identical to the pre-change run.

## Later change: designs from double/triple-bond cuts (2026-11-23)

`mmpkit/report.py` gained the same change as *mmpa*'s `mmp_report.py`, so the two
are still identical apart from imports:

- designs may now come from double/triple-bond cut positions (whole-fragment
  swap), not only single-bond cuts;
- the attachment pairing is read off each pair (dummy order is not a convention
  in the deposits), required to be unambiguous across all pairings consistent
  with the compound, and every result is re-cut to confirm it returns the same
  core and fragment; anything that fails is skipped and counted in
  `multi_designs_dropped`;
- the headline "best-supported swap" is now taken from the design list itself,
  so it always carries a `design_id` (D1) instead of advertising a swap the
  report never builds.

Re-verified after the change: skill report vs *mmpa* report on the same dataset
and compound = identical output; 32 emitted multi-cut designs across six
datasets all passed the re-cut check (the check is now part of generation).
