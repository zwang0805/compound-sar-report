---
name: compound-sar-report
description: >-
  Build a self-contained HTML SAR report for ONE compound from a set of
  molecules with measured properties: fragments every molecule, finds matched
  molecular pairs (MMP), then writes the report — SAR overview figure (the
  compound in the centre, one coloured region per cut position listing every
  R-group seen there), site map, direct analogues, per-position transformation
  statistics, rule-based design proposals (single-point swaps, whole-fragment
  swaps at double/triple-bond cuts, and additive combinations) and a master
  molecule list. Use this whenever the user gives a
  compound (SMILES) and a table/SDF of molecules with activity/ADME values and
  wants its SAR, an MMP/SAR report, "给定一个分子提取SAR"、"生成化合物SAR报告"、
  "做MMP分析并出报告"、"这个分子哪些位置可以优化". Standalone: needs only
  rdkit/pandas/numpy/scipy, no server, no LLM, no other project files.
---

# Compound SAR report

Takes a CSV (SMILES + name + numeric property columns) or an SDF, runs the full
MMP pipeline in-process, and writes one HTML report about a chosen compound.
Deterministic: numbers come from the data, wording from fixed templates.

The code in `scripts/mmpkit/` was extracted unchanged from the original *mmpa*
project (fragmentation, core abstraction levels 0-5, deposition, report) and on
the same input produced a **byte-identical** dataset and report (checked on a
250-molecule subset of the AZ microsomal-clearance set). `scripts/mmpkit/report.py`
is still line-for-line identical to *mmpa*'s `mmp_report.py` except for its two
imports, so both stay in sync when either is changed.

## Quick start

```bash
cd <this skill>/scripts
python sar_report.py run \
  --input molecules.csv --props "pIC50,Clearance" --direction "Clearance=lower" \
  --smiles "<SMILES of the compound>" --out-dir out/
# -> out/<name>_alllevels.csv (the MMP dataset) and out/compound_SAR_report.html
```

Use an environment that has rdkit, pandas, numpy, scipy (`requirements.txt`).
On this machine: `/data/zhen.wang/softwares/miniconda3/envs/mmpa/bin/python`.

Two steps if you want to reuse one dataset for several compounds:

```bash
python sar_report.py build  --input molecules.csv --props "pIC50" --out-dir out/ --job myset
python sar_report.py report --dataset out/myset_alllevels.csv --smiles "<SMILES>" --out a.html
python sar_report.py report --dataset out/myset_alllevels.csv --smiles "<SMILES 2>" --out b.html
```

## Inputs

| Option | Meaning |
|---|---|
| `--input` | CSV with a SMILES column (any case; or `--smiles-col`), optional `Name`/`ID` column (or `--name-col`), and numeric property columns; or an SDF whose property tags are the properties |
| `--props` | Comma-separated property columns/tags. **First = primary** (drives ranking and design grades). Required for `build`/`run`; optional for `report` (default: all in the dataset) |
| `--direction` | `"Clearance=lower,pIC50=higher"`. Default is *higher is better* for everything — set `lower` for IC50/Kd/clearance-type values, otherwise "improvement" is read backwards |
| `--smiles` | The compound the report is about. It need not be in the dataset (then there are no direct analogues or measured values, only position-level statistics) |
| `--max-pool-level` | **Default 1.** How far a cut position may be widened to pool pairs from other series: 0 = identical core only, 1 = ring system kept / substituents stripped. Levels 2+ are atom-environment fingerprints that are no longer readable structures and pool far more loosely, so 1 is the recommended ceiling |
| `--lang` | `zh` (default) or `en` |
| `--max-designs`, `--improve-threshold`, `--worsen-tolerance` | Design-proposal knobs (defaults 8, 0.1, 0.3, in the property's own units) |
| `--workers` | Processes for the fragmentation step (default 3) |
| `--max-molecules` | Safety cap (default 5000): every molecule is cut at every bond combination, so big sets get slow |

Rows whose SMILES cannot be parsed are skipped and counted on stderr. A
property value that is not a number is treated as missing for that molecule.

## What the report contains

1. Key findings  2. SAR overview figure (regions around the structure, lines in
the matching colour)  3. Site map  4. Transformations by position  5. Direct
analogues  6. Proposed designs  7. Limitations  8. Master molecule list
(clicking a fragment card in section 2 filters it).

The command prints a JSON summary (measured values/rank, number of analogues
and positions with data, best-supported swap with its `cut_type` and
`design_id`, designs with grade/kind/cut_type, and `multi_designs_by_check` /
`multi_designs_dropped` for the double/triple-cut ones) — use it to answer the
user in a few sentences and point to the HTML file. The headline swap is always
`design_id` D1, so it can never advertise a swap the report does not deliver.

## Things to tell the user (inherited from the original, not fixed here)

- Grades: A = ≥3 pairs all the same sign, B = ≥2 pairs, C = a single pair.
  Combination designs assume the two effects add up.
- **Designs now also come from double/triple-cut positions**, where a whole
  fragment touching 2-3 attachment points is replaced at once. Those designs are
  only emitted when the attachment pairing is unambiguous: the pairing is read
  off the compound itself, every consistent pairing must give the same molecule,
  and the result is re-cut to confirm it gives back the same core + fragment.
  Pairs that fail any of that are skipped and counted (`multi_designs_dropped` in
  the JSON). Say so when reporting: the count can be a large share on some
  datasets, and the grade of a multi-cut design still reflects only how many
  pairs back it, not how many atoms change.
- **Measured values need the absolute values in the data.** Datasets built by
  this skill always carry them; an older dataset without `mol1_value*` columns
  gives deltas only (no ranks, no predicted absolute values).
- The HTML is large for rich compounds (several MB; the master molecule list
  embeds one SVG per molecule).
- Small data = weak statistics: most swaps rest on 1-3 pairs and have no
  significance test; the report says so in section 7.

## Layout

```
scripts/sar_report.py        CLI (build / report / run)
scripts/mmpkit/mmp_core.py   SDF I/O + core abstraction levels 0-5
scripts/mmpkit/sdf.py       SDF record reader (robust: no silent half-loss on Glide exports)
scripts/mmpkit/frag_index.py single/double/triple-cut fragmentation and pair finding
scripts/mmpkit/attachment_order.py  attachment-point alignment for multi-cut pairs
scripts/mmpkit/deposition.py all-levels dataset extraction
scripts/mmpkit/stats.py      paired-difference statistics
scripts/mmpkit/report.py     the report generator (compound report only)
reference/NOTES.md           provenance and differences from the original
```
