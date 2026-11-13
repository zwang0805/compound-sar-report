# compound-sar-report

A [Claude Code](https://claude.com/claude-code) skill that builds a self-contained HTML SAR report for one compound
from a set of molecules with measured properties (matched molecular pair analysis, MMP).

Input: a CSV (SMILES + numeric property columns) or an SDF, and the SMILES of the compound of interest.
Output: an all-levels MMP dataset (CSV) and an HTML report (SAR overview figure, site map, direct analogues,
per-position transformation statistics, design proposals, master molecule list).

Needs only `rdkit`, `pandas`, `numpy`, `scipy` (`pip install -r requirements.txt`). Usage, options and limitations
are in [SKILL.md](SKILL.md); provenance of the vendored code is in [reference/NOTES.md](reference/NOTES.md).

```bash
cd scripts
python sar_report.py run --input molecules.csv --props "pIC50" --smiles "<SMILES>" --out-dir out/
```

To use it as a Claude Code skill, put this directory under `~/.claude/skills/compound-sar-report/`.
