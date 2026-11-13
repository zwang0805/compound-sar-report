#!/usr/bin/env python
"""
Compound SAR report, end to end and self-contained.

  build   molecules (CSV with SMILES / SDF)  ->  all-levels MMP dataset (CSV)
  report  all-levels MMP dataset + one SMILES ->  self-contained HTML SAR report
  run     build + report in one go

Only needs rdkit, pandas, numpy, scipy (see ../requirements.txt) — everything
else is in ./mmpkit, extracted from the original mmpa project.

Examples
  python sar_report.py run --input mols.csv --props "pIC50,Clearance" \\
      --direction "Clearance=lower" --smiles "CCOc1ccccc1N" --out-dir out/
  python sar_report.py build  --input mols.sdf --props "pIC50" --out-dir out/ --job myset
  python sar_report.py report --dataset out/myset_alllevels.csv --smiles "..." --out report.html
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


# ---------------------------------------------------------------------------
# property naming — same conventions as the original deposition pipeline
# ---------------------------------------------------------------------------
def sanitize_prop(p):
    """'IC50 (uM)' -> 'subtract_IC50_uM_mmp' (the delta-column token)."""
    return 'subtract_' + p.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '') + '_mmp'


def clean_label(raw):
    """'subtract_JAK2_pKi_mmp' -> 'JAK2 pKi'."""
    cleaned = re.sub(r'^(subtract|divide)_', '', str(raw))
    cleaned = re.sub(r'_mmp$', '', cleaned)
    return cleaned.strip('_').replace('_', ' ')


def make_prop_spec(key, is_primary=False, value_col=None, direction='higher', label=None):
    return {'key': key, 'label': label or clean_label(key), 'direction': direction,
            'is_primary': is_primary, 'value_col': value_col,
            'stat_prefix': '' if is_primary else f'{key}__', 'weight': 1.0}


def parse_direction(text):
    """'Clearance=lower,pIC50=higher' -> {'clearance': 'lower', 'pic50': 'higher'} (lower-cased keys)."""
    out = {}
    for part in (text or '').split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            v = v.strip().lower()
            if v not in ('higher', 'lower'):
                sys.exit(f"--direction: '{v}' must be 'higher' or 'lower'")
            out[k.strip().lower()] = v
    return out


# ---------------------------------------------------------------------------
# build: molecules -> all-levels MMP CSV
# ---------------------------------------------------------------------------
def _csv_to_sdf(csv_path, sdf_path, smiles_col, name_col, props):
    df = pd.read_csv(csv_path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    smiles_col = smiles_col or cols.get('smiles')
    if smiles_col not in df.columns:
        sys.exit(f"SMILES column not found; columns are: {list(df.columns)} (use --smiles-col)")
    missing = [p for p in props if p not in df.columns]
    if missing:
        sys.exit(f"property column(s) not in the CSV: {missing}; columns are: {list(df.columns)}")
    name_col = name_col or cols.get('name') or cols.get('id')
    writer = Chem.SDWriter(sdf_path)
    n_ok = n_bad = 0
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is None:
            n_bad += 1
            continue
        nm = row[name_col] if name_col in df.columns else None
        mol.SetProp('_Name', str(nm) if nm is not None and str(nm) not in ('', 'nan') else f'mol_{i + 1}')
        for p in props:
            v = pd.to_numeric(row[p], errors='coerce')
            if pd.notna(v):
                mol.SetProp(p, str(float(v)))
        writer.write(mol)
        n_ok += 1
    writer.close()
    print(f'CSV -> SDF: {n_ok} molecules written, {n_bad} unparseable SMILES skipped', file=sys.stderr)
    return n_ok


def _prepare_sdf(src, dst, name_col):
    """Copy an SDF, giving every molecule a name: the tag named by --name-col when
    given, else its title line when that is not empty, else mol_<n>. (The pair
    records identify molecules by name, so blank titles would make every pair
    anonymous.) Warns about duplicate names."""
    writer = Chem.SDWriter(dst)
    names, n_ok, n_bad = [], 0, 0
    for i, mol in enumerate(Chem.SDMolSupplier(src), 1):
        if mol is None:
            n_bad += 1
            continue
        nm = None
        if name_col and mol.HasProp(name_col) and mol.GetProp(name_col).strip():
            nm = mol.GetProp(name_col).strip()
        elif mol.HasProp('_Name') and mol.GetProp('_Name').strip():
            nm = mol.GetProp('_Name').strip()
        mol.SetProp('_Name', nm or f'mol_{i}')
        names.append(mol.GetProp('_Name'))
        writer.write(mol)
        n_ok += 1
    writer.close()
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        print(f'WARNING: {len(dup)} duplicate molecule name(s), e.g. {dup[:5]} — pairs involving them '
              f'cannot be told apart by name; pick a unique --name-col', file=sys.stderr)
    print(f'SDF: {n_ok} molecules read, {n_bad} unparseable skipped', file=sys.stderr)
    return n_ok


def _sdf_prop_tags(sdf_path):
    tags = {}
    for mol in Chem.SDMolSupplier(sdf_path):
        if mol is None:
            continue
        for k in mol.GetPropNames():
            if not k.startswith('_'):
                tags[k] = tags.get(k, 0) + 1
    return tags


def build_dataset(args):
    from mmpkit import frag_index
    from mmpkit.mmp_core import MMP

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    job = args.job or os.path.splitext(os.path.basename(args.input))[0]
    job = re.sub(r'[^A-Za-z0-9_.-]+', '_', job)
    props = [p.strip() for p in (args.props or '').split(',') if p.strip()]
    sdf_name = f'{job}.sdf'
    sdf_path = os.path.join(out_dir, sdf_name)

    if args.input.lower().endswith(('.sdf', '.sd')):
        if not props:
            tags = _sdf_prop_tags(args.input)
            sys.exit('--props is required; property tags found in the SDF: ' + (', '.join(tags) or '(none)'))
        n_mols = _prepare_sdf(args.input, sdf_path, args.name_col)
    else:
        if not props:
            sys.exit('--props is required (comma-separated numeric column names)')
        n_mols = _csv_to_sdf(args.input, sdf_path, args.smiles_col, args.name_col, props)
    if n_mols < 2:
        sys.exit('need at least 2 parseable molecules to form matched pairs')
    if n_mols > args.max_molecules:
        sys.exit(f'{n_mols} molecules exceeds --max-molecules {args.max_molecules}; fragmenting every '
                 f'molecule is slow on big sets — raise the limit if you really want this.')

    frag_index.MMP_ANALYSIS_WORKERS = max(1, args.workers)
    mmp = MMP(working_directory=out_dir)
    print(f'[1/2] fragmenting {n_mols} molecules and finding matched pairs ...', file=sys.stderr)
    mmps = mmp.comparison.compare_molecules(sdf_name, job)
    print(f'      {len(mmps) if mmps else 0} matched pairs', file=sys.stderr)
    mmp_sdf = f'sdfs/{job}_mmp.sdf'
    if not os.path.exists(os.path.join(out_dir, mmp_sdf)):
        sys.exit('no matched pairs were produced (the molecules share no common cores) — nothing to analyse')

    sanitized = [sanitize_prop(p) for p in props]
    dest = f'{job}_alllevels.csv'
    print('[2/2] extracting all core levels (0-5) ...', file=sys.stderr)
    df = mmp.deposition.extract_mmp_data_all_levels(
        filename=mmp_sdf, destination_file=dest, property_names=sanitized,
        raw_property_names=dict(zip(sanitized, props)), levels=[0, 1, 2, 3, 4, 5], dataset=False)
    path = os.path.join(out_dir, dest)
    print(f'dataset: {path} ({len(df)} rows)', file=sys.stderr)
    return path


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _resolve_specs(raw, wanted, directions):
    delta_cols = [c for c in raw.columns if c.startswith('property__')]
    keys = [c[len('property__'):] for c in delta_cols]
    if not keys:
        sys.exit("the dataset has no 'property__<name>' columns — build it with this skill's `build` first")
    if wanted:
        chosen = []
        for w in wanted:
            hit = next((k for k in keys if k == w or k == sanitize_prop(w) or clean_label(k).lower() == w.lower()), None)
            if hit is None:
                sys.exit(f"property '{w}' not in the dataset; available: {[clean_label(k) for k in keys]}")
            chosen.append(hit)
    else:
        chosen = keys
    specs = []
    for i, k in enumerate(chosen):
        label = clean_label(k)
        direction = directions.get(label.lower()) or directions.get(k.lower()) or 'higher'
        specs.append(make_prop_spec(k, is_primary=(i == 0), value_col=f'property__{k}',
                                    direction=direction, label=label))
    return specs


def make_report(args, dataset_path=None):
    from mmpkit import report as rpt

    dataset_path = dataset_path or args.dataset
    raw = pd.read_csv(dataset_path)
    wanted = [p.strip() for p in (args.props or '').split(',') if p.strip()]
    specs = _resolve_specs(raw, wanted, parse_direction(args.direction))
    rpt.MAX_POOL_LEVEL = args.max_pool_level
    try:
        html_text, summary = rpt.build_compound_report(
            raw, args.smiles, specs, lang=args.lang, dataset_name=os.path.basename(dataset_path),
            max_designs=args.max_designs, improve_threshold=args.improve_threshold,
            worsen_tolerance=args.worsen_tolerance)
    except ValueError as e:
        sys.exit(f'error: {e}')
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(dataset_path)), 'compound_SAR_report.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html_text)
    print(f'report: {out} ({len(html_text) // 1024} KB)', file=sys.stderr)
    print(json.dumps({k: v for k, v in summary.items() if k != 'designs'}
                     | {'designs': [f"{d['smiles']} (grade {d['grade']}, {d['kind']})" for d in summary['designs']],
                        'report_file': out}, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common_build(p):
        p.add_argument('--input', required=True, help='CSV (needs a SMILES column) or SDF of the molecules')
        p.add_argument('--props', help='comma-separated numeric property columns/SDF tags (first = primary)')
        p.add_argument('--out-dir', required=True)
        p.add_argument('--job', help='dataset name (default: input file name)')
        p.add_argument('--smiles-col', help='CSV only: the SMILES column (default: a column named smiles)')
        p.add_argument('--name-col', help='CSV column or SDF tag holding the molecule name (default: Name/ID column, or the SDF title)')
        p.add_argument('--workers', type=int, default=3, help='processes for the fragmentation step (default 3)')
        p.add_argument('--max-molecules', type=int, default=5000)

    def common_report(p):
        p.add_argument('--smiles', required=True, help='SMILES of the compound the report is about')
        p.add_argument('--direction', default='', help='e.g. "Clearance=lower,pIC50=higher" (default: higher is better)')
        p.add_argument('--max-pool-level', type=int, default=1,
                       help='how far a position may be widened to pool pairs across series; 0-5 (default 1 = '
                            'ring system kept, substituents stripped; higher levels are atom-environment '
                            'fingerprints and pool far more loosely)')
        p.add_argument('--lang', choices=['zh', 'en'], default='zh')
        p.add_argument('--max-designs', type=int, default=8)
        p.add_argument('--improve-threshold', type=float, default=0.1)
        p.add_argument('--worsen-tolerance', type=float, default=0.3)
        p.add_argument('--out', help='output HTML path')

    pb = sub.add_parser('build', help='molecules -> all-levels MMP dataset CSV'); common_build(pb)
    pr = sub.add_parser('report', help='all-levels dataset + SMILES -> HTML report')
    pr.add_argument('--dataset', required=True, help='all-levels CSV made by `build`')
    pr.add_argument('--props', help='properties to cover (default: all in the dataset; first = primary)')
    common_report(pr)
    pu = sub.add_parser('run', help='build + report'); common_build(pu); common_report(pu)

    args = ap.parse_args()
    if args.cmd == 'build':
        print(build_dataset(args))
    elif args.cmd == 'report':
        make_report(args)
    else:
        ds = build_dataset(args)
        if not args.out:
            args.out = os.path.join(os.path.abspath(args.out_dir), 'compound_SAR_report.html')
        make_report(args, ds)


if __name__ == '__main__':
    main()
