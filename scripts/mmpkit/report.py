"""
Compound-centred SAR report: given one compound and a raw all-levels MMP
deposition CSV, summarise what the dataset says about that compound and render
a self-contained HTML document (inline SVG structures, no external assets).

Everything here is deterministic — numbers come from the dataset, wording comes
from fixed templates. No LLM is involved, so the same inputs always give the
same report.

Public entry point: build_compound_report().

Sections of the report
  1  headline findings
  2  SAR overview figure (the compound at the centre, one coloured region per
     single-cut position around it, each listing its most frequent R-groups)
  3  site map (every cut of the compound and how much data backs it)
  4  per-site transformation tables (same fragment swap, wider core match)
  5  direct analogues (pairs where the compound itself is mol1)
  6  recommended designs (single swaps + a few additive combinations)
  7  limitations

"Position" matters: a fragment such as a methyl group shows up in many unrelated
places in a dataset, so a swap R1->R2 is only pooled with the compound's own site
when the *core* around the cut matches at some abstraction level. Core levels
that have collapsed to a one-/two-atom stub (e.g. '*C') no longer say where the
cut was, so they are never used for pooling.
"""
import functools
import json
import html
import itertools
import math
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog('rdApp.*')

MIN_CORE_HEAVY_ATOMS = 3      # a pooled core must keep at least this many non-* atoms
MAX_LEVEL = 5                 # highest core_level column the raw deposit format carries
MAX_POOL_LEVEL = 1            # how far a site's "wider" match is allowed to climb (see
                               # the sites loop below) — capped at 1 (ring system kept,
                               # substituents stripped) rather than MAX_LEVEL's 5, since
                               # levels 2+ pool on an atom-environment SMARTS fingerprint
                               # that's no longer a recognizable structure, and level 1's
                               # own _remove_substituents() (main.py) only anchors to a
                               # single '*' anyway, making anything looser unreliable for
                               # a multi-star (double/triple-cut) core.
CUT_ORDER = {'single': 0, 'double': 1, 'triple': 2}

# Shared by build_compound_report and build_core_report so the two report
# flavors look consistent and never drift apart — neither function has its
# own copy.
_REPORT_CSS = '''
body{font:14px/1.6 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;max-width:1080px;margin:24px auto;padding:0 18px;color:#222}
h1{font-size:24px;margin-bottom:4px}h2{font-size:18px;margin-top:34px;border-bottom:2px solid #e3e3e3;padding-bottom:4px}h3{font-size:15px;margin:22px 0 4px}
.sub{color:#666}.card{display:flex;gap:24px;align-items:center;background:#f7f8fa;border-radius:8px;padding:12px 18px}
.kv td{padding:2px 14px 2px 0}.kv td:first-child{color:#666}
table.t{border-collapse:collapse;width:100%;margin:8px 0}.t th{background:#f0f2f5;text-align:left;padding:6px 8px;font-size:12.5px}
.t td{border-top:1px solid #e6e6e6;padding:4px 8px;vertical-align:middle}.t tr.hl td{background:#fff7d6}
td.st svg{display:block}.sm{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:#444;word-break:break-all}
.h{display:inline-block;padding:10px;color:#555}.note{color:#777;font-size:12.5px;margin:2px 0}
.bar{position:relative;height:20px;width:130px;background:#fafafa;border-left:1px solid #ccc;margin-right:52px}
.bar:before{content:"";position:absolute;left:50%;top:0;bottom:0;border-left:1px solid #999}
.bar i{position:absolute;top:3px;bottom:3px;opacity:.55}.bar i.pos{background:#2a9d5b}.bar i.neg{background:#d0453b}
.bar b{position:absolute;right:-48px;top:1px;font-weight:600;font-size:12px}.na{color:#aaa}
.warn{background:#fff4f0;border-left:4px solid #d0453b;padding:8px 14px;margin:14px 0}
.ok{background:#f0f9f3;border-left:4px solid #2a9d5b;padding:8px 14px;margin:14px 0}
small{color:#888;margin-left:4px}.g{display:inline-block;color:#fff;font-weight:700;padding:0 8px;border-radius:10px}
.btn-clear-filter{font:inherit;font-size:12px;padding:3px 10px;border:1px solid #ccc;border-radius:5px;background:#fff;cursor:pointer}
.btn-clear-filter:hover{background:#f0f2f5}
.site-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin:10px 0}
.site-card{border:1px solid #e3e3e3;border-radius:8px;padding:10px;background:#fafbfc;text-align:center}
.site-card .pos-tag{display:inline-block;font-size:11px;color:#666;background:#eef1f5;border-radius:10px;padding:1px 8px;margin-bottom:6px}
.site-card .core-row{margin-top:2px;border-top:1px dashed #ddd;padding-top:4px}
.site-card .core-row .core-label{display:block;font-size:10.5px;color:#999;margin-bottom:2px}
.site-card .freq{font-size:12px;color:#444;margin-top:6px;text-align:left}
.site-card .freq b{color:#222}
a.site-card{display:block;text-decoration:none;color:inherit;cursor:pointer;transition:box-shadow .15s,transform .15s}
a.site-card:hover{box-shadow:0 2px 8px rgba(0,0,0,.14);transform:translateY(-2px)}
.site-hero{text-align:center;background:#fafbfc;border:1px solid #e3e3e3;border-radius:8px;padding:12px;margin:10px 0}
.site-hero svg{margin:0 auto}
.swap-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin:10px 0 18px}
.swap-card{border:1px solid #e3e3e3;border-radius:8px;padding:10px;background:#fafbfc;text-align:center}
.swap-card .name-row{display:flex;align-items:center;justify-content:center;gap:6px;margin-top:4px;font-size:12.5px;color:#333;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.swap-card .n-row{font-size:11.5px;color:#888;margin-top:2px}
.swap-card .props{margin-top:8px;border-top:1px dashed #ddd;padding-top:6px;text-align:left}
.swap-card .prop-block{margin:6px 0}
.swap-card .plabel{font-size:10.5px;color:#999;margin-bottom:2px}
.swap-card .bar{width:90px;margin-right:40px}
.swap-card .bar b{right:-38px}
.mol-cell{display:inline-block;text-align:center;vertical-align:middle}
.mol-cell .sm{display:block;margin-top:2px;max-width:140px}
.transform-cell{display:inline-flex;align-items:center;gap:6px;vertical-align:middle}
.transform-cell .arrow{color:#999;font-size:14px}
.ov-wrap{width:min(1500px,96vw);position:relative;left:50%;transform:translateX(-50%);margin:12px 0}
.ov-fig{position:relative;display:grid;grid-template-columns:max-content minmax(380px,440px) max-content;justify-content:center;gap:14px;align-items:center}
.ov-top,.ov-bottom{grid-column:1 / -1;display:flex;flex-wrap:wrap;justify-content:center;gap:14px}
.ov-left,.ov-right{display:flex;flex-direction:column;gap:14px;align-self:center}
.ov-left{grid-column:1;grid-row:2;align-items:flex-end}.ov-right{grid-column:3;grid-row:2;align-items:flex-start}
.ov-center{grid-column:2;grid-row:2;text-align:center;background:#fff;border:2px solid #2c7a6b;border-radius:10px;padding:10px}
.ov-top{grid-row:1}.ov-bottom{grid-row:3}
.ov-lines{position:absolute;left:0;top:0;pointer-events:none;overflow:visible;z-index:2}
.ov-center h3{margin:2px 0 0;font-size:16px;color:#1f5f53}
.ov-region{border:1px solid #e3e3e3;border-left-width:5px;border-radius:8px;padding:8px 10px;background:#fcfcfd}
.ov-top .ov-region,.ov-bottom .ov-region{border-left-width:1px;border-top-width:5px}
.ov-region h4{margin:0 0 6px;font-size:14px}.ov-region h4 small{font-weight:400}
.ov-region h4 a{color:inherit;text-decoration:none}
.ov-cards{display:grid;gap:8px}
.ov-card{width:118px;text-align:center;border:1px solid #ececec;border-radius:6px;padding:4px;background:#fff;cursor:pointer;transition:box-shadow .15s,transform .15s}
.ov-card:hover{box-shadow:0 2px 8px rgba(0,0,0,.14);transform:translateY(-2px)}
.ov-card.self{background:#f4f9f7;border-style:dashed}
.ov-card .freq{font-size:12px;font-weight:700;margin-top:2px}
.ov-card .tagline{font-size:10.5px;color:#888}
@media (max-width:1000px){.ov-wrap{width:auto;left:0;transform:none}.ov-fig{display:flex;flex-direction:column;align-items:stretch}.ov-left,.ov-right{align-items:stretch}.ov-center{order:-1}}
'''

_GRADE_COLORS = {'A': '#2a9d5b', 'B': '#e0a100', 'C': '#d0453b'}


def _grade_badge(g):
    return f'<span class="g" style="background:{_GRADE_COLORS[g]}">{g}</span>'


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _name_str(v):
    """A molecule name as text, or None when missing. Names are often purely
    numeric ('1', '48', ...), in which case pandas reads the name column as
    int64/float64 — an `isinstance(v, str)` check then wrongly treated every
    such molecule as unnamed. Integer-valued floats ('48.0' after a NaN forced
    the column to float) are written as '48'."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v if v and v != 'nan' else None
    try:
        if v != v:
            return None
    except Exception:
        pass
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m is not None else None


@functools.lru_cache(maxsize=None)
def _mmp_for_cores():
    import tempfile
    from .mmp_core import MMP
    return MMP(working_directory=tempfile.gettempdir())


@functools.lru_cache(maxsize=None)
def _core_at_level(core_smiles, level):
    """Abstract a core to `level` the same way core_level{N} columns were built at deposit time."""
    mol = Chem.MolFromSmiles(core_smiles)
    if mol is None:
        return None
    try:
        res = _mmp_for_cores().process_core_structures(mol, sign='*', level=level)
        return Chem.MolToSmiles(res, True)
    except Exception:
        return None


def _heavy_atoms_excluding_dummy(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return 0 if m is None else sum(1 for a in m.GetAtoms() if a.GetAtomicNum() != 0)


def _svg(smi, w=150, h=100):
    if smi in ('*[H]', '*[H]'):
        return '<span class="h">H<br><small>(–)</small></span>'
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return html.escape(str(smi))
    dr = rdMolDraw2D.MolDraw2DSVG(w, h)
    dr.drawOptions().addStereoAnnotation = False
    dr.drawOptions().setBackgroundColour((1, 1, 1, 0))
    dr.DrawMolecule(m)
    dr.FinishDrawing()
    return dr.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", '')


_SITE_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2',
                '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']


def _hex_to_rgb(hexcolor):
    hexcolor = hexcolor.lstrip('#')
    return tuple(int(hexcolor[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _strip_stars_to_mol(smi):
    """Caps every '*' attachment-point atom in `smi` with an explicit H
    (rather than deleting it outright) and re-sanitizes — turning each
    dummy into an ordinary H substituent is what keeps ring-fusion/aromatic
    nitrogens (e.g. a substituted bridgehead N that must become an -NH-
    once its external substituent is gone) kekulizable; outright removing
    the atom left such atoms with no way to recover a valid valence.
    Returns the resulting Mol (implicit-H form), or None if `smi` doesn't
    parse or can't be re-sanitized afterward."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetAtomicNum(1)
            a.SetNoImplicit(True)
            a.SetFormalCharge(0)
            a.SetIsAromatic(False)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return Chem.RemoveHs(out)
    except Exception:
        return None


def _strip_stars_canon(smi):
    mol = _strip_stars_to_mol(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _frag_atom_indices(mol, frag_smiles):
    """Atom indices in `mol` covered by a fragment's heavy-atom skeleton
    (its '*' attachment points stripped) — _enumerate_sites() discards the
    atom-index provenance of a cut (mmp_frag_and_index.py converts fragments
    to SMILES, not index maps), so highlighting a site's region on the
    original molecule has to re-derive it via substructure match instead.
    Returns an empty frozenset for a bare '*[H]' fragment (no substituent to
    highlight) and None if the skeleton can't be matched back onto `mol`."""
    frag = Chem.MolFromSmiles(frag_smiles) if frag_smiles else None
    if frag is None:
        return None
    rw = Chem.RWMol(frag)
    dummies = sorted((a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0), reverse=True)
    for idx in dummies:
        rw.RemoveAtom(idx)
    if rw.GetNumAtoms() == 0:
        return frozenset()
    # The departed '*' leaves its real neighbor one bond short of what its
    # SMILES chirality/explicit-H markup assumed (e.g. '[C@@H]' hard-codes
    # "exactly 1 H, no more") — left alone, RDKit keeps that atom under-valent
    # instead of filling the gap with an implicit H, so it silently fails to
    # match back onto `mol` at all. Clearing both lets valence/stereo recompute
    # from the atom's new (lower) connectivity.
    for a in rw.GetAtoms():
        a.SetNoImplicit(False)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    try:
        skeleton = rw.GetMol()
        Chem.SanitizeMol(skeleton)
    except Exception:
        # Stripping the '*' left an atom that can't be sanitised on its own — typically
        # an aromatic [n] in a pyridone-type ring whose '*' substituent (e.g. the
        # N-methyl of a double-cut fragment) was its third connection. Before this
        # fallback such a position resolved to None, i.e. "unknown region", which
        # _select_nonoverlapping_sites treats as never overlapping anything — so it
        # was never merged into the better-supported position it actually sits on.
        # Match the fragment as written instead, with each '*' as a wildcard; only the
        # non-'*' matches count as the fragment's own atoms.
        # (read as SMARTS, where '*' is a real wildcard; atom order is the SMILES's own)
        query = Chem.MolFromSmarts(frag_smiles)
        match = mol.GetSubstructMatch(query) if query is not None else ()
        if not match or query.GetNumAtoms() != frag.GetNumAtoms():
            return None
        return frozenset(m for m, fa in zip(match, frag.GetAtoms()) if fa.GetAtomicNum() != 0)
    match = mol.GetSubstructMatch(skeleton)
    return frozenset(match) if match else None


def _select_nonoverlapping_sites(sites_with_data, mol):
    """Greedily keep sites in their existing (best-first, i.e. highest n_pool
    first) order, dropping any whose matched region overlaps one already
    kept — directly implements '两个切割方式重叠时优先保留 MMP 多的那组'
    since callers pass sites already sorted by -n_pool. A site whose region
    can't be resolved (empty/unmatched, e.g. a bare '*[H]' position) never
    conflicts with anything and is always kept, just without a highlight.
    Returns [(site, atom_indices_frozenset), ...]."""
    selected = []
    used_atoms = set()
    for s in sites_with_data:
        atoms = _frag_atom_indices(mol, s['R1']) or frozenset()
        if atoms and (used_atoms & atoms):
            continue
        selected.append((s, atoms))
        if atoms:
            used_atoms |= atoms
    return selected


def _svg_mol_regions(mol, atom_groups, w=440, h=320, return_coords=False):
    """One picture of `mol` with each (atom_indices, hex_color) group in
    `atom_groups` highlighted in its own color — atoms and the bonds that
    lie entirely inside the group."""
    atom_colors, highlight_atoms = {}, []
    bond_colors, highlight_bonds = {}, []
    for atoms, color in atom_groups:
        rgb = _hex_to_rgb(color)
        atom_set = set(atoms)
        for a in atom_set:
            if a not in atom_colors:
                highlight_atoms.append(a)
                atom_colors[a] = rgb
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in atom_set and j in atom_set and bond.GetIdx() not in bond_colors:
                highlight_bonds.append(bond.GetIdx())
                bond_colors[bond.GetIdx()] = rgb
    dr = rdMolDraw2D.MolDraw2DSVG(w, h)
    dr.drawOptions().addStereoAnnotation = False
    dr.drawOptions().setBackgroundColour((1, 1, 1, 0))
    rdMolDraw2D.PrepareAndDrawMolecule(
        dr, Chem.Mol(mol), highlightAtoms=highlight_atoms, highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds, highlightBondColors=bond_colors,
    )
    dr.FinishDrawing()
    svg = dr.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", '')
    if not return_coords:
        return svg
    # pixel position of every atom in the drawing just made (PrepareAndDrawMolecule
    # lays out its own copy, so the coordinates can only be read from the drawer)
    coords = {}
    for i in range(mol.GetNumAtoms()):
        pt = dr.GetDrawCoords(i)
        coords[i] = (pt.x, pt.y)
    return svg, coords


# Draws one same-colour line from each overview region to its coloured part of the
# central structure. Positions only exist once the browser has laid the page out,
# so this runs in the page itself (no external assets); the stacked narrow-screen
# layout has no meaningful geometry and simply gets no lines.
_OV_CONNECTOR_JS = '''<script>
(function(){
  function draw(){
    document.querySelectorAll('.ov-fig').forEach(function(fig){
      var old=fig.querySelector('svg.ov-lines'); if(old) old.remove();
      if(getComputedStyle(fig).display!=='grid') return;
      var anchors=JSON.parse(fig.getAttribute('data-anchors')||'{}');
      var W=+fig.getAttribute('data-w'), H=+fig.getAttribute('data-h');
      var svg=fig.querySelector('.ov-center svg'); if(!svg) return;
      var f=fig.getBoundingClientRect(), sr=svg.getBoundingClientRect();
      var k=sr.width/W, NS='http://www.w3.org/2000/svg';
      var o=document.createElementNS(NS,'svg'); o.setAttribute('class','ov-lines');
      o.setAttribute('width',f.width); o.setAttribute('height',f.height);
      Object.keys(anchors).forEach(function(id){
        var a=anchors[id], reg=fig.querySelector('.ov-region[data-region="'+id+'"]'); if(!reg) return;
        var r=reg.getBoundingClientRect(), side=a[3];
        var x1=side==='left'?r.right:side==='right'?r.left:r.left+r.width/2;
        var y1=side==='top'?r.bottom:side==='bottom'?r.top:r.top+r.height/2;
        var x2=sr.left+a[0]*k, y2=sr.top+a[1]*k;
        var l=document.createElementNS(NS,'line');
        l.setAttribute('x1',x1-f.left); l.setAttribute('y1',y1-f.top);
        l.setAttribute('x2',x2-f.left); l.setAttribute('y2',y2-f.top);
        l.setAttribute('stroke',a[2]); l.setAttribute('stroke-width','2'); l.setAttribute('opacity','.75');
        o.appendChild(l);
        var c=document.createElementNS(NS,'circle');
        c.setAttribute('cx',x2-f.left); c.setAttribute('cy',y2-f.top); c.setAttribute('r','4'); c.setAttribute('fill',a[2]);
        o.appendChild(c);
      });
      fig.appendChild(o);
    });
  }
  window.addEventListener('load',draw); window.addEventListener('resize',draw);
})();
</script>'''


def _mol_cell(smi, w=90, h=70):
    """Structure image with its SMILES captioned directly below it — the
    'structure on top, SMILES below' presentation used everywhere a bare
    SMILES string would otherwise appear on its own."""
    return f'<div class="mol-cell">{_svg(smi, w, h)}<div class="sm">{html.escape(str(smi))}</div></div>'


def _transform_cell(smi1, smi2, w=90, h=70):
    """Two _mol_cell()s with an arrow between them, for an R1->R2-style swap."""
    return (f'<div class="transform-cell">{_mol_cell(smi1, w, h)}'
            f'<span class="arrow">→</span>{_mol_cell(smi2, w, h)}</div>')


def _enumerate_sites(mol, max_fragment_size=10):
    """All 1/2/3-bond cuts of `mol` via the SAME engine the deposits were built with.
    Returns [{'R1', 'core', 'n_stars', 'cut_type'}]."""
    from .frag_index import MMPIdentifier
    ident = MMPIdentifier(parent=None, max_fragment_size=max_fragment_size)
    ident.perform_single_cut(mol, mol_id=0)
    ident.perform_double_cut(mol, mol_id=0)
    if len(ident.find_acyclic_bonds(mol)) <= 25:
        ident.perform_triple_cut(mol, mol_id=0)
    out, seen = [], set()
    for cut_type, n, index in (('single', 1, ident.single_cut_index),
                               ('double', 2, ident.double_cut_index),
                               ('triple', 3, ident.triple_cut_index)):
        for core_key, entries in index.items():
            for e in entries:
                key = (e[0], core_key)
                if key in seen:
                    continue
                seen.add(key)
                out.append({'R1': e[0], 'core': core_key, 'n_stars': n, 'cut_type': cut_type})
    return out


def _attach(core_smiles, frag_smiles):
    """Join a single-* core and a single-* fragment into a molecule (SMILES) — None on failure.
    '*[H]' as the fragment just caps the core with hydrogen."""
    core = Chem.MolFromSmiles(core_smiles)
    if core is None:
        return None
    cd = [a.GetIdx() for a in core.GetAtoms() if a.GetAtomicNum() == 0]
    if len(cd) != 1:
        return None
    if _canon(frag_smiles) == _canon('*[H]'):
        rw = Chem.RWMol(core)
        rw.RemoveAtom(cd[0])
        try:
            m = rw.GetMol(); Chem.SanitizeMol(m)
            return Chem.MolToSmiles(m)
        except Exception:
            return None
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None:
        return None
    fd = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(fd) != 1:
        return None
    n_core = core.GetNumAtoms()
    combo = Chem.RWMol(Chem.CombineMols(core, frag))
    a_nb = core.GetAtomWithIdx(cd[0]).GetNeighbors()[0].GetIdx()
    b_nb = frag.GetAtomWithIdx(fd[0]).GetNeighbors()[0].GetIdx() + n_core
    bt = core.GetBondBetweenAtoms(cd[0], a_nb).GetBondType()
    combo.AddBond(a_nb, b_nb, bt)
    for idx in sorted([cd[0], fd[0] + n_core], reverse=True):
        combo.RemoveAtom(idx)
    try:
        m = combo.GetMol(); Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def _exact_match(series, target):
    ct = _canon(target)
    uniq = series.dropna().unique()
    cm = {s: _canon(s) for s in uniq}
    return series.map(cm) == ct


def _fragment_occurrence_info(raw, cut_type, core_at_level, level, props):
    """How many DISTINCT MOLECULES (not pairs/rows) in the whole dataset
    carry each fragment at this position — i.e. how often a fragment
    actually shows up in the patent, independent of what it was ever paired
    with. Different from r2_table's own 'n' (a pair/transformation count
    restricted to pairs whose R1 is one specific fragment): this scans every
    row sharing the same (cut_type, core at this level) regardless of which
    side carries which fragment, so a fragment counts once whether it was
    recorded as R1 or R2 for a given underlying pair (the raw deposition
    stores each pair in both directions — see mmp_frag_and_index.py — so a
    fragment seen only ever as 'R2' would otherwise be invisible here).
    Canonical SMILES is the grouping key (same as _exact_match) since the
    same fragment can appear written differently across rows.

    Also collects each contributing molecule's own name/measured values (for
    `props`) — feeds section 8's master molecule list and the SAR overview
    figure's "click a fragment to see its molecules" feature (build_compound_
    report), not just the "出现 N" count on its own.

    Returns {canonical_fragment_smiles: {'count': n, 'molecules':
    [{'smiles','name','values': {label: value_or_None}}, ...]}}."""
    ct_mask = raw['cut_type'].astype(str).str.startswith(cut_type)
    core_mask = raw[f'core_level{level}'] == core_at_level
    sub = raw[ct_mask & core_mask]
    if sub.empty:
        return {}

    def _side_frame(mol_col, frag_col, name_col, value_col_map):
        out = pd.DataFrame({'mol': sub[mol_col], 'frag': sub[frag_col]})
        out['name'] = sub[name_col] if name_col in sub.columns else None
        for label, col in value_col_map.items():
            out[f'value__{label}'] = pd.to_numeric(sub[col], errors='coerce') if col and col in sub.columns else float('nan')
        return out

    value_map_1 = {p['label']: p['m1'] for p in props}
    value_map_2 = {p['label']: p['m2'] for p in props}
    occ1 = _side_frame('mol1_smiles', 'R1', 'mol1_name', value_map_1)
    occ2 = _side_frame('mol2_smiles', 'R2', 'mol2_name', value_map_2)
    occ = pd.concat([occ1, occ2], ignore_index=True).dropna(subset=['mol', 'frag'])
    occ['frag_canon'] = occ['frag'].map(lambda s: _canon(s) or s)
    occ = occ.drop_duplicates(['mol', 'frag_canon'])

    result = {}
    for frag_canon, g in occ.groupby('frag_canon'):
        molecules = []
        for _, row in g.iterrows():
            name = _name_str(row['name'])
            values = {label: (row[f'value__{label}'] if pd.notna(row[f'value__{label}']) else None)
                      for label in value_map_1}
            molecules.append({'smiles': row['mol'], 'name': name, 'values': values})
        result[frag_canon] = {'count': len(molecules), 'molecules': molecules}
    return result


def _collect_absolute_values(raw, props):
    """Every molecule's own absolute measured value per active property,
    straight from the raw deposition columns (mol1_value__X/mol2_value__X) —
    independent of any network-relative reconstruction, so a report can show
    a real reference number for a molecule without needing a whole
    Free-Wilson regression just to get one. Returns {canon_smiles:
    {'name': str_or_None, 'values': {label: value}}}."""
    out = {}

    def _collect(mol_col, name_col, value_map):
        if mol_col not in raw.columns:
            return
        cols = [mol_col] + ([name_col] if name_col in raw.columns else []) \
            + [c for c in set(value_map.values()) if c and c in raw.columns]
        for _, row in raw[cols].drop_duplicates(subset=[mol_col]).iterrows():
            smi = row[mol_col]
            if not isinstance(smi, str):
                continue
            canon = _canon(smi) or smi
            rec = out.setdefault(canon, {'name': None, 'values': {}})
            if name_col in raw.columns:
                nm = _name_str(row[name_col])
                if nm and not rec['name']:
                    rec['name'] = nm
            for label, col in value_map.items():
                if not col or col not in raw.columns:
                    continue
                v = row[col]
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = None
                if v is not None and v == v and rec['values'].get(label) is None:
                    rec['values'][label] = v

    _collect('mol1_smiles', 'mol1_name', {p['label']: p['m1'] for p in props})
    _collect('mol2_smiles', 'mol2_name', {p['label']: p['m2'] for p in props})
    return out


# --------------------------------------------------------------------------
# dataset introspection
# --------------------------------------------------------------------------
def _prop_columns(raw, specs):
    """Per spec: delta column, mol1/mol2 absolute-value columns (or None)."""
    out = []
    for s in specs:
        key = s['key']
        d = s.get('value_col') or (f'property__{key}' if f'property__{key}' in raw.columns else 'property')
        if d not in raw.columns:
            continue
        if d == 'property':
            m1, m2 = 'mol1_value', 'mol2_value'
        else:
            m1, m2 = f'mol1_value__{key}', f'mol2_value__{key}'
        out.append({'spec': s, 'label': s['label'], 'higher': s.get('direction', 'higher') != 'lower',
                    'd': d, 'm1': m1 if m1 in raw.columns else None,
                    'm2': m2 if m2 in raw.columns else None,
                    'additive': str(key).startswith('subtract_') or d == 'property'})
    return out


def _stat(vals):
    v = pd.to_numeric(vals, errors='coerce').dropna()
    return {'n': int(len(v)), 'mean': float(v.mean()) if len(v) else float('nan'),
            'sd': float(v.std()) if len(v) > 1 else float('nan'),
            'min': float(v.min()) if len(v) else float('nan'),
            'max': float(v.max()) if len(v) else float('nan')}


def _grade(stats_primary):
    """A: >=3 pairs, all the same sign.  B: >=2 pairs.  C: a single pair."""
    n = stats_primary['n']
    if n >= 3 and (stats_primary['min'] > 0 or stats_primary['max'] < 0):
        return 'A'
    if n >= 2:
        return 'B'
    return 'C'


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_compound_report(raw, smiles, specs, lang='zh', dataset_name='', max_designs=8,
                          max_analogs=30, max_site_tables=6, rows_per_site=12,
                          improve_threshold=0.1, worsen_tolerance=0.3):
    """Returns (html_text, summary_dict). Raises ValueError with a user-facing message on bad input."""
    L = (lambda zh, en: zh if lang == 'zh' else en)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(L(f'无法解析 SMILES：{smiles}', f'Cannot parse SMILES: {smiles}'))
    q = Chem.MolToSmiles(mol)

    need = ['mol1_smiles', 'mol2_smiles', 'R1', 'R2', 'cut_type'] + [f'core_level{i}' for i in range(MAX_LEVEL + 1)]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise ValueError(L('数据集不是 all-levels 原始 MMP 格式，缺少列：', 'Dataset is not a raw all-levels MMP file, missing columns: ') + ', '.join(miss))
    props = _prop_columns(raw, specs)
    if not props:
        raise ValueError(L('找不到所选属性对应的数值列。', 'No numeric column found for the selected properties.'))
    prim = props[0]

    # canonical mol1 / mol2 once
    m1c = raw['mol1_smiles'].map({s: _canon(s) for s in raw['mol1_smiles'].dropna().unique()})
    all_mols = set(m1c.dropna().unique()) | set(
        raw['mol2_smiles'].map({s: _canon(s) for s in raw['mol2_smiles'].dropna().unique()}).dropna().unique())
    A = raw[m1c == q].copy()
    in_dataset = len(A) > 0
    name = None
    if in_dataset and 'mol1_name' in A.columns:
        names = A['mol1_name'].dropna().astype(str).unique()
        name = ', '.join(names[:3]) if len(names) else None

    # ---- measured values and rank among all molecules ------------------------
    base = {}
    ranks = {}
    for p in props:
        if p['m1'] is None:
            continue
        v = pd.to_numeric(A[p['m1']], errors='coerce').dropna() if in_dataset else pd.Series(dtype=float)
        base[p['label']] = float(v.iloc[0]) if len(v) else None
        allv = pd.concat([
            pd.DataFrame({'s': m1c, 'v': pd.to_numeric(raw[p['m1']], errors='coerce')}),
        ]).dropna().drop_duplicates('s')
        if base[p['label']] is not None and len(allv):
            better = (allv['v'] > base[p['label']] + 1e-9).sum() if p['higher'] else (allv['v'] < base[p['label']] - 1e-9).sum()
            ranks[p['label']] = (int(better) + 1, int(len(allv)))

    # ---- direct analogues -----------------------------------------------------
    analogs = []
    if in_dataset:
        A['_ct'] = A['cut_type'].astype(str).str.split('#').str[0].map(CUT_ORDER).fillna(9)
        A = A.sort_values('_ct').drop_duplicates('mol2_smiles')
        for _, r in A.iterrows():
            analogs.append(r)
        sort_col = prim['d']
        analogs.sort(key=lambda r: -(pd.to_numeric(r[sort_col], errors='coerce') if prim['higher'] else -pd.to_numeric(r[sort_col], errors='coerce'))
                     if pd.notna(pd.to_numeric(r[sort_col], errors='coerce')) else 1e9)

    # ---- sites ----------------------------------------------------------------
    sites = []
    for s in _enumerate_sites(mol):
        r1_mask = _exact_match(raw['R1'], s['R1'])
        ct_mask = raw['cut_type'].astype(str).str.startswith(s['cut_type'])
        base_mask = r1_mask & ct_mask
        levels = {}
        best_L = None
        for lv in range(MAX_POOL_LEVEL + 1):
            c = _core_at_level(s['core'], lv)
            if c is None or _heavy_atoms_excluding_dummy(c) < MIN_CORE_HEAVY_ATOMS:
                continue
            m = base_mask & (raw[f'core_level{lv}'] == c)
            levels[lv] = (c, m)
            if m.any():
                best_L = lv           # deepest valid level that still has data
        n0 = int(levels[0][1].sum()) if 0 in levels else 0
        pool_mask = levels[best_L][1] if best_L is not None else None
        pool = raw[pool_mask].drop_duplicates(['mol1_smiles', 'mol2_smiles']) if pool_mask is not None else raw.iloc[0:0]
        sites.append({**s, 'level': best_L, 'core_at_level': levels[best_L][0] if best_L is not None else None,
                      'n_same_series': n0, 'n_pool': len(pool), 'pool': pool,
                      'valid_levels': sorted(levels)})
    sites.sort(key=lambda s: (-s['n_pool'], s['n_stars']))

    # ---- per-site transformation tables ---------------------------------------
    def r2_table(site):
        pool = site['pool']
        rows = []
        for r2, g in pool.groupby('R2'):
            same = g[g['core_level0'] == _core_at_level(site['core'], 0)] if site['level'] is not None else g.iloc[0:0]
            row = {'R2': r2, 'n': len(g), 'n_same': len(same), 'per': {}, 'example': g.iloc[0]}
            for p in props:
                row['per'][p['label']] = _stat(g[p['d']])
            row['_imp'] = row['per'][prim['label']]['mean'] if prim['higher'] else -row['per'][prim['label']]['mean']
            rows.append(row)
        rows.sort(key=lambda r: (-(r['n']), -(r['_imp'] if r['_imp'] == r['_imp'] else -1e9)))
        return rows

    site_tables, seen_sig = [], set()
    for s in sites:
        if s['n_pool'] > 0:
            sig = frozenset(zip(s['pool']['mol1_smiles'], s['pool']['mol2_smiles']))
            if sig in seen_sig:       # same pairs already shown under another cut of the same position
                continue
            seen_sig.add(sig)
            site_tables.append((s, r2_table(s)))

    # Maps a site's identity (R1, core) to the anchor id of its per-site
    # transformation block (section 4) — only sites actually rendered there
    # (site_tables[:max_site_tables]) get one, so a site-map card (section 3)
    # whose position didn't make that cut just stays a plain, non-clickable card.
    site_anchor = {(s['R1'], s['core']): idx for idx, (s, _) in enumerate(site_tables[:max_site_tables])}

    # ---- designs --------------------------------------------------------------
    designs = []
    mol_c = q

    def imp(p, d):
        return d if p['higher'] else -d

    def _evidence_stats(row):
        """stats of the property that actually justifies the candidate (largest gain)"""
        best, bg = None, -1e9
        for p in props:
            st = row['per'][p['label']]
            if st['n'] and imp(p, st['mean']) > bg:
                best, bg = st, imp(p, st['mean'])
        return best

    def _total_gain(row):
        return sum(imp(p, row['per'][p['label']]['mean']) for p in props if row['per'][p['label']]['n'])

    def candidate_ok(row):
        # at least one property clearly improves, and none gets clearly worse
        gains = []
        for p in props:
            st = row['per'][p['label']]
            if not st['n']:
                continue
            g = imp(p, st['mean'])
            if g < -worsen_tolerance:
                return False
            gains.append(g)
        return bool(gains) and max(gains) >= improve_threshold

    singles = []
    for s, rows in site_tables:
        if s['n_stars'] != 1:
            continue
        # sanity: re-attaching R1 to the core must give back the compound
        if _attach(s['core'], s['R1']) != q:
            continue
        for row in rows:
            if row['R2'] == s['R1'] or not candidate_ok(row):
                continue
            new = _attach(s['core'], row['R2'])
            if new is None or new == q:
                continue
            singles.append({'site': s, 'row': row, 'smiles': new, 'grade': _grade(_evidence_stats(row))})
    # one entry per resulting molecule (best-supported first)
    gr = {'A': 0, 'B': 1, 'C': 2}
    singles.sort(key=lambda d: (gr[d['grade']], -_total_gain(d['row'])))
    seen_smi, uniq_singles = set(), []
    for d in singles:
        if d['smiles'] in seen_smi:
            continue
        seen_smi.add(d['smiles'])
        uniq_singles.append(d)

    def predict(base_vals, rows):
        out = {}
        for p in props:
            b = base_vals.get(p['label'])
            deltas = []
            for r in rows:
                st = r['per'][p['label']]
                if st['n']:
                    deltas.append(st['mean'])
            if b is None or not p['additive']:
                out[p['label']] = None
            else:
                out[p['label']] = b + sum(deltas)
        return out

    for d in uniq_singles[:max_designs]:
        d['pred'] = predict(base, [d['row']])
        d['kind'] = 'single'
        designs.append(d)

    # additive combinations of two single swaps at different positions
    combos = []
    top = uniq_singles[:6]
    for a, b in itertools.combinations(top, 2):
        if a['site']['R1'] == b['site']['R1'] and a['site']['core'] == b['site']['core']:
            continue
        mid = a['smiles']
        # position check: swap B must still be possible on the intermediate molecule at the same
        # (abstract) position — i.e. the two swaps do not overlap.
        ok = False
        for s2 in _enumerate_sites(Chem.MolFromSmiles(mid)):
            if s2['n_stars'] == 1 and s2['R1'] == b['site']['R1']:
                lv = b['site']['level']
                if lv is not None and _core_at_level(s2['core'], lv) == b['site']['core_at_level']:
                    new = _attach(s2['core'], b['row']['R2'])
                    if new:
                        ok = True
                        break
        if not ok:
            continue
        pred = predict(base, [a['row'], b['row']])
        grade = max(a['grade'], b['grade'])
        combos.append({'site': a['site'], 'site2': b['site'], 'row': a['row'], 'row2': b['row'],
                       'smiles': new, 'grade': grade if grade != 'A' else 'B', 'pred': pred, 'kind': 'combo',
                       'ok_props': all((pred[p['label']] is None or base.get(p['label']) is None
                                        or imp(p, pred[p['label']] - base[p['label']]) >= -worsen_tolerance) for p in props)})
    combos = [c for c in combos if c['ok_props'] and c['smiles'] not in seen_smi and c['smiles'] != q]
    combos.sort(key=lambda d: (gr[d['grade']], -(_total_gain(d['row']) + _total_gain(d['row2']))))
    cs = set()
    for c in combos:
        if c['smiles'] in cs:
            continue
        cs.add(c['smiles'])
        designs.append(c)
        if len([d for d in designs if d['kind'] == 'combo']) >= max(2, max_designs // 3):
            break

    # worst swaps (informational)
    worse = []
    for s, rows in site_tables:
        for row in rows:
            ps = row['per'][prim['label']]
            if ps['n'] >= 2 and imp(prim, ps['mean']) < -0.5:
                worse.append((s, row))
    worse.sort(key=lambda t: imp(prim, t[1]['per'][prim['label']]['mean']))

    # ------------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------------
    allv = []
    for _, rows in site_tables:
        for r in rows:
            for p in props:
                m = r['per'][p['label']]['mean']
                if m == m:
                    allv.append(abs(m))
    scale = max(1.0, float(np.percentile(allv, 90))) if allv else 1.0

    def bar(v, sc=None, higher=True):
        if v is None or v != v:
            return '<span class="na">–</span>'
        sc = sc or scale
        good = (v >= 0) == higher
        pct = min(abs(v) / sc, 1) * 50
        left = 50 if v >= 0 else 50 - pct
        return (f'<div class="bar"><i class="{"pos" if good else "neg"}" style="left:{left:.1f}%;width:{pct:.1f}%"></i>'
                f'<b>{v:+.2f}</b></div>')

    def n_str(x):
        return '–' if x is None else (f'{x:.2f}')

    css = _REPORT_CSS
    grade_badge = _grade_badge

    parts = []
    title = L('化合物 SAR 报告', 'Compound SAR report')
    parts.append(f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head><body>')
    parts.append(f'<h1>{title}{": " + html.escape(name) if name else ""}</h1>')
    parts.append(f'<p class="sub">{L("数据集", "Dataset")}：{html.escape(dataset_name or "")}。'
                 f'{L("Δ = 变换后 − 变换前，单位与属性相同；绿色表示向更优方向变化。", "Δ = after − before, in the property’s own units; green means a change in the better direction.")}</p>')

    # card
    kv = [f'<tr><td>SMILES</td><td class="sm">{html.escape(q)}</td></tr>']
    if name:
        kv.append(f'<tr><td>{L("数据集中的名称", "Name in dataset")}</td><td>{html.escape(name)}</td></tr>')
    for p in props:
        b = base.get(p['label'])
        rk = ranks.get(p['label'])
        txt = n_str(b) if b is not None else L('无实测值', 'not measured')
        if rk:
            txt += L(f'（{rk[1]} 个分子中第 {rk[0]} 名）', f' (rank {rk[0]} of {rk[1]} molecules)')
        kv.append(f'<tr><td>{html.escape(p["label"])}（{L("越高越好" if p["higher"] else "越低越好", "higher is better" if p["higher"] else "lower is better")}）</td><td><b>{txt}</b></td></tr>')
    mw, lp, tp = Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol)
    kv.append(f'<tr><td>{L("理化性质", "Properties")}</td><td>MW {mw:.0f} · cLogP {lp:.1f} · TPSA {tp:.0f}</td></tr>')
    parts.append(f'<div class="card"><div>{_svg(q, 330, 230)}</div><div><table class="kv">{"".join(kv)}</table></div></div>')
    if not in_dataset:
        parts.append(f'<div class="warn">{L("该化合物不在数据集中，所以没有直接类似物和实测值；下面只列出它的各个切割位点在数据集里已有的取代基变换，设计部分只给 Δ，不给预测绝对值。", "This compound is not in the dataset, so there are no direct analogues or measured values; only fragment-level swaps at its cut positions are shown, and designs carry Δ only (no predicted absolute values).")}</div>')

    # 1 headline
    parts.append(f'<h2>1. {L("先说结论", "Key findings")}</h2>')
    parts.append(f'<p class="note">{L("证据等级", "Grade")}：'
                 f'{grade_badge("A")} {L("多对且方向一致", "≥3 consistent pairs")} '
                 f'{grade_badge("B")} {L("多对但不一致，或组合", "several mixed pairs, or a combination")} '
                 f'{grade_badge("C")} {L("仅 1 对", "a single pair")}'
                 f'{L("——下面和第 6 节里每个变换/设计后面括号里的字母就是这个。", "— the letter in parentheses after every swap/design below is this.")}</p>')
    parts.append('<ul>')
    sites_with_data = [s for s in sites if s['n_pool'] > 0]
    sites_no_data = [s for s in sites if s['n_pool'] == 0]
    if in_dataset:
        parts.append(f'<li>{L(f"数据集里有 <b>{len(analogs)}</b> 个与该化合物直接配对的类似物（其余部分结构相同，仅一处不同）。", f"<b>{len(analogs)}</b> direct analogues (identical apart from one change) are paired with this compound in the dataset.")}</li>')
    parts.append(f'<li>{L(f"它的 {len(sites)} 种切割方式中，<b>{len(sites_with_data)}</b> 种在数据集里有对应位点的数据，<b>{len(sites_no_data)}</b> 种没有。", f"Of its {len(sites)} possible cuts, <b>{len(sites_with_data)}</b> have data at the matching position in the dataset and <b>{len(sites_no_data)}</b> have none.")}</li>')
    best = None
    for s, rows in site_tables:
        for row in rows:
            ps = row['per'][prim['label']]
            if ps['n'] and imp(prim, ps['mean']) > 0:
                score = (gr[_grade(ps)], -imp(prim, ps['mean']))
                if best is None or score < best[0]:
                    best = (score, s, row)
    if best:
        _, s, row = best
        ps = row['per'][prim['label']]
        parts.append(f'<li>{L("支持最充分的提升性变换", "Best-supported improving swap")}：{_transform_cell(s["R1"], row["R2"], 70, 55)} '
                     f'{html.escape(prim["label"])} {L("平均 Δ", "mean Δ")} {ps["mean"]:+.2f}（n={ps["n"]}，{L("证据等级", "grade")} {grade_badge(_grade(ps))}）。</li>')
    else:
        parts.append(f'<li>{L("在有数据的位点里，没有发现主属性平均向更优方向变化的取代基变换。", "No swap with a mean improvement in the primary property was found at any position with data.")}</li>')
    parts.append(f'<li>{L(f"共给出 <b>{len(designs)}</b> 个设计（见第 6 节）。", f"<b>{len(designs)}</b> designs are proposed (section 6).")}</li>')
    parts.append('</ul>')

    # 2 SAR overview figure — the compound at the centre with each of its
    # best-supported single-cut positions as a coloured region around it; each
    # region lists its most frequent R-groups (the R2 side of that position's
    # pooled pairs, i.e. exactly the per-position table of section 4) with
    # their pair counts. Built from the same site/pool data as the later
    # sections, not from separate tool calls, so every number here matches them.
    # Shared by the overview (section 2) and the site map (section 3): ONE
    # non-overlap selection over all positions and ONE color assignment, so both
    # sections show the same positions in the same colors. The overview just
    # filters this list down to single-cut positions.
    sites_with_data = [s for s in sites if s['n_pool'] > 0]
    n_no_data = len(sites) - len(sites_with_data)
    selected = _select_nonoverlapping_sites(sites_with_data, mol)
    n_overlap_dropped = len(sites_with_data) - len(selected)
    colored = []
    color_i = 0
    for s, atoms in selected:
        if atoms:
            color = _SITE_COLORS[color_i % len(_SITE_COLORS)]
            color_i += 1
        else:
            color = None
        colored.append((s, atoms, color))
    # Every position that has its own colour in the site map (single, double and
    # triple cuts) — positions merged into a better-supported one are NOT shown
    # here either, so both sections list the same positions.
    ov_sites = list(colored)
    _letter = lambda i: chr(ord('A') + i) if i < 26 else f'R{i + 1}'
    # position -> region letter, so the site map's cards can carry the same letter
    ov_letter = {(s_['R1'], s_['core']): _letter(i) for i, (s_, _a, _c) in enumerate(ov_sites)}

    # Every molecule any fragment card (section 2) links to, keyed by its own
    # canonical SMILES — feeds section 8's master molecule list. Declared here
    # (not inside `if ov_sites:` below) so it's always defined even when this
    # compound has no cut positions with data at all.
    master_molecules = {}

    def _record_molecule(frag_id, mol_entry):
        key = _canon(mol_entry['smiles']) or mol_entry['smiles']
        rec = master_molecules.setdefault(
            key, {'smiles': mol_entry['smiles'], 'name': mol_entry['name'],
                  'values': {}, 'frag_ids': set()})
        if mol_entry['name'] and not rec['name']:
            rec['name'] = mol_entry['name']
        for label, v in mol_entry['values'].items():
            if v is not None and rec['values'].get(label) is None:
                rec['values'][label] = v
        rec['frag_ids'].add(frag_id)

    if ov_sites:
        parts.append(f'<h2>2. {L("SAR 先行图", "SAR overview")}</h2>')
        parts.append(f'<p class="note">{L(f"中央是该化合物，不同颜色标出它在数据集里有数据的 {len(ov_sites)} 个切割位点（单切、双切、三切都包括，与位点示意一致）；每个区域放在它在结构上所处的一侧，列出该位点出现过的全部 R 基团，按转化次数从高到低排列。每张卡片两个数：<b>出现</b> = 这个 R 基团在数据集里实际出现在多少个不同化合物中（不管它和谁配对过，含本化合物自己，所以本化合物卡片的出现数至少是 1）；<b>转化</b> = 这个 R 基团作为“替换目标”，从本化合物这个位点的原始片段换过去的配对数——本化合物自己这项通常是 0，因为它是这组配对的起点，不是别人的替换目标。", f"The compound sits in the centre; colours mark its {len(ov_sites)} cut positions that have data (single, double and triple cuts, same set as the site map). Each region is placed on the side of the structure where its fragment sits and lists every R-group seen at that position, most-transformed first. Each card shows two numbers: <b>seen</b> = how many distinct compounds in the dataset actually carry this R-group, regardless of what it was ever paired with (including the compound itself, so its own card is always ≥1); <b>swap</b> = how many pairs recorded this R-group as the TARGET swapped to FROM this compound's own fragment at this position — usually 0 for the compound's own card, since it's the starting point these pairs are counted from, not a swap target.")}</p>')
        ov_groups = [(a_, c_) for _s, a_, c_ in ov_sites if a_]
        OV_W, OV_H = 440, 320
        if ov_groups:
            center_svg, ov_xy = _svg_mol_regions(mol, ov_groups, OV_W, OV_H, return_coords=True)
        else:
            center_svg, ov_xy = _svg(q, OV_W, OV_H), {}

        # -- per region: R-group cards + which side of the structure it belongs on
        regions = []
        for i, (s_, atoms_, color) in enumerate(ov_sites):
            color = color or '#8C8C8C'   # no highlightable atoms (e.g. a bare '*[H]' position)
            all_rows = r2_table(s_)
            # The compound's own fragment is this site's R1, by construction never
            # one of ITS OWN R2 alternatives (the pool is grouped by what R1 got
            # SWAPPED TO) — the only way it shows up in all_rows at all is a
            # degenerate R1==R2 entry (some other molecule happening to carry
            # the identical fragment at this position, paired back to itself),
            # which is rare; absent, its own count really is 0, not unknown —
            # shown as such instead of silently leaving the "本化合物" card
            # the only one on the figure with no n = ... line.
            self_row = next((r for r in all_rows if r['R2'] == s_['R1']), None)
            r2rows = [r for r in all_rows if r['R2'] != s_['R1']]
            # How many distinct molecules actually carry each fragment at this
            # position (occurrence in the dataset/patent) — a different number
            # from n above (a transformation/pair count), see
            # _fragment_occurrence_info's own docstring for why. Also collects
            # which molecules those are, for the click-to-filter feature below.
            occ_info = _fragment_occurrence_info(raw, s_['cut_type'], s_['core_at_level'], s_['level'], props)
            self_frag_id = f'frag-{i}-self'
            self_entry = occ_info.get(_canon(s_['R1']) or s_['R1'])
            self_occ = self_entry['count'] if self_entry else 0
            if self_entry:
                for m in self_entry['molecules']:
                    _record_molecule(self_frag_id, m)
            for j, row in enumerate(r2rows):
                row['frag_id'] = f'frag-{i}-{j}'
                entry = occ_info.get(_canon(row['R2']) or row['R2'])
                row['occ'] = entry['count'] if entry else 0
                if entry:
                    for m in entry['molecules']:
                        _record_molecule(row['frag_id'], m)
            regions.append({'i': i, 's': s_, 'atoms': atoms_, 'color': color, 'rows': r2rows,
                            'self_n': self_row['n'] if self_row else 0, 'self_occ': self_occ,
                            'self_frag_id': self_frag_id,
                            'weight': len(r2rows) + 1, 'pref': None})
        # preference order of the 4 sides from the fragment's centroid relative to the
        # structure's centre (normalised by the drawing's size, so a wide picture isn't
        # biased towards left/right)
        cx0, cy0 = OV_W / 2.0, OV_H / 2.0
        if ov_xy:
            allx = [x for x, _y in ov_xy.values()]; ally = [y for _x, y in ov_xy.values()]
            cx0, cy0 = (min(allx) + max(allx)) / 2.0, (min(ally) + max(ally)) / 2.0
        half_w = max(1.0, (max(allx) - min(allx)) / 2.0) if ov_xy else OV_W / 2.0
        half_h = max(1.0, (max(ally) - min(ally)) / 2.0) if ov_xy else OV_H / 2.0
        for r in regions:
            pts = [ov_xy[a] for a in r['atoms'] if a in ov_xy]
            if pts:
                dx = (sum(x for x, _ in pts) / len(pts) - cx0) / half_w
                dy = (sum(y for _, y in pts) / len(pts) - cy0) / half_h
                scores = {'left': -dx, 'right': dx, 'top': -dy, 'bottom': dy}
                r['pref'] = sorted(scores, key=lambda k: -scores[k])
                r['pos'] = (dx, dy)
            else:
                r['pref'] = ['bottom', 'top', 'right', 'left']
                r['pos'] = (0.0, 0.0)
        # greedy balanced assignment: heaviest regions first, each to its best side
        # that isn't already carrying more than ~40% of all cards (so one crowded
        # side spills into the next-nearest one instead of leaving the other sides empty)
        total_w = sum(r['weight'] for r in regions)
        cap = max(0.4 * total_w, max(r['weight'] for r in regions))
        load = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        for r in sorted(regions, key=lambda r: -r['weight']):
            side = next((sd for sd in r['pref'] if load[sd] + r['weight'] <= cap), r['pref'][0])
            r['side'] = side
            load[side] += r['weight']
        # order along each side by where the fragment really sits (top→bottom / left→right)
        side_regions = {sd: sorted([r for r in regions if r['side'] == sd],
                                   key=lambda r: r['pos'][1] if sd in ('left', 'right') else r['pos'][0])
                        for sd in load}

        def _region_html(r):
            s_, color, i = r['s'], r['color'], r['i']
            n_cards = r['weight']
            # columns chosen from the fragment count so a small region is a small block
            # and a big one grows wider rather than very tall
            if r['side'] in ('left', 'right'):
                cols = 1 if n_cards <= 4 else 2 if n_cards <= 12 else 3
            else:
                cols = min(n_cards, 8)
            click_hint = L('点击查看带有该片段的所有分子', 'Click to see every molecule carrying this fragment')
            cards = [f'<div class="ov-card self" onclick="filterMolList(\'{r["self_frag_id"]}\')" '
                     f'title="{click_hint}"><div class="tagline">{L("本化合物", "This compound")}</div>'
                     f'{_svg(s_["R1"], 100, 75)}<div class="freq" style="color:{color}">'
                     f'{L("出现", "seen")} {r["self_occ"]} · {L("转化", "swap")} {r["self_n"]}</div></div>']
            for row in r['rows']:
                cards.append(f'<div class="ov-card" onclick="filterMolList(\'{row["frag_id"]}\')" '
                             f'title="{click_hint}">{_svg(row["R2"], 100, 75)}<div class="freq" style="color:{color}">'
                             f'{L("出现", "seen")} {row["occ"]} · {L("转化", "swap")} {row["n"]}</div></div>')
            ttl = f'{_letter(i)} {L("区", "region")} · {s_["cut_type"]}'
            anchor_idx = site_anchor.get((s_['R1'], s_['core']))
            if anchor_idx is not None:
                ttl = f'<a href="#site-{anchor_idx}">{ttl}</a>'
            return (f'<div class="ov-region" data-region="{i}" style="border-color:{color}">'
                    f'<h4 style="color:{color}">{ttl} <small>· {s_["n_pool"]} {L("对", "pairs")} · {len(r["rows"])} {L("个 R 基团", "R-groups")}</small></h4>'
                    f'<div class="ov-cards" style="grid-template-columns:repeat({cols},118px)">{"".join(cards)}</div></div>')

        # connector anchor on the structure: the region's atom that sticks out furthest
        # towards its own side, in the drawing's pixel coordinates (the browser script
        # below maps them to the rendered size)
        anchors = {}
        for r in regions:
            pts = [ov_xy[a] for a in r['atoms'] if a in ov_xy]
            if not pts:
                continue
            key = {'left': lambda p: p[0], 'right': lambda p: -p[0],
                   'top': lambda p: p[1], 'bottom': lambda p: -p[1]}[r['side']]
            anchors[str(r['i'])] = [round(v, 1) for v in min(pts, key=key)] + [r['color'], r['side']]
        legend = ' '.join(f'<span class="g" style="background:{r["color"]}">{_letter(r["i"])}</span>' for r in regions)
        side_html = {sd: ''.join(_region_html(r) for r in side_regions[sd]) for sd in side_regions}
        parts.append(
            f'<div class="ov-wrap"><div class="ov-fig" data-anchors=\'{html.escape(json.dumps(anchors), quote=True)}\' data-w="{OV_W}" data-h="{OV_H}">'
            + (f'<div class="ov-top">{side_html["top"]}</div>' if side_html['top'] else '')
            + f'<div class="ov-left">{side_html["left"]}</div>'
            f'<div class="ov-center"><h3>{L("中央结构", "Central structure")}{": " + html.escape(name) if name else ""}</h3>'
            f'{center_svg}<div>{legend}</div></div>'
            f'<div class="ov-right">{side_html["right"]}</div>'
            + (f'<div class="ov-bottom">{side_html["bottom"]}</div>' if side_html['bottom'] else '')
            + '</div></div>')
        parts.append(_OV_CONNECTOR_JS)

    # 3 site map — one card per cut (position), grid-laid-out (not a plain
    # table): fragment + its matched core, frequency, and one representative
    # paired compound actually found at that position — loosely inspired by
    # a reference "regions around a central structure" SAR infographic the
    # user shared, scaled down to a plain responsive grid rather than
    # hand-positioned regions (robust regardless of how many sites there
    # are), and without the reference's per-fragment potency-tier badges
    # (frequency + representative compound only, per the user's own choice).
    parts.append(f'<h2>3. {L("位点示意", "Site map")}</h2>')
    parts.append(f'<p class="note">{L("同系列 = 核心完全相同（level 0）的配对数；跨系列 = 放宽到 level 1（保留环系、去掉取代基）后的配对数（括号内标注实际用的 level）。不会放宽到更高的抽象级别——level 1 以上核心只是局部原子环境的指纹，不再是可读的结构，也无法可靠处理有 2-3 个连接点的位点。下方只列出在数据集里找到配对数据的切割位置。", "Same series = pairs whose core is identical (level 0); wider = pairs after relaxing the core match to level 1 (ring system kept, substituents stripped; the level actually used is shown in brackets). Matching is never relaxed beyond level 1 — anything looser is just a local atom-environment fingerprint, no longer a readable structure, and can't reliably handle a position with 2-3 attachment points. Only cuts with matching pair data in the dataset are listed below.")}</p>')
    atom_groups = [(atoms, color) for _, atoms, color in colored if color]
    if atom_groups:
        parts.append(
            f'<div class="site-hero">{_svg_mol_regions(mol, atom_groups)}'
            f'<p class="note">{L("同一结构上不同颜色对应下方不同的切割位点（按 MMP 支持数量从高到低分配颜色；两个切割方式覆盖同一位置时只保留支持更多的一组）。", "Different colors on this one structure mark the different cut positions below (assigned by MMP-pair support, highest first; where two cuts cover the same position only the better-supported one is kept).")}</p></div>'
        )
    parts.append('<div class="site-grid">')
    for s, atoms, color in colored[:24]:
        lvtxt = f' (level {s["level"]})' if s['level'] is not None else ''
        tag_style = f' style="background:{color};color:#fff"' if color else ''
        card_style = f' style="border-top:3px solid {color}"' if color else ''
        anchor_idx = site_anchor.get((s['R1'], s['core']))
        tag = 'a' if anchor_idx is not None else 'div'
        open_tag = f'<{tag} class="site-card" href="#site-{anchor_idx}"{card_style}>' if anchor_idx is not None else f'<{tag} class="site-card"{card_style}>'
        letter = ov_letter.get((s['R1'], s['core']))
        region_txt = f' · {letter} {L("区", "region")}' if letter else ''
        parts.append(
            f'{open_tag}'
            f'<span class="pos-tag"{tag_style}>{html.escape(str(s["cut_type"]))}{region_txt}</span>'
            f'{_svg(s["R1"], 150, 100)}'
            f'<div class="core-row"><span class="core-label">{L("对应核心", "Matched core")}</span>{_svg(s["core"], 110, 70)}</div>'
            f'<div class="freq">{L("同系列", "Same series")} <b>{s["n_same_series"]}</b> · {L("跨系列", "Wider")} <b>{s["n_pool"]}</b>{lvtxt}</div>'
            f'</{tag}>'
        )
    parts.append('</div>')
    if len(colored) > 24:
        parts.append(f'<p class="note">{L(f"另有 {len(colored) - 24} 个有数据的位点未列出。", f"{len(colored) - 24} more cuts with data not shown.")}</p>')
    if n_overlap_dropped:
        parts.append(f'<p class="note">{L(f"另有 {n_overlap_dropped} 种切割因与上方支持更多的位点位置重叠，已合并显示。", f"{n_overlap_dropped} more cuts overlapped a better-supported position above and were merged into it.")}</p>')
    if n_no_data:
        parts.append(f'<p class="note">{L(f"另有 {n_no_data} 种切割在数据集里没有找到对应数据，未列出。", f"{n_no_data} more cuts had no matching data in the dataset and are not shown.")}</p>')

    # 3 per-site tables
    parts.append(f'<h2>4. {L("位点级变换统计", "Transformations by position")}</h2>')
    parts.append(f'<p class="note">{L("把同一个片段在同一类位置上被替换的所有配对放在一起（同一对分子只计一次）。n = 配对数；括号内是核心完全相同的同系列配对数。", "All pairs where this fragment was swapped at the same kind of position (each molecule pair counted once). n = pairs; the figure in brackets is the number from the identical core.")}</p>')
    for idx, (s, rows) in enumerate(site_tables[:max_site_tables]):
        parts.append(f'<div id="site-{idx}">')
        parts.append(f'<h3>{L("位点", "Position")}：{html.escape(s["R1"])}  <small>{s["cut_type"]} · level {s["level"]} · {s["n_pool"]} {L("对", "pairs")}</small></h3>')
        core_cell = _mol_cell(s["core_at_level"], 110, 80) if s["core_at_level"] else '–'
        parts.append(f'<div style="display:flex;gap:16px;align-items:center">{_svg(s["R1"], 140, 90)}<span class="note">{L("匹配的核心（抽象后）", "Matched core (abstracted)")}：</span>{core_cell}</div>')
        parts.append('<div class="swap-grid">')
        for row in rows[:rows_per_site]:
            ex = row['example']
            ex_name = ex.get('mol2_name')
            ex_name_text = (html.escape(str(ex_name)) if ex_name is not None and str(ex_name) != 'nan'
                            else L('（配对分子）', '(paired molecule)'))
            grade = _grade(row['per'][prim['label']])
            prop_html = ''.join(
                f'<div class="prop-block"><div class="plabel">{html.escape(p["label"])}</div>'
                f'{bar(row["per"][p["label"]]["mean"] if row["per"][p["label"]]["n"] else None, higher=p["higher"])}'
                f'<small>n={row["per"][p["label"]]["n"]}</small></div>'
                for p in props
            )
            parts.append(
                f'<div class="swap-card">'
                f'{_svg(row["R2"], 140, 100)}'
                f'<div class="name-row">{ex_name_text}{grade_badge(grade)}</div>'
                f'<div class="n-row">n={row["n"]}（{row["n_same"]} {L("同核心", "same core")}）</div>'
                f'<div class="props">{prop_html}</div>'
                f'</div>'
            )
        parts.append('</div>')
        if len(rows) > rows_per_site:
            parts.append(f'<p class="note">{L(f"另有 {len(rows) - rows_per_site} 种替换仅少量配对，未列出。", f"{len(rows) - rows_per_site} more swaps with few pairs not shown.")}</p>')
        parts.append('</div>')
    if len(site_tables) > max_site_tables:
        parts.append(f'<p class="note">{L(f"另有 {len(site_tables) - max_site_tables} 个有数据的位点未列出。", f"{len(site_tables) - max_site_tables} more positions with data not shown.")}</p>')

    # 4 direct analogues
    if in_dataset:
        parts.append(f'<h2>5. {L("直接类似物", "Direct analogues")}</h2>')
        if analogs:
            head = ''.join(f'<th>{html.escape(p["label"])}<br><small>{L("实测 / Δ", "value / Δ")}</small></th>' for p in props)
            parts.append(f'<table class="t"><tr><th>{L("类似物", "Analogue")}</th><th>{L("名称", "Name")}</th><th>{L("变化", "Change")}</th>{head}</tr>')
            for r in analogs[:max_analogs]:
                cells = ''
                for p in props:
                    d = pd.to_numeric(r[p['d']], errors='coerce')
                    v2 = pd.to_numeric(r[p['m2']], errors='coerce') if p['m2'] else float('nan')
                    cells += f'<td>{"" if v2 != v2 else f"{v2:.2f}"}{bar(d if d == d else None, higher=p["higher"])}</td>'
                parts.append(f'<tr><td class="st">{_svg(r["mol2_smiles"], 190, 130)}</td><td>{html.escape(str(r.get("mol2_name", "")))}</td>'
                             f'<td>{_transform_cell(r["R1"], r["R2"], 90, 65)}</td>{cells}</tr>')
            parts.append('</table>')
            if len(analogs) > max_analogs:
                parts.append(f'<p class="note">{L(f"按主属性改善幅度排序，只列前 {max_analogs} 个，共 {len(analogs)} 个。", f"Sorted by improvement in the primary property; first {max_analogs} of {len(analogs)} shown.")}</p>')
        else:
            parts.append(f'<p>{L("没有直接类似物。", "No direct analogues.")}</p>')

    # 5 designs
    parts.append(f'<h2>6. {L("推荐的分子设计", "Proposed designs")}</h2>')
    plabels = ', '.join(p['label'] for p in props)
    parts.append(f'<p>{L(f"目标：<b>{html.escape(prim["label"])}</b> 有提升（平均 Δ ≥ {improve_threshold}），其余属性（{html.escape(plabels)}）不明显变差（Δ ≥ −{worsen_tolerance}，有数据时）。方法：以该化合物的实测值为基准，加上同一位点同一替换的平均 Δ；组合设计假设两处效应可相加，<b>这个假设不一定成立</b>。证据等级：", f"Goal: improve <b>{html.escape(prim["label"])}</b> (mean Δ ≥ {improve_threshold}) without clear loss in the other properties ({html.escape(plabels)}; Δ ≥ −{worsen_tolerance} where data exist). Method: start from the compound’s measured value and add the mean Δ of the same swap at the same position; combinations assume the two effects add up, <b>which may not hold</b>. Grades:")} '
                 f'{grade_badge("A")} {L("多对且方向一致", "≥3 consistent pairs")} {grade_badge("B")} {L("多对但不一致，或组合", "several mixed pairs, or a combination")} {grade_badge("C")} {L("仅 1 对", "a single pair")}</p>')
    if designs:
        head = ''.join(f'<th>{html.escape(p["label"])}<br><small>{L("预测 / Δ", "pred. / Δ")}</small></th>' for p in props)
        parts.append(f'<table class="t"><tr><th>#</th><th>{L("结构", "Structure")}</th>{head}<th>{L("证据", "Grade")}</th><th>{L("新颖性", "Novelty")}</th><th>{L("依据", "Basis")}</th></tr>')
        for i, d in enumerate(designs, 1):
            cells = ''
            for p in props:
                rows_ = [d['row']] + ([d['row2']] if d['kind'] == 'combo' else [])
                dl = sum(r['per'][p['label']]['mean'] for r in rows_ if r['per'][p['label']]['n'])
                have = any(r['per'][p['label']]['n'] for r in rows_)
                pv = d['pred'].get(p['label'])
                cells += f'<td>{"" if pv is None else f"{pv:.2f}"}{bar(dl if have else None, higher=p["higher"])}</td>'
            if d['kind'] == 'single':
                basis = (f'{L("位点", "Position")} {_transform_cell(d["site"]["R1"], d["row"]["R2"], 70, 55)}'
                         f'<div class="note">n={d["row"]["per"][prim["label"]]["n"]}</div>')
            else:
                basis = (f'{L("两处替换叠加", "Two swaps combined")}：'
                         f'{_transform_cell(d["site"]["R1"], d["row"]["R2"], 70, 55)}'
                         f'<div class="note">n={d["row"]["per"][prim["label"]]["n"]}</div>'
                         f'{_transform_cell(d["site2"]["R1"], d["row2"]["R2"], 70, 55)}'
                         f'<div class="note">n={d["row2"]["per"][prim["label"]]["n"]}</div>')
            novelty = L('<b>已在数据集中</b>', '<b>already in dataset</b>') if d['smiles'] in all_mols else L('新分子', 'new')
            mw2, lp2, tp2 = (lambda m: (Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m)))(Chem.MolFromSmiles(d['smiles']))
            parts.append(f'<tr><td><b>D{i}</b></td><td class="st">{_mol_cell(d["smiles"], 200, 150)}</td>{cells}'
                         f'<td>{grade_badge(d["grade"])}</td><td>{novelty}</td><td>{basis}<br><small>MW {mw2:.0f} · cLogP {lp2:.1f} · TPSA {tp2:.0f}</small></td></tr>')
        parts.append('</table>')
        parts.append(f'<p class="note">{L("“已在数据集中”表示这个结构已经被合成并出现在数据里，可直接查看它的实测值，不算新设计。预测值只在属性是差值型（subtract）时给出。", "“Already in dataset” means the structure was already made and appears in the data, so its measured value can be looked up; it is not a new design. Predicted values are only given for difference-type (subtract) properties.")}</p>')
    else:
        parts.append(f'<p>{L("按上述条件，没有可推荐的设计：有数据的位点里没有同时满足“主属性提升、其余不变差”的替换。", "No design meets the criteria: no swap at a position with data improves the primary property without hurting the others.")}</p>')
    if worse:
        parts.append(f'<h3>{L("已评估但不推荐（明显变差且至少 2 对）", "Evaluated but not recommended (clearly worse, ≥2 pairs)")}</h3>')
        parts.append(f'<table class="t"><tr><th>{L("位点", "Position")}</th><th>{L("替换为", "Replaced by")}</th><th>n</th><th>{html.escape(prim["label"])} {L("平均 Δ", "mean Δ")}</th></tr>')
        for s, row in worse[:8]:
            parts.append(f'<tr><td class="st">{_svg(s["R1"], 110, 80)}</td><td class="st">{_svg(row["R2"], 110, 80)}</td><td>{row["n"]}</td><td>{bar(row["per"][prim["label"]]["mean"], higher=prim["higher"])}</td></tr>')
        parts.append('</table>')

    # 6 limitations
    parts.append(f'<h2>7. {L("局限", "Limitations")}</h2><div class="warn">{L("① 位点级统计来自整个数据集，可能混合了不同系列、不同实验批次的数据，跨系列的数字不等于它在这个化合物上的预期效果。② 多数变换只有 1–3 对，没有显著性检验，只能当方向参考。③ 组合设计的预测值是加和假设，两处取代基之间可能有相互作用。④ 只涉及数据集里已有的属性，不含 hERG、代谢稳定性、溶解度等其他信息。⑤ 报告由程序按固定模板生成，不含人工判断。", "(1) Position-level statistics pool the whole dataset, possibly mixing series and assay batches; a wider-match figure is not the expected effect on this compound. (2) Most swaps have 1–3 pairs and no significance test — treat them as directional. (3) Combination predictions assume additivity; the two substituents may interact. (4) Only the properties in this dataset are considered (no hERG, metabolic stability, solubility, …). (5) The report is produced by fixed templates with no human judgement.")}</div>')

    # 8 master molecule list — every molecule any section-2 fragment card
    # links to (built up via _record_molecule while section 2 was rendered),
    # one row per molecule, each row tagged data-frags="<frag ids...>" so
    # filterMolList() (the embedded <script> below) can show just the ones
    # carrying whichever fragment the user clicked.
    parts.append(f'<h2 id="molecule-list-anchor">8. {L("总分子列表", "Master molecule list")}</h2>')
    parts.append(f'<p class="note">{L("这里列出了第 2 节 SAR 先行图涉及到的所有化合物——点击第 2 节的任意一张片段卡片，会自动跳到这里并只显示带有该片段的化合物；点击下面的“显示全部”清除筛选。", "Every molecule referenced anywhere in section 2’s SAR overview — clicking any fragment card there jumps here and filters the table down to just the molecules carrying that fragment; click “Show all” below to clear the filter.")}</p>')
    parts.append(f'<div id="molist-filter-note" style="display:none;margin:6px 0 10px">'
                 f'<span class="note">{L("已按片段筛选", "Filtered by fragment")}</span> '
                 f'<button onclick="clearMolListFilter()" class="btn-clear-filter">{L("显示全部", "Show all")}</button></div>')
    mol_list = sorted(master_molecules.values(), key=lambda m: (m['name'] or '~', m['smiles']))
    if mol_list:
        head = ''.join(f'<th>{html.escape(p["label"])}</th>' for p in props)
        parts.append(f'<table class="t" id="molist-table"><tr><th>#</th><th>{L("结构", "Structure")}</th>'
                     f'<th>{L("名称", "Name")}</th>{head}</tr>')
        for idx, m in enumerate(mol_list, 1):
            cells = ''.join(f'<td>{n_str(m["values"].get(p["label"]))}</td>' for p in props)
            frag_attr = ' '.join(sorted(m['frag_ids']))
            name_text = html.escape(m['name']) if m['name'] else L('（无名称）', '(unnamed)')
            parts.append(f'<tr data-frags="{frag_attr}"><td>{idx}</td><td class="st">{_mol_cell(m["smiles"], 130, 90)}</td>'
                         f'<td>{name_text}</td>{cells}</tr>')
        parts.append('</table>')
        parts.append(f'<p class="note">{L(f"共 {len(mol_list)} 个化合物。", f"{len(mol_list)} compounds total.")}</p>')
    else:
        parts.append(f'<p>{L("没有可列出的化合物——该化合物在数据集里没有任何有数据的切割位点。", "No compounds to list — this compound has no cut positions with data in the dataset.")}</p>')

    parts.append('''
<script>
function filterMolList(fragId) {
  document.querySelectorAll('#molist-table tr[data-frags]').forEach(function(tr) {
    var frags = (tr.getAttribute('data-frags') || '').split(' ');
    tr.style.display = frags.indexOf(fragId) !== -1 ? '' : 'none';
  });
  var note = document.getElementById('molist-filter-note');
  if (note) note.style.display = '';
  var anchor = document.getElementById('molecule-list-anchor');
  if (anchor) anchor.scrollIntoView({behavior: 'smooth', block: 'start'});
}
function clearMolListFilter() {
  document.querySelectorAll('#molist-table tr[data-frags]').forEach(function(tr) { tr.style.display = ''; });
  var note = document.getElementById('molist-filter-note');
  if (note) note.style.display = 'none';
}
</script>
''')
    parts.append('</body></html>')

    summary = {
        'smiles': q, 'in_dataset': in_dataset, 'name': name,
        'measured': {k: v for k, v in base.items()},
        'ranks': {k: {'rank': v[0], 'of': v[1]} for k, v in ranks.items()},
        'n_direct_analogs': len(analogs), 'n_sites': len(sites), 'n_sites_with_data': len(sites_with_data),
        'n_designs': len(designs),
        'designs': [{'smiles': d['smiles'], 'grade': d['grade'], 'kind': d['kind'],
                     'predicted': {k: (round(v, 3) if v is not None else None) for k, v in d['pred'].items()},
                     'already_in_dataset': d['smiles'] in all_mols} for d in designs],
        'best_swap': ({'from': best[1]['R1'], 'to': best[2]['R2'], 'mean_delta': round(best[2]['per'][prim['label']]['mean'], 3),
                       'n': best[2]['per'][prim['label']]['n']} if best else None),
    }
    return '\n'.join(parts), summary
