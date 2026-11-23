"""Robust SDF record reading, shared by server.py (分子列表 upload), main.py
(MMP.read_sdf, i.e. the whole dataset-building pipeline) and mmp_tools.py (property
probes).

Why not Chem.SDMolSupplier: it silently drops every OTHER record when a record's
last data-field value is not followed by a blank line before '$$$$'. Actelion/Glide
exports write exactly that shape ('...Yes\\n$$$$\\n<next title>'), and the supplier
then treats the next record's title line as part of the previous record's field
block, so that record is discarded — it yields neither a molecule nor a None, so
nothing downstream can notice or count it. Measured: 3344-record file -> 1672
molecules, with no error beyond 'Problems encountered parsing data fields', and
'成功读取 1672 个分子' printed as if that were the whole file.

Reading the text ourselves and parsing each record with MolFromMolBlock reads all of
them, and a genuinely broken record can be counted and reported.
"""
import re

_RECORD_SPLIT = re.compile(r'(?m)^\$\$\$\$[ \t]*\n?')
_TAG = re.compile(r'^>+\s*<([^>]*)>')


def split_records(text):
    """SDF text -> list of record chunks. Line endings are normalised first: the same
    export pipeline writes some files with CRLF, and a '$$$$\\r' separator does not
    match a plain '$$$$' line. Leading blank lines are NOT stripped — in a record with
    an empty title line the record legitimately starts with a newline, and dropping it
    shifts the header and makes MolFromMolBlock fail."""
    if isinstance(text, bytes):
        text = text.decode('utf-8', 'replace')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return _RECORD_SPLIT.split(text)


def parse_record(chunk):
    """One record chunk -> (mol, None) or (None, reason). The molfile runs up to
    'M  END'; everything after it is the data-field block ('>  <tag>' + value lines).
    The mol comes back with its SDF properties (and '_Name' when the title is set)
    attached, exactly as SDMolSupplier would hand it over, so fragmentation/
    deposition code that reads mol.GetPropNames() keeps working unchanged."""
    from rdkit import Chem
    if not chunk or not chunk.strip():
        return None, 'empty'
    lines = chunk.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    if not any(ln.strip() for ln in lines):
        return None, 'empty'
    try:
        end = next(k for k, ln in enumerate(lines) if ln.rstrip() == 'M  END')
    except StopIteration:
        return None, 'no M END'
    mol = Chem.MolFromMolBlock('\n'.join(lines[:end + 1]) + '\n', sanitize=True, removeHs=True)
    if mol is None:
        return None, 'parse'
    title = lines[0].strip()
    if title:
        mol.SetProp('_Name', title)
    tag, buf = None, []

    def _flush():
        if tag and tag not in ('SMILES', 'Name') and not tag.startswith('_'):
            mol.SetProp(tag, '\n'.join(buf).strip())

    for ln in lines[end + 1:] + ['']:
        stripped = ln.strip()
        m = _TAG.match(stripped)
        if m:
            _flush()
            tag, buf = m.group(1).strip(), []
            continue
        if stripped == '':
            _flush()
            tag, buf = None, []
            continue
        if tag:
            buf.append(ln.rstrip())
    return mol, None


def read_records(path_or_text, limit=0):
    """Yields (mol, reason) per record; mol is None for a record that could not be
    parsed, with `reason` saying why. limit>0 stops after that many PARSED molecules."""
    if hasattr(path_or_text, 'read'):
        text = path_or_text.read()
    elif isinstance(path_or_text, (bytes, bytearray)):
        text = bytes(path_or_text)
    else:
        with open(path_or_text, 'rb') as f:
            text = f.read()
    n_ok = 0
    for chunk in split_records(text):
        mol, why = parse_record(chunk)
        if why == 'empty':
            continue
        if mol is not None:
            n_ok += 1
        yield mol, why
        if limit and n_ok >= limit:
            return


def read_mols(path_or_text, limit=0):
    """Convenience wrapper: parsed molecules only, plus a failure count."""
    mols, failed = [], 0
    for mol, why in read_records(path_or_text, limit=limit):
        if mol is None:
            failed += 1
        else:
            mols.append(mol)
    return mols, failed
