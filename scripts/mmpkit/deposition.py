import os
import pandas as pd
from rdkit import Chem
from .stats import group_extended_stats


def _safe_prop_float(val):
    """Parse an SDF property value as float, returning NaN (not raising) on anything
    unparsable — e.g. a censored/qualified reading like '>10' or '<0.1' (R07: these are
    real 'this assay was run' results, just not resolved to an exact number). Before this
    existed, extract_mmp_data/extract_mmp_data_all_levels called bare float() here, so a
    single censored value anywhere in the property crashed the entire deposition instead of
    degrading to NaN for that one cell, same as a genuinely missing property already did."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return float('nan')


def _property_display_name(col):
    """'property__Solubility' -> 'Solubility'; legacy 'property' col -> 'property'."""
    return col[len('property__'):] if col.startswith('property__') else col


def _detect_property_columns(df):
    """
    多属性数据集里每个属性各占一列 property__<name>；旧的单属性 CSV 只有一个
    字面量 'property' 列。两种情况统一返回属性列名列表，供 groupby 聚合复用。
    """
    cols = [c for c in df.columns if c.startswith('property__')]
    return cols or ['property']


def _resolve_primary_column(df, prop_cols):
    """主属性 = 入库时选中的第一个属性，由 'prop_name' 列记录；找不到就退回第一列。"""
    if 'prop_name' in df.columns and df['prop_name'].notna().any():
        candidate = f"property__{df['prop_name'].dropna().iloc[0]}"
        if candidate in prop_cols:
            return candidate
    return prop_cols[0]


def multi_paired_stats_grouped(df, group_cols, prop_cols, primary_col):
    """
    For each (core,R1,R2,cut_type) group, computes extended_stats per property
    column. The primary property's stats land on the unprefixed traditional
    column names (count/mean/.../p_value), read by mmp_tools.py /
    mmp_sar_network.py / llm_advisor.py; every other property's stats land on
    '{name}__count' / '{name}__mean' / ... prefixed columns. A group missing
    entirely for one property (but present for another) still gets a row here
    — with count=0/all-NaN for that property — since group membership is
    defined by the full frame, not per-property.

    Computed via group_extended_stats (vectorized, see mmp_stats.py) rather
    than a `groupby(group_cols).apply(python_function)` per-group callback,
    which doesn't scale once a dataset's grouping doesn't meaningfully
    collapse its raw rows (e.g. a broad public dataset with near-unique
    core/R1/R2 per row can have millions of groups).
    """
    canonical_index = df.groupby(group_cols, sort=True).size().index

    ordered_cols = [primary_col] + [c for c in prop_cols if c != primary_col]
    parts = []
    for col in ordered_cols:
        stats_df = group_extended_stats(df, group_cols, col).reindex(canonical_index)
        stats_df['count'] = stats_df['count'].fillna(0).astype(int)
        if col != primary_col:
            stats_df = stats_df.add_prefix(f'{_property_display_name(col)}__')
        parts.append(stats_df)

    out = pd.concat(parts, axis=1)
    return out.reset_index()


class Deposition:
    """负责从MMP结果提取数据"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def extract_mmp_data(self, filename, destination_file, property_name='',
                        property_names=None, process=0, dataset=False, sign='*'):
        """从MMP结果中提取数据。property_names 支持一次传入多个属性名（宽表输出，
        每个属性各占一列 property__<name>）；property_name 仍可用于单属性调用。"""
        property_names = property_names or ([property_name] if property_name else [])
        primary_name = property_names[0] if property_names else ''

        self.parent.mols = []
        self.parent.read_sdf(filename, num_of_mols=0, clean=False)

        cores, smiles1_list, smiles2_list, mmp_names, cut_types = [], [], [], [], []
        mol1_smiles_list, mol2_smiles_list = [], []
        prop_values = {p: [] for p in property_names}

        for mol in self.parent.mols:
            prop_dict = mol.GetPropsAsDict()

            if property_names and not any(p in prop_dict for p in property_names):
                print(f'跳过: 分子缺少全部选定属性 {property_names}')
                continue

            mmp_name = mol.GetProp('_Name')
            core_smiles = mol.GetProp('core_smiles')
            core_mol=Chem.MolFromSmiles(core_smiles)
            clean_mol=self.parent.process_core_structures(core_mol,sign=sign,level=process)
            clean_smiles = Chem.MolToSmiles(clean_mol, True)
            frag1_smiles = mol.GetProp('frag1_smiles')
            frag2_smiles = mol.GetProp('frag2_smiles')
            cut_type =mol.GetProp('cut_type')

            cores.append(clean_smiles)
            smiles1_list.append(frag1_smiles)
            smiles2_list.append(frag2_smiles)
            for p in property_names:
                prop_values[p].append(_safe_prop_float(prop_dict[p]) if p in prop_dict else float('nan'))
            mmp_names.append(mmp_name)
            cut_types.append(cut_type)
            mol1_smiles_list.append(mol.GetProp('mol1_smiles'))
            mol2_smiles_list.append(mol.GetProp('mol2_smiles'))

        data = {
            'core': cores,
            'R1': smiles1_list,
            'R2': smiles2_list,
            'mol1_smiles': mol1_smiles_list,
            'mol2_smiles': mol2_smiles_list,
            **{f'property__{p}': vals for p, vals in prop_values.items()},
            'mmp': mmp_names,
            'cut_type': cut_types
        }

        df = pd.DataFrame(data)
        df['property'] = df[f'property__{primary_name}'] if primary_name else 0.0
        df = df.drop_duplicates(keep='first')
        df['filename'] = filename
        df['prop_name'] = primary_name

        print(f"提取的数据:\n{df.head()}")
        
        # 保存到文件
        csv_path = os.path.join(self.parent.working_directory, filename.replace('.sdf', '.csv'))
        df.to_csv(csv_path, index=False)
        
        # 存入数据集
        self.parent._deposit_to_dataset(df, destination_file, dataset=dataset)
        
        # 移动原始文件到历史记录
        self.parent._move_to_history(filename)
        
        return df
    
    def extract_mmp_data_all_levels(self, filename, destination_file,
                                    property_name='', property_names=None, levels=None,
                                    dataset=False, sign='*', raw_property_names=None):
        """
        Read the MMP SDF once, then apply process_core_structures for every
        requested level in-memory.  Returns a single merged DataFrame with
        columns  core_level0 … core_levelN, R1, R2, mol1_smiles, mol2_smiles,
        property (primary-property alias), property__<name> (one per selected
        property), mol1_value/mol2_value (primary-property absolute-value
        aliases), mol1_value__<name>/mol2_value__<name> (one pair per selected
        property — each side's own true measured value, NOT the delta), mmp,
        cut_type.

        property_names accepts multiple property names in one pass (wide-table
        output); property_name remains as a single-property convenience arg.
        The first entry of property_names is treated as the "primary" property
        and is also exposed under the legacy 'property'/'prop_name' columns so
        downstream single-property consumers keep working unchanged.

        raw_property_names: optional {sanitized delta key -> raw SDF property
        name} map (server.py's run_mmp_pipeline builds this via its own
        `_mmp_prop` sanitizer). mmp_frag_and_index.py's _collect_mmp_info tags
        each pair's own absolute values under the RAW property name
        (f'{raw_prop}_mmp_mol1'/'_mol2'), never the sanitized 'subtract_..._mmp'
        token `property_names` itself uses for the delta — this map is how we
        find those tags without re-deriving the raw name here. Falls back to
        using `p` itself as the raw name when no mapping is given (e.g. an
        older MMP SDF, or a caller whose property_names were never sanitized in
        the first place) — either just means no absolute-value tag is found and
        mol1_value/mol2_value end up NaN for that property, same as any other
        "this dataset predates the feature" case in this function.

        This is equivalent to calling extract_mmp_data N times but performs
        only one SDF parse, making it significantly faster.
        """
        if levels is None:
            levels = [0, 1, 2, 3, 4, 5]

        property_names = property_names or ([property_name] if property_name else [])
        primary_name = property_names[0] if property_names else ''
        raw_property_names = raw_property_names or {}

        self.parent.mols = []
        self.parent.read_sdf(filename, num_of_mols=0, clean=False)

        raw_cores, r1s, r2s, mmps, cuts = [], [], [], [], []
        mol1s, mol2s = [], []
        mol1_names, mol2_names = [], []
        prop_values = {p: [] for p in property_names}
        mol1_values = {p: [] for p in property_names}
        mol2_values = {p: [] for p in property_names}

        for mol in self.parent.mols:
            prop_dict = mol.GetPropsAsDict()
            if property_names and not any(p in prop_dict for p in property_names):
                print(f'跳过: 分子缺少全部选定属性 {property_names}')
                continue

            raw_cores.append(mol.GetProp('core_smiles'))
            r1s.append(mol.GetProp('frag1_smiles'))
            r2s.append(mol.GetProp('frag2_smiles'))
            for p in property_names:
                prop_values[p].append(_safe_prop_float(prop_dict[p]) if p in prop_dict else float('nan'))
                raw_name = raw_property_names.get(p, p)
                mol1_values[p].append(_safe_prop_float(prop_dict[f'{raw_name}_mmp_mol1'])
                                       if f'{raw_name}_mmp_mol1' in prop_dict else float('nan'))
                mol2_values[p].append(_safe_prop_float(prop_dict[f'{raw_name}_mmp_mol2'])
                                       if f'{raw_name}_mmp_mol2' in prop_dict else float('nan'))
            mmps.append(mol.GetProp('_Name'))
            cuts.append(mol.GetProp('cut_type'))
            mol1s.append(mol.GetProp('mol1_smiles'))
            mol2s.append(mol.GetProp('mol2_smiles'))
            # Set by mmp_frag_and_index.py's _collect_mmp_info alongside
            # mol1_smiles/mol2_smiles — carried through here so a compound's
            # own name (not just its combined-pair '_Name'/'mmp' value) can be
            # shown next to its structure in the UI's instance-level Details
            # drill-down. Falls back to None on an older MMP SDF that predates
            # this tag, so a re-deposit is required to backfill names.
            mol1_names.append(prop_dict.get('Name_mmp_mol1'))
            mol2_names.append(prop_dict.get('Name_mmp_mol2'))

        # Apply process_core_structures for each level in one pass
        core_cols = {}
        for lv in levels:
            transformed = []
            for smi in raw_cores:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    result = self.parent.process_core_structures(mol, sign=sign, level=lv)
                    transformed.append(Chem.MolToSmiles(result, True))
                except Exception:
                    transformed.append(None)
            core_cols[f'core_level{lv}'] = transformed
            n_uniq = len(set(s for s in transformed if s is not None))
            print(f'  level{lv}: {n_uniq} unique cores')

        df = pd.DataFrame({
            **core_cols,
            'R1': r1s,
            'R2': r2s,
            'mol1_smiles': mol1s,
            'mol2_smiles': mol2s,
            'mol1_name': mol1_names,
            'mol2_name': mol2_names,
            **{f'property__{p}': vals for p, vals in prop_values.items()},
            **{f'mol1_value__{p}': vals for p, vals in mol1_values.items()},
            **{f'mol2_value__{p}': vals for p, vals in mol2_values.items()},
            'mmp': mmps,
            'cut_type': cuts,
        })
        df['property'] = df[f'property__{primary_name}'] if primary_name else 0.0
        df['mol1_value'] = df[f'mol1_value__{primary_name}'] if primary_name else float('nan')
        df['mol2_value'] = df[f'mol2_value__{primary_name}'] if primary_name else float('nan')
        df = df.drop_duplicates(keep='first')
        df['filename'] = filename
        df['prop_name'] = primary_name

        csv_path = os.path.join(self.parent.working_directory, destination_file)
        df.to_csv(csv_path, index=False)
        print(f"提取完成：{len(df)} 行，levels={levels}，保存为 {destination_file}")

        # Only deposit when targeting the permanent dataset store. When
        # dataset=False, _deposit_to_dataset's target dir is the same
        # working_directory we just wrote csv_path to — calling it here would
        # find that file, treat it as pre-existing, and concat df with itself,
        # silently doubling every row.
        if dataset:
            self.parent._deposit_to_dataset(df, destination_file, dataset=dataset)
        self.parent._move_to_history(filename)

        return df

    def summarize_all_levels(self, merged_filename, levels=None,
                             lower_limit=1, dataset=False):
        """
        Generate one summary CSV per level from a merged all-levels CSV
        produced by extract_mmp_data_all_levels.

        For each level, renames core_levelX → core and runs the standard
        groupby aggregation (count, mean, std, min, max).

        Returns a dict {level_int: summary_DataFrame}.
        """
        if levels is None:
            levels = [0, 1, 2, 3, 4, 5]

        data_path = self.parent.dataset_path if dataset else self.parent.working_directory
        merged_df = pd.read_csv(os.path.join(data_path, merged_filename))

        prop_cols = _detect_property_columns(merged_df)
        primary_col = _resolve_primary_column(merged_df, prop_cols)

        stem = merged_filename.rsplit('.', 1)[0]
        summaries = {}

        for lv in levels:
            col = f'core_level{lv}'
            if col not in merged_df.columns:
                print(f'跳过 level{lv}：列 {col} 不存在')
                continue

            df_lv = (merged_df[[col, 'R1', 'R2', 'cut_type'] + prop_cols]
                     .rename(columns={col: 'core'})
                     .dropna(subset=['core']))

            grouped = multi_paired_stats_grouped(
                df_lv, ['core', 'R1', 'R2', 'cut_type'], prop_cols, primary_col)
            filtered = grouped[grouped['count'] > float(lower_limit)]

            summary_name = f'summary_{stem}_level{lv}.csv'
            summary_path = os.path.join(self.parent.working_directory, summary_name)
            filtered.to_csv(summary_path, index=False)
            summaries[lv] = filtered
            print(f'  level{lv}: {len(filtered)} 组，保存为 {summary_name}')

        return summaries

    def summarize_data(self, filename, lower_limit=5, dataset=False, level=None):
        """
        对提取的数据进行统计分析。

        支持两种 CSV 格式：
        - 单 level 格式（含 'core' 列）：原始行为，忽略 level 参数。
        - alllevels 格式（含 'core_level0'…'core_level5' 列，无 'core' 列）：
            level=None  → 调用 summarize_all_levels，生成 6 个 summary 文件，
                          返回 {level: DataFrame} 字典。
            level=int   → 只对指定 level 汇总，生成 summary_<stem>_level<N>.csv，
                          返回该 level 的 DataFrame。
        """
        if dataset:
            data_path = self.parent.dataset_path
        else:
            data_path = self.parent.working_directory

        df0 = pd.read_csv(os.path.join(data_path, filename))
        is_alllevels = 'core_level0' in df0.columns and 'core' not in df0.columns
        prop_cols = _detect_property_columns(df0)
        primary_col = _resolve_primary_column(df0, prop_cols)

        if is_alllevels:
            if level is not None:
                col = f'core_level{level}'
                available = [c for c in df0.columns if c.startswith('core_level')]
                if col not in df0.columns:
                    raise ValueError(
                        f'列 {col} 不存在，可用 level: {[c.replace("core_level","") for c in available]}'
                    )
                df = (df0[[col, 'R1', 'R2', 'cut_type'] + prop_cols]
                      .rename(columns={col: 'core'})
                      .dropna(subset=['core']))
                grouped = multi_paired_stats_grouped(
                    df, ['core', 'R1', 'R2', 'cut_type'], prop_cols, primary_col)
                filtered = grouped[grouped['count'] > float(lower_limit)]
                stem = filename.rsplit('.', 1)[0]
                summary_path = os.path.join(
                    self.parent.working_directory, f'summary_{stem}_level{level}.csv')
                filtered.to_csv(summary_path, index=False)
                print(f"统计摘要 level{level} ({len(filtered)} 组):\n{filtered.head()}")
                return filtered
            else:
                return self.summarize_all_levels(
                    filename, lower_limit=lower_limit, dataset=dataset)
        else:
            df = df0.dropna(subset=['core'])
            print(f"原始数据:\n{df.head()}")
            grouped_df = multi_paired_stats_grouped(
                df, ['core', 'R1', 'R2', 'cut_type'], prop_cols, primary_col)
            filtered_df = grouped_df[grouped_df['count'] > float(lower_limit)]
            summary_path = os.path.join(
                self.parent.working_directory, f'summary_{filename.split(".")[0]}.csv')
            filtered_df.to_csv(summary_path, index=False)
            print(f"统计摘要:\n{filtered_df.head()}")
            return filtered_df
