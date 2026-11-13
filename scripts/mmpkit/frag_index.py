import numpy as np
from collections import defaultdict
from rdkit import Chem
import itertools
import re
import multiprocessing as mp

from . import attachment_order

# Tentative worker count for compare_molecules' parallel cutting stage (see
# below) — profiling on a real 18MB SDF found perform_single_cut/
# perform_double_cut/perform_triple_cut's per-molecule loop to be the actual
# bottleneck of "运行 MMP 分析" (single-threaded, it didn't finish in 10+
# minutes), NOT the core-abstraction level processing that was suspected
# first. Just a module constant for now — bump it later if useful.
MMP_ANALYSIS_WORKERS = 3


def _cut_molecules_chunk(args):
    """Runs in its own process (see compare_molecules below): single/double/
    triple-cuts one shard of molecules and returns its own LOCAL index dicts
    — perform_single_cut/perform_double_cut/perform_triple_cut only ever
    touch `mol`/`mol_id` plus self.cut_bond_smarts/self.max_fragment_size/
    their own index dict (never self.parent), so a worker-local
    MMPIdentifier(parent=None) is safe to cut with; only add_hydrogen_pairs/
    _find_mmps_from_index (which need the FULL merged index + self.parent.mols)
    stay in the main process after all shards are merged back together.

    Molecules are passed as RDKit's own binary serialization (Mol.ToBinary()/
    Chem.Mol(data)) rather than re-parsed from SMILES — cheaper than a second
    SMILES round-trip per molecule and doesn't risk losing stereochemistry
    that a plain SMILES reparse could re-canonicalize differently. Must be a
    module-level function (not a bound method) — multiprocessing pickles the
    worker callable, and a method bound to a live MMPIdentifier instance
    (holding compiled RDKit SMARTS/Mol objects) is not reliably picklable.
    """
    mol_binaries, mol_ids, max_fragment_size = args
    identifier = MMPIdentifier(parent=None, max_fragment_size=max_fragment_size)
    for data, mol_id in zip(mol_binaries, mol_ids):
        mol = Chem.Mol(data)
        identifier.perform_single_cut(mol, mol_id)
        identifier.perform_double_cut(mol, mol_id)
        identifier.perform_triple_cut(mol, mol_id)
    return (dict(identifier.single_cut_index),
            dict(identifier.double_cut_index),
            dict(identifier.triple_cut_index))


class MMPIdentifier:
    def __init__(self, parent, max_fragment_size=10):
        self.parent = parent
        self.max_fragment_size = max_fragment_size
        self.single_cut_index = defaultdict(list)
        self.double_cut_index = defaultdict(list)
        self.triple_cut_index = defaultdict(list)
        self._canonical_smiles_cache = None
        # mmpdb's actual default cut_smarts (the pattern that decides *where to
        # cut*, not the separate rotatable-bond-count filter): left atom must be
        # a neutral carbon not itself double/triple-bonded to a heteroatom (this
        # alone keeps amide/ketone/ester carbonyl-N and C=O bonds intact, with
        # no amide-specific rule needed); right atom must be a real substituent
        # start — not a wildcard/H, and not a bare mid-chain CH2 or the CH2 of a
        # terminal ethyl, which avoids chopping alkyl chains into meaningless
        # interior fragments.
        self.cut_bond_smarts = Chem.MolFromSmarts(
            '[#6+0;!$(*=,#[!#6])]!@!=!#[!#0;!#1;!$([CH2]);!$([CH3][CH2])]')

    def find_acyclic_bonds(self, mol):
        """识别可切割的非环单键（mmpdb 默认 cut_smarts）"""
        bond_set = set()
        for a1, a2 in mol.GetSubstructMatches(self.cut_bond_smarts):
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is not None:
                bond_set.add(bond.GetIdx())
        return list(bond_set)
    
    def canonicalize_fragment(self, fragment_smiles):
        """规范化片段SMILES"""
        mol = Chem.MolFromSmiles(fragment_smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return fragment_smiles

    def _core_with_hydrogen(self, core_smiles):
        """
        把单切割 core 片段上的 * 去掉、换成 H，返回补完后的完整分子 canonical SMILES。
        用于 mmpdb 风格的"1-cut 氢配对"：检查这个补完后的结构是否恰好是数据集里
        另一个真实分子的完整结构。
        """
        mol = Chem.MolFromSmiles(core_smiles)
        if mol is None:
            return None

        dummy_idx = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx = atom.GetIdx()
                break
        if dummy_idx is None:
            return None

        rw_mol = Chem.RWMol(mol)
        rw_mol.RemoveAtom(dummy_idx)
        for atom in rw_mol.GetAtoms():
            atom.SetNoImplicit(False)
        try:
            rw_mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(rw_mol)
        except Exception:
            return None

        return Chem.MolToSmiles(rw_mol, canonical=True)

    def _canonical_hydrogen_smiles(self):
        """
        规范化 '*[H]' 片段的 SMILES。绕开 RDKit 对裸 '[H]' 原子直接连接
        通配符(*)时会打印的 'not removing hydrogen atom with dummy atom
        neighbors' 警告——先用同位素标记版本 '*[1H]' 规范化，再清掉同位素标记
        （mmpdb 的 fragment_algorithm.py 用的是同一个规避手法）。
        """
        mol = Chem.MolFromSmiles('*[1H]')
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        return Chem.MolToSmiles(mol)

    def add_hydrogen_pairs(self):
        """
        mmpdb 风格的"1-cut 氢配对"：对每个已出现在 single_cut_index 里的单切割
        core，检查"把取代基替换为 H"之后的完整分子，是否恰好等于数据集中
        另一个真实分子的整体结构。如果是，把该分子当作这个核心位置上
        "取代基=H"的配对对象加入索引——即使数据集里没有专门收录/测量过这个
        母核化合物对应的"母核+H"记录本身也没关系，用到的只是"它恰好是数据集中
        另一个真实分子"这一事实。这能显著增加卤素扫描一类单点取代数据里可用的
        MMP 数量（母核本身如果也被测过，就能推出 X→H 的变换）。

        必须在所有分子都完成 perform_single_cut 之后、_find_mmps_from_index
        之前调用，且只作用于 single_cut_index（mmpdb 同样只对 1-cut 做这个处理）。
        """
        full_smiles_to_id = {}
        for mol_id, mol in enumerate(self.parent.mols):
            try:
                full_smiles_to_id.setdefault(Chem.MolToSmiles(mol), mol_id)
            except Exception:
                continue

        canon_h = self._canonical_hydrogen_smiles()
        for core_key in list(self.single_cut_index.keys()):
            core_with_h = self._core_with_hydrogen(core_key)
            if core_with_h is None:
                continue
            match_id = full_smiles_to_id.get(core_with_h)
            if match_id is None:
                continue
            self.single_cut_index[core_key].append((canon_h, match_id))
    
    def get_fragment_size(self, fragment_smiles):
        """计算片段中非氢原子数量"""
        mol = Chem.MolFromSmiles(fragment_smiles)
        if mol:
            return mol.GetNumHeavyAtoms()
        return 0
    
    def perform_single_cut(self, mol, mol_id):
        """执行单切割并建立索引 - 使用FragmentOnBonds"""
        acyclic_bonds = self.find_acyclic_bonds(mol)
        
        for bond_idx in acyclic_bonds:
            try:
                # 使用FragmentOnBonds进行切割
                fragments=self.cut_mol_from_bonds(mol,[bond_idx])
                
                if len(fragments) == 2:
                    frag1, frag2 = fragments
                    
                    # 将片段转换为SMILES
                    frag1_smiles = Chem.MolToSmiles(frag1)
                    frag2_smiles = Chem.MolToSmiles(frag2)
                    #print(f'perform single cuts: {frag1_smiles} and {frag2_smiles}')
                    # 检查片段大小
                    if (self.get_fragment_size(frag1_smiles) <= self.max_fragment_size or 
                        self.get_fragment_size(frag2_smiles) <= self.max_fragment_size):
                        
                        # 规范化片段
                        canon_frag1 = self.canonicalize_fragment(frag1_smiles)
                        canon_frag2 = self.canonicalize_fragment(frag2_smiles)
                        
                        
                        # 建立双向索引
                        if self.get_fragment_size(frag2_smiles) <= self.max_fragment_size:
                            self.single_cut_index[canon_frag1].append((canon_frag2, mol_id))
                        if self.get_fragment_size(frag1_smiles) <= self.max_fragment_size:
                            self.single_cut_index[canon_frag2].append((canon_frag1, mol_id))
                        
            except Exception as e:
                print(f"单切割错误 (分子 {mol_id}, 键 {bond_idx}): {e}")
                continue
    
    def perform_double_cut(self, mol, mol_id):
        """执行双切割并建立索引"""
        acyclic_bonds = self.find_acyclic_bonds(mol)

        # 生成所有非相邻键对组合
        for bond_pair in itertools.combinations(acyclic_bonds, 2):
            bond1, bond2 = bond_pair

            # 检查键是否相邻（共享原子）
            if not self.are_bonds_adjacent(mol, bond1, bond2):
                try:
                    info = self._fragment_multi_cut(mol, list(bond_pair))
                    if info is None:
                        continue
                    variable_smiles, key, sym_class, attach_order, distance_sig = info
                    if self.get_fragment_size(variable_smiles) <= self.max_fragment_size:
                        self.double_cut_index[key].append(
                            (variable_smiles, mol_id, sym_class, attach_order, distance_sig))
                except Exception as e:
                    print(f"双切割错误 (分子 {mol_id}, 键对 {bond_pair}): {e}")
                    continue

    def cut_mol_from_bonds(self,mol,bond_list):
        re_match = r'\[[0-9]*\*\]'
        fragmented_mol = Chem.FragmentOnBonds(mol, bond_list)
        # 获取切割后的片段
        frag_smiles = Chem.MolToSmiles(fragmented_mol, True)
        fragments=[]
        for frag_smi in frag_smiles.split('.'):
            new_smi=re.sub(re_match, '*', frag_smi)
            fragments.append(Chem.MolFromSmiles(new_smi))
        return fragments

    def _fragment_multi_cut(self, mol, bond_idx_list):
        """
        双/三切割共用：一次性切开 len(bond_idx_list)（2 或 3）条键，同时
        追踪"可变片段（未来的 R1/R2）的每个连接点到底连的是哪个不变片段"，
        供 mmp_attachment_order 用来算出连接点对齐一致的 SMIRKS——而不是
        像旧版那样把每个碎片独立转成 SMILES 再独立重新解析，直接丢掉这个
        对应关系。

        做法：给每一刀两侧的哑原子打上相同的同位素编号（`dummyLabels`），
        这样同位素编号就是"这个连接点属于哪一刀"的天然标签；不变片段各自
        只有 1 个连接点，独立解析后记下它的同位素编号和去同位素后的规范
        SMILES；可变片段解析后，按它自己 SMILES 里 * 出现的顺序读出各连接
        点的同位素编号，查表换成"不变片段的第几个"，就是 attachment_order。

        返回 (variable_smiles, constant_key, variable_symmetry_class,
        attachment_order, distance_sig)，形状不对（比如某一刀切完不是"1 个
        多连接点核心 + N 个单连接点末端"）时返回 None。distance_sig 是
        {不变片段编号的无序对: 两个连接点之间的最短路径键数} ——同一个
        core 下比较两条记录时，靠这张表过滤掉"连接点之间跨的键数对不上"
        的配对（比如把三氟甲基硬劈开、多绕一个碳的那种切法）。
        """
        num_cuts = len(bond_idx_list)
        dummy_labels = [(i + 1, i + 1) for i in range(num_cuts)]
        fragmented_mol = Chem.FragmentOnBonds(mol, bond_idx_list, dummyLabels=dummy_labels)
        frag_smiles_list = Chem.MolToSmiles(fragmented_mol).split('.')
        if len(frag_smiles_list) != num_cuts + 1:
            return None

        variable_idx = None
        terminal_idxs = []
        for i, frag_smi in enumerate(frag_smiles_list):
            n_star = frag_smi.count('*')
            if n_star == num_cuts:
                if variable_idx is not None:
                    return None
                variable_idx = i
            elif n_star == 1:
                terminal_idxs.append(i)
            else:
                return None
        if variable_idx is None or len(terminal_idxs) != num_cuts:
            return None

        # 不变（末端）片段：各自只有 1 个连接点，没有连接顺序歧义，直接
        # 独立规范化即可，和旧版行为一致。
        terminal_entries = []  # [(isotope, canonical_smiles), ...]
        for i in terminal_idxs:
            tmol = Chem.MolFromSmiles(frag_smiles_list[i], sanitize=False)
            Chem.SanitizeMol(tmol)
            dummy = next(a for a in tmol.GetAtoms() if a.GetAtomicNum() == 0)
            isotope = dummy.GetIsotope()
            dummy.SetIsotope(0)
            terminal_entries.append((isotope, Chem.MolToSmiles(tmol)))

        sorted_entries = sorted(terminal_entries, key=lambda t: t[1])
        rgroup_label_of_isotope = {iso: idx for idx, (iso, _smi) in enumerate(sorted_entries)}
        key = '.'.join(smi for _iso, smi in sorted_entries)

        # 可变片段（R1/R2）。
        vmol = Chem.MolFromSmiles(frag_smiles_list[variable_idx], sanitize=False)
        Chem.SanitizeMol(vmol)
        dummy_atoms_raw = [(a.GetIdx(), a.GetIsotope())
                           for a in vmol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummy_atoms_raw) != num_cuts:
            return None

        # 对称性判断要在"去掉同位素"的副本上算 canonical rank——同位素本身
        # 就会让两个连接点被当成不同原子（isotope 是 rank 算法的输入之一），
        # 掩盖掉真正的拓扑对称性，必须先去标记再算。
        rank_mol = Chem.Mol(vmol)
        for a in rank_mol.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetIsotope(0)
        ranks = list(Chem.CanonicalRankAtoms(rank_mol, breakTies=False))
        rank_values = [ranks[idx] for idx, _iso in dummy_atoms_raw]
        variable_symmetry_class = attachment_order.get_symmetry_class(*rank_values)

        # 最终展示用的裸 SMILES：和旧版一样独立重新规范化（不带同位素），
        # 保证同一个 R 基团不管从哪个分子切出来，字符串都完全一致，才能
        # 正确按片段分组。
        variable_smiles = self.canonicalize_fragment(
            re.sub(r'\[\d+\*\]', '*', frag_smiles_list[variable_idx]))

        # 连接顺序必须按"最终这个独立重新规范化过的字符串"里 * 出现的顺序来
        # 算——不能直接用去同位素之前那个字符串的顺序，因为 RDKit 独立重新
        # 规范化时可能换一个起始原子/遍历方向，把两个连接点的先后次序也
        # 带着换了。用子结构匹配把两边的原子对应起来，再按最终顺序重新读出
        # 同位素、查回不变片段编号。
        final_mol = Chem.MolFromSmiles(variable_smiles, sanitize=False)
        Chem.SanitizeMol(final_mol)
        match = final_mol.GetSubstructMatch(rank_mol)
        if not match:
            return None
        final_dummy_atoms = sorted(
            (match[raw_idx], iso) for raw_idx, iso in dummy_atoms_raw)
        attach_order = ''.join(
            str(rgroup_label_of_isotope[iso]) for _final_idx, iso in final_dummy_atoms)

        # 每一对连接点之间隔了几个键——用来过滤"同一对不变片段之间存在多种
        # 合法切法"时那些明显不对等的切法（比如把一个三氟甲基硬劈开、让
        # 其中一个连接点比另一边多绕一个碳），键 = 两个连接点各自对应的
        # 不变片段编号（无序对），值 = 最短路径键数。同一个 core 下如果两条
        # 记录的这张表对不上，说明两个 R 基团在骨架上占据的"槽位"形状不同，
        # 不该被当作同一种取代基变换配对。
        distance_sig = {}
        for i in range(len(final_dummy_atoms)):
            for j in range(i + 1, len(final_dummy_atoms)):
                idx_a, iso_a = final_dummy_atoms[i]
                idx_b, iso_b = final_dummy_atoms[j]
                rgroup_pair = frozenset((rgroup_label_of_isotope[iso_a],
                                         rgroup_label_of_isotope[iso_b]))
                path = Chem.GetShortestPath(final_mol, idx_a, idx_b)
                distance_sig[rgroup_pair] = len(path) - 1

        return variable_smiles, key, variable_symmetry_class, attach_order, distance_sig

    def perform_triple_cut(self, mol, mol_id):
        """执行三切割并建立索引"""
        acyclic_bonds = self.find_acyclic_bonds(mol)

        for bond_triplet in itertools.combinations(acyclic_bonds, 3):
            if not self.are_bonds_adjacent_triplet(mol, bond_triplet):
                try:
                    info = self._fragment_multi_cut(mol, list(bond_triplet))
                    if info is None:
                        continue
                    variable_smiles, key, sym_class, attach_order, distance_sig = info
                    if self.get_fragment_size(variable_smiles) <= self.max_fragment_size:
                        self.triple_cut_index[key].append(
                            (variable_smiles, mol_id, sym_class, attach_order, distance_sig))
                except Exception as e:
                    print(f"三切割错误 (分子 {mol_id}, 键三重 {bond_triplet}): {e}")
                    continue
    
    def are_bonds_adjacent(self, mol, bond_idx1, bond_idx2):
        """检查两个键是否相邻（共享原子）"""
        bond1 = mol.GetBondWithIdx(bond_idx1)
        bond2 = mol.GetBondWithIdx(bond_idx2)
        
        atoms_bond1 = {bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()}
        atoms_bond2 = {bond2.GetBeginAtomIdx(), bond2.GetEndAtomIdx()}
        
        return bool(atoms_bond1.intersection(atoms_bond2))
    
    def are_bonds_adjacent_triplet(self, mol, bond_triplet):
        """检查三个键中是否有任意两个相邻"""
        for i in range(len(bond_triplet)):
            for j in range(i+1, len(bond_triplet)):
                if self.are_bonds_adjacent(mol, bond_triplet[i], bond_triplet[j]):
                    return True
        return False
    
    def compare_molecules(self, filename, jobname, sign='*'):
        """主函数：识别所有MMP"""
        print("开始MMP识别...")

        self.parent.mols = []
        self.parent.read_sdf(filename, num_of_mols=0, clean=False)
        self._canonical_smiles_cache = None

        # 第一步：对所有分子进行切割和索引——按分子数平均分成
        # MMP_ANALYSIS_WORKERS 份，各自在独立进程里跑 perform_single_cut/
        # perform_double_cut/perform_triple_cut（这三个方法只碰 mol/mol_id +
        # 自己的索引字典，不碰 self.parent，天然可以在 worker 里用一个全新的
        # MMPIdentifier(parent=None) 实例安全地跑），跑完把每个 worker 自己的
        # 局部索引按 key 合并回 self.single_cut_index/double_cut_index/
        # triple_cut_index——这一步实测是"运行 MMP 分析"里真正的耗时大头
        # （单线程对一个真实 18MB SDF 跑了 10+ 分钟没跑完），不是 core
        # abstraction level 处理那一段。
        valid_mols = [(i, mol) for i, mol in enumerate(self.parent.mols) if mol]
        mol_binaries = [mol.ToBinary() for _, mol in valid_mols]
        mol_ids = [i for i, _ in valid_mols]
        n_workers = min(MMP_ANALYSIS_WORKERS, len(mol_binaries)) or 1
        chunks = np.array_split(np.arange(len(mol_binaries)), n_workers)
        tasks = [([mol_binaries[i] for i in chunk], [mol_ids[i] for i in chunk], self.max_fragment_size)
                 for chunk in chunks if len(chunk)]

        print(f"切割 {len(mol_binaries)} 个分子（{len(tasks)} 个进程并行）...")
        if len(tasks) > 1:
            with mp.Pool(processes=len(tasks)) as pool:
                results = pool.map(_cut_molecules_chunk, tasks)
        else:
            results = [_cut_molecules_chunk(t) for t in tasks]

        for single_part, double_part, triple_part in results:
            for k, v in single_part.items():
                self.single_cut_index[k].extend(v)
            for k, v in double_part.items():
                self.double_cut_index[k].extend(v)
            for k, v in triple_part.items():
                self.triple_cut_index[k].extend(v)

        # 1-cut 氢配对：数据集里若同时有取代分子和它对应的母核（H）分子，
        # 补一条虚拟的 R1=*[H] 配对，增加单点取代类数据的可用 MMP 数量
        self.add_hydrogen_pairs()

        # 第二步：从索引中识别MMP
        mmps = []

        # 单切割MMP
        mmps.extend(self._find_mmps_from_index(self.single_cut_index, 'single'))
        # 双切割MMP  
        mmps.extend(self._find_mmps_from_index(self.double_cut_index, 'double'))
        # 三切割MMP
        mmps.extend(self._find_mmps_from_index(self.triple_cut_index, 'triple'))

        self._collect_mmp_info(mmps,jobname)
        self.summary(mmps)
        return mmps

    def _collect_mmp_info(self, mmps, jobname):
        """收集MMP对的信息"""

        save_list=[]
        for mmp in mmps:
            core_smiles=mmp['key']
            core = Chem.MolFromSmiles(core_smiles)
            #core_smiles = Chem.MolToSmiles(core_mol, True)
            core.SetProp('core_smiles', core_smiles)
            core.SetProp('cut_type', mmp['cut_type'])

            frag1=self.parent.mols[mmp['mol_id1']]
            frag2=self.parent.mols[mmp['mol_id2']]
            core1=Chem.Mol(core)
            core2=Chem.Mol(core)
            R1=mmp['frag1']
            R2=mmp['frag2']

            # smirks/transformation 都是方向性的（frag1>>frag2），core1 这份
            # 记录的 frag1_smiles/frag2_smiles 是 R1/R2 顺着来，core2 那份是反过来
            # 的（R2/R1）——两份各自的 smirks/transformation 也要跟着方向反转，
            # 不能像 core_smiles/cut_type 那样直接共用同一份，否则反向那份记录
            # 里 smirks 显示的连接关系会跟它自己的 frag1_smiles/frag2_smiles 对不上。
            smirks_lhs, smirks_rhs = mmp['smirks'].split('>>', 1)
            core1.SetProp('smirks', mmp['smirks'])
            core2.SetProp('smirks', f'{smirks_rhs}>>{smirks_lhs}')
            core1.SetProp('transformation', mmp['transformation'])
            core2.SetProp('transformation', f'{R2}>>{R1}')

            results=[[core1,frag1,frag2,R1,R2],[core2,frag2,frag1,R2,R1]]
            
            for mols in results:
                # 添加分子1的属性
                mol,mol1,mol2,Rgroup1,Rgroup2=mols
                prop_dict1 = mol1.GetPropsAsDict()
                for prop in prop_dict1.keys():
                    mol.SetProp(f'{prop}_mmp_mol1', str(prop_dict1[prop]))
                mol.SetProp('Name_mmp_mol1', mol1.GetProp('_Name'))
                
                # 添加分子2的属性
                prop_dict2 = mol2.GetPropsAsDict()
                for prop in prop_dict2.keys():
                    mol.SetProp(f'{prop}_mmp_mol2', str(prop_dict2[prop]))
                mol.SetProp('Name_mmp_mol2', mol2.GetProp('_Name'))
                
                # 计算性质差异（用 try/float 兼容数值型和字符串型数值属性）
                for prop in prop_dict1.keys():
                    if prop in prop_dict2.keys():
                        prop1 = prop_dict1[prop]
                        prop2 = prop_dict2[prop]
                        prop_name = prop.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')
                        try:
                            v1 = float(prop1)
                            v2 = float(prop2)
                            mol.SetProp(f'subtract_{prop_name}_mmp', str(v2 - v1))
                            if v1 != 0.0:
                                mol.SetProp(f'divide_{prop_name}_mmp', str(v2 / v1))
                        except (TypeError, ValueError):
                            pass
                
                # 添加片段信息
                mol.SetProp('mol1_smiles', Chem.MolToSmiles(mol1))
                mol.SetProp('mol2_smiles', Chem.MolToSmiles(mol2))
                mol.SetProp('frag1_smiles', Rgroup1)
                mol.SetProp('frag2_smiles', Rgroup2)
                #mol.SetProp('reaction', f'{frag1_smi}>>{frag2_smi}')
                mol.SetProp('_Name', f'{mol1.GetProp("_Name")}_{mol2.GetProp("_Name")}')
                save_list.append(mol)
        self.parent.save_molecules(save_list, 'sdfs', '%s_mmp' % jobname.split('.')[0])
        
        
    
    def _get_canonical_smiles_cache(self):
        """
        每个真实分子（按 self.parent.mols 的下标）的整体 canonical
        SMILES，懒加载一次、全程复用——供 _find_mmps_from_index 判断"这两条
        候选记录是不是同一个真实化合物（哪怕在数据集里登记成了不同名字的
        两条记录）"，避免把重复化合物之间的比较误当成一次真实的取代基变换。
        """
        if self._canonical_smiles_cache is None:
            self._canonical_smiles_cache = {}
            for i, mol in enumerate(self.parent.mols):
                try:
                    self._canonical_smiles_cache[i] = Chem.MolToSmiles(mol)
                except Exception:
                    self._canonical_smiles_cache[i] = None
        return self._canonical_smiles_cache

    def _find_mmps_from_index(self, index, cut_type):
        """从索引中找出MMP"""
        mmps = []
        mol_smiles_cache = self._get_canonical_smiles_cache()
        relabel_cache = attachment_order.RelabelCache()
        n_skipped_duplicates = 0
        n_skipped_geometry = 0

        for key, values in index.items():
            if len(values) > 1:
                # 生成所有分子对组合
                for i in range(len(values)):
                    for j in range(i+1, len(values)):
                        if cut_type == 'single':
                            frag1, mol_id1 = values[i]
                            frag2, mol_id2 = values[j]
                            sym1 = sym2 = order1 = order2 = None
                            dist1 = dist2 = None
                        else:
                            frag1, mol_id1, sym1, order1, dist1 = values[i]
                            frag2, mol_id2, sym2, order2, dist2 = values[j]

                        if mol_id1 == mol_id2 or frag1 == frag2:  # 确保不是同一个分子
                            continue

                        # 两条记录如果对应的是同一个真实分子（数据集里的重复
                        # 登记，比如同一个化合物的两个复孔），哪怕切法不同、
                        # 片段 SMILES 写法不一样，也不能当成一次真实变换——
                        # mmpdb 本身也不防这个，交给这里直接比较整体结构堵住。
                        m1 = mol_smiles_cache.get(mol_id1)
                        m2 = mol_smiles_cache.get(mol_id2)
                        if m1 is not None and m1 == m2:
                            n_skipped_duplicates += 1
                            continue

                        # 双/三切割里，同一对不变片段之间往往能找到不止一种
                        # 合法切法（比如把三氟甲基硬劈开、让某个连接点多绕
                        # 一个碳）。如果 R1、R2 各自"连接点之间隔了几个键"对
                        # 不上，说明两者在骨架上占的槽位形状不一样，不是真的
                        # 同一种取代基变换，直接跳过（mmpdb 官方实现也会把
                        # 这种情况当成两条独立规则收录，我们这里选择更严格、
                        # 只保留槽位形状一致的配对）。
                        if dist1 is not None and dist1 != dist2:
                            n_skipped_geometry += 1
                            continue

                        if cut_type == 'single':
                            smirks = f"{frag1}>>{frag2}"
                            record_cut_type = cut_type
                        else:
                            # 双/三切割：用对称性感知的连接点匹配算出真正
                            # 对齐一致的 SMIRKS（mmp_attachment_order，移植
                            # 自 mmpdb），而不是瞎拼一个不管连接顺序的占位符。
                            smirks = attachment_order.build_smirks(
                                frag1, sym1, order1, key, frag2, sym2, order2,
                                relabel_cache=relabel_cache)

                            # 核心（key）不对称时，R1/R2 各自哪个连接点接到
                            # 哪个不变片段上是有化学意义的——同一个 frag1/frag2
                            # 字符串在不同朝向下其实是两种不同的真实变换，
                            # 不打 tag 的话会被 (core,R1,R2,cut_type) 分组
                            # 静默合并成一条虚假规则。把 mmpdb 移植过来的
                            # 对称性最小字典序 tag（get_orientation_tag）拼进
                            # cut_type——同一个真实变换（不管从哪个分子对切
                            # 出来）总是得到相同的 tag，真正不同朝向的变换会
                            # 得到不同的 tag；核心对称时所有朝向本就等价，
                            # tag 自然收敛成一个，不会被误拆分。这样下游所有
                            # 已经把 cut_type 当分组键之一的地方（deposition/
                            # server 的 groupby、row_details 的过滤）都不需要
                            # 改一行。
                            orientation_tag = attachment_order.get_orientation_tag(
                                sym1, order1, key, sym2, order2)
                            record_cut_type = f'{cut_type}#{orientation_tag}'

                        mmps.append({
                            'key': key,
                            'mol_id1': mol_id1,
                            'mol_id2': mol_id2,
                            'smirks': smirks,
                            'cut_type': record_cut_type,
                            'frag1': frag1,
                            'frag2': frag2,
                            'transformation': f"{frag1}>>{frag2}"
                        })

        if n_skipped_duplicates:
            print(f"跳过 {n_skipped_duplicates} 对重复结构之间的虚假配对（{cut_type}）")
        if n_skipped_geometry:
            print(f"跳过 {n_skipped_geometry} 对连接点槽位形状不一致的配对（{cut_type}）")
        return mmps
    
    def get_statistics(self):
        """获取算法统计信息"""
        single_cut_pairs = sum(len(values) for values in self.single_cut_index.values())
        double_cut_pairs = sum(len(values) for values in self.double_cut_index.values())
        triple_cut_pairs = sum(len(values) for values in self.triple_cut_index.values())
        
        return {
            'single_cut_fragments': len(self.single_cut_index),
            'double_cut_fragments': len(self.double_cut_index),
            'triple_cut_fragments': len(self.triple_cut_index),
            'total_potential_pairs': single_cut_pairs + double_cut_pairs + triple_cut_pairs
        }
    
    def summary(self,mmps):
        print(f"\n共找到 {len(mmps)} 个MMP")
        # 输出前10个MMP
        for i, mmp in enumerate(mmps[:]):
            print(f"MMP {i+1}:")
            print(f"  分子对: {mmp['mol_id1']} - {mmp['mol_id2']}")
            print(f"  SMIRKS: {mmp['smirks']}")
            print(f"  切割类型: {mmp['cut_type']}")
            print(f"  变换: {mmp['transformation']}")
            print()

def example_usage():
    """使用示例"""
    # 示例分子列表
    smiles_list = [
        "c1ccccc1O",  # 苯酚
        "c1ccccc1N",  # 苯胺
        "c1ccccc1C",  # 甲苯
        "Cc1ccccc1O", # 甲基苯酚
        "Cc1ccccc1N", # 甲基苯胺
        "Cc1ccncc1N", # 甲基吡啶胺
        "BrC1=CC(F)=CC(C)=C1",
        "BrC1=CC(F)=NC(C)=C1"
    ]
    
    # 初始化MMP识别器
    mmp_identifier = MMPIdentifier(max_fragment_size=10)
    
    # 识别MMP
    mmps = mmp_identifier.identify_mmps(smiles_list)
    
    print(f"共找到 {len(mmps)} 个MMP")
    
    # 输出前10个MMP
    for i, mmp in enumerate(mmps[:]):
        if mmp['cut_type']!='triple':
            continue
        print(f"MMP {i+1}:")
        print(f"  分子对: {mmp['mol_id1']} - {mmp['mol_id2']}")
        print(f"  SMIRKS: {mmp['smirks']}")
        print(f"  切割类型: {mmp['cut_type']}")
        print(f"  变换: {mmp['transformation']}")
        print()

# 运行示例
if __name__ == "__main__":
    example_usage()