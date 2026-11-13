"""
Minimal MMP driver for the compound-sar-report skill.

Extracted (unchanged) from the original project's main.py: only the methods the
fragment-index -> all-levels-deposition -> report pipeline actually uses
(SDF reading, core abstraction at levels 0-5, SDF saving). Everything else in
the original class (LLM advisor, suggestion, SAR network, ...) is not needed and
is left out, so this skill depends on no other project file.
"""
import os

from rdkit import Chem
from rdkit.Chem import MolStandardize

from .frag_index import MMPIdentifier
from .deposition import Deposition


class MMP:
    """Working-directory holder + core abstraction (levels 0-5)."""

    def __init__(self, working_directory=''):
        self.working_directory = working_directory or os.getcwd()
        os.makedirs(self.working_directory, exist_ok=True)
        self.mols = []
        self.comparison = MMPIdentifier(self)
        self.deposition = Deposition(self)

    def _move_to_history(self, filename):
        """The original archives the intermediate MMP SDF into a history folder;
        here it simply stays in the working directory."""
        return

    def read_sdf(self, filename, num_of_mols=0, clean=False):
        """读取SDF文件"""
        self.mols = []
        file_path = os.path.join(self.working_directory, filename)
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return
        
        suppl = Chem.SDMolSupplier(file_path)
        chooser = MolStandardize.rdMolStandardize.LargestFragmentChooser()
        
        for i, mol in enumerate(suppl):
            if num_of_mols > 0 and i >= num_of_mols:
                break
            
            if mol is None:
                continue
            
            if clean:
                try:
                    cleaned_mol = chooser.choose(mol)
                    if self._filter_molecule(cleaned_mol):
                        self.mols.append(cleaned_mol)
                except Exception as e:
                    print(f"清洗分子失败: {e}")
                    continue
            else:
                self.mols.append(mol)
            
            print(f"\r读取进度: {i+1}", end="")
        
        print(f'\n成功读取 {len(self.mols)} 个分子\n')

    def _filter_molecule(self, mol):
        """过滤分子（可根据需要扩展）"""
        # 这里可以添加各种过滤条件
        if mol is None:
            return False
        return True

    def process_core_structures(self,mol, sign='*',level=2):
        """统一处理复数个分子:
        level 0 : no process (full core)
        level 1 : remove substituents, keep ring system
        level 2 : local atom-environment SMARTS around each * (radius=2 bonds)
        level 3 : local atom-environment SMARTS around each * (radius=1 bond)
        level 4 : local atom-environment SMARTS around each * (radius=0 — just
                  the directly-bonded atom's own element/aromaticity/bond type)
        level 5 : discard all chemical identity — keep only, per disconnected
                  fragment, HOW MANY * attachment points it has (see
                  _count_only_placeholder)
        """
        if level==0:
            return mol
        mol_smis=Chem.MolToSmiles(mol)
        processed_smiles=[]
        for smi in mol_smis.split('.'):
            frag=Chem.MolFromSmiles(smi)
            if level == 1:
                processed_mol=self._remove_substituents(frag,sign=sign)
                processed_smiles.append(Chem.MolToSmiles(processed_mol))
            if level == 2:
                processed_smiles.append(self._bfs_smarts_env(frag, radius=2, sign=sign))
            if level == 3:
                processed_smiles.append(self._bfs_smarts_env(frag, radius=1, sign=sign))
            if level == 4:
                processed_smiles.append(self._bfs_smarts_env(frag, radius=0, sign=sign))
            if level == 5:
                processed_smiles.append(self._count_only_placeholder(frag, sign=sign))
        collected_mol_smi='.'.join(processed_smiles)
        return Chem.MolFromSmiles(collected_mol_smi,sanitize=False)

    def _bfs_smarts_env(self, mol, radius, sign='*'):
        """
        Levels 2 (radius=2) and 3 (radius=0): local atom-environment SMARTS
        fingerprint around each * attachment point — follows mmpdb's own design
        (a rooted circular/Morgan-style environment per radius, written as
        canonical SMARTS since mmpdb 3.0) rather than the previous hand-rolled
        SMILES-based approximation.

        For each * attachment atom, collect all atoms within `radius` bonds of
        the atom it's bonded to (BFS, pure topological distance — no ring-
        completion needed: unlike SMILES, a SMARTS pattern doesn't require a
        ring to be fully present to be well-formed, since aromatic bonds use
        the unambiguous ':' bond symbol instead of requiring a complete
        alternating-bond Kekule structure). A prior version tried writing
        Kekulized SMILES instead to sidestep the same problem, which worked
        but reintroduced a different one: the SAME chemically-identical local
        ring environment can Kekulize to different bond-position variants
        depending on remote substituents elsewhere in the source molecule,
        splitting what should be one pooled group into several. SMARTS
        doesn't have this ambiguity (verified empirically against real
        datasets — see RECOMMENDATION_TABLE_PLAN.md).

        Each attachment's SMARTS is computed independently (radius=0 correctly
        covers EVERY attachment point on a multi-cut core, unlike the old
        level-3 implementation, which only ever inspected one arbitrarily-
        indexed atom); results are sorted and joined with '.' so the combined
        string is a canonical, order-independent grouping key.
        """
        envs = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 0:      # 0 = wildcard / attachment point
                continue
            attach_idx = atom.GetIdx()
            nbrs = [a.GetIdx() for a in atom.GetNeighbors()]
            if not nbrs:
                continue
            core_atom_idx = nbrs[0]           # the first real atom bonded to *

            # BFS from core_atom_idx; attach_idx pre-visited so we don't expand toward *
            visited       = {attach_idx}
            current_level = {core_atom_idx}
            visited.add(core_atom_idx)
            for _ in range(radius):
                next_level = set()
                for aidx in current_level:
                    for nbr in mol.GetAtomWithIdx(aidx).GetNeighbors():
                        nidx = nbr.GetIdx()
                        if nidx not in visited:
                            next_level.add(nidx)
                            visited.add(nidx)
                current_level = next_level
            atom_set = visited

            # All bonds whose BOTH endpoints are in atom_set
            bond_set = [b.GetIdx() for b in mol.GetBonds()
                        if b.GetBeginAtomIdx() in atom_set
                        and b.GetEndAtomIdx() in atom_set]

            try:
                smarts = Chem.MolFragmentToSmarts(mol, atomsToUse=sorted(atom_set), bondsToUse=bond_set)
                if smarts:
                    envs.append(smarts)
            except Exception:
                # Rare RDKit canonicalization edge case (a double bond whose
                # stereo-defining neighbor fell outside atom_set) — skip just
                # this one attachment's environment rather than the whole core.
                pass

        if not envs:
            return Chem.MolToSmiles(mol)    # fallback to original

        # Sorted join → deterministic canonical grouping key
        return '.'.join(sorted(envs))

    def _count_only_placeholder(self, mol, sign='*'):
        """
        Level 5: the coarsest rung — discards every last bit of chemical
        identity (element, aromaticity, bonding), keeping only HOW MANY *
        attachment points this one fragment has. Two fragments with the same
        attachment-point count always pool together here, regardless of what
        they actually looked like — that's the point of this level.

        Represented as a plain carbon chain of length n, one * substituent
        per carbon (e.g. n=3 -> '*CC(*)C*') rather than a single central atom
        with n branches, since a chain has no valence ceiling — it stays
        renderable (via the same strict Chem.MolFromSmiles every core_levelN
        column round-trips through, see process_core_structures) for any n,
        whereas a hub atom tops out at carbon's 4 bonds. The exact backbone
        shape carries no meaning beyond "n mounting points" — n is the only
        thing this level is meant to distinguish on.
        """
        n = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
        if n == 0:
            return Chem.MolToSmiles(mol)   # fallback to original, same as _bfs_smarts_env
        if n == 1:
            return '*C'
        return '*C' + 'C(*)' * (n - 2) + 'C*'

    def _get_ring_systems(self, mol, include_spiro=True):
        """获取分子的环系统"""
        ri = mol.GetRingInfo()
        systems = []
        
        for ring in ri.AtomRings():
            ring_atoms = set(ring)
            new_systems = []
            
            for system in systems:
                n_in_common = len(ring_atoms.intersection(system))
                if n_in_common and (include_spiro or n_in_common > 1):
                    ring_atoms = ring_atoms.union(system)
                else:
                    new_systems.append(system)
            
            new_systems.append(ring_atoms)
            systems = new_systems
        
        # 添加与环系统共轭的原子
        new_systems = []
        for system in systems:
            additional_atoms = []
            for atom in mol.GetAtoms():
                if atom.GetIdx() in system:
                    continue
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() in system:
                        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                        if bond.GetBondTypeAsDouble() > 1.0:  # 双键或三键
                            additional_atoms.append(atom.GetIdx())
            new_systems.append(list(system) + additional_atoms)
        
        return new_systems

    def _remove_substituents(self, mol, sign='*'):
        """移除指定标记的取代基"""
        # Kekulize on a copy first: once we delete an exocyclic substituent below,
        # any ring atom that lost it (e.g. a pyrrole-type ring N) needs its implicit
        # H recount redone. That recount only works reliably starting from explicit
        # single/double bonds — RWMol.RemoveAtom does not itself refresh the cached
        # aromatic valence of the atoms left behind, which produced un-kekulizable
        # SMILES (dangling ring heteroatoms) for a meaningful fraction of real cores.
        mol = Chem.Mol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        rw_mol = Chem.RWMol(mol)

        # 查找标记原子
        emp_atom_id = None
        for atom in mol.GetAtoms():
            if atom.GetSmarts() == f'[{sign}]' or atom.GetSmarts() == '*':
                emp_atom_id = atom.GetIdx()
                break
        
        if emp_atom_id is None:
            return rw_mol
        
        # 计算原子到标记原子的距离
        path_dict = {1: [emp_atom_id]}
        for atom in mol.GetAtoms():
            atom_id = atom.GetIdx()
            if atom_id != emp_atom_id:
                shortest_path = len(Chem.rdmolops.GetShortestPath(mol, emp_atom_id, atom_id))
                if shortest_path not in path_dict:
                    path_dict[shortest_path] = []
                path_dict[shortest_path].append(atom_id)
        
        # 查找连接环系统的原子
        neighbor_atom = None
        for distance in sorted(path_dict.keys()):
            for atom_id in path_dict[distance]:
                atom = mol.GetAtomWithIdx(atom_id)
                for neighbor in atom.GetNeighbors():
                    for ring in self._get_ring_systems(mol):
                        if neighbor.GetIdx() in ring:
                            neighbor_atom = neighbor
                            break
                    if neighbor_atom:
                        break
                if neighbor_atom:
                    break
            if neighbor_atom:
                break
        
        if neighbor_atom:
            # 确定要移除的原子
            path_length = len(Chem.rdmolops.GetShortestPath(mol, emp_atom_id, neighbor_atom.GetIdx()))
            atoms_to_keep = []
            for dist in range(1, path_length + 1):
                atoms_to_keep.extend(path_dict[dist])
            
            for ring in self._get_ring_systems(mol):
                if neighbor_atom.GetIdx() in ring:
                    atoms_to_keep.extend(ring)
                    break
            
            # 移除不在保留列表中的原子
            atoms_to_remove = []
            for atom in mol.GetAtoms():
                if atom.GetIdx() not in atoms_to_keep:
                    atoms_to_remove.append(atom.GetIdx())
            
            for atom_id in sorted(atoms_to_remove, reverse=True):
                rw_mol.RemoveAtom(atom_id)

            # Recompute implicit Hs/aromaticity now that some atoms lost neighbors.
            for atom in rw_mol.GetAtoms():
                atom.SetNoImplicit(False)
            rw_mol.UpdatePropertyCache(strict=False)
            try:
                Chem.SanitizeMol(rw_mol)
            except Exception:
                pass

        return rw_mol

    def _atoms_in_same_ring(self, mol, atom_idx1, atom_idx2):
        """检查两个原子是否在同一个环中"""
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if atom_idx1 in ring and atom_idx2 in ring:
                return True
        return False

    def save_molecules(self, molecule_list, folder, filename, format='sdf'):
        """保存分子到文件"""
        save_folder = os.path.join(self.working_directory, folder)
        os.makedirs(save_folder, exist_ok=True)
        
        file_path = os.path.join(save_folder, f'{filename}.{format}')
        
        if format.lower() == 'sdf':
            with Chem.SDWriter(file_path) as writer:
                for mol in molecule_list:
                    writer.write(mol)
        elif format.lower() == 'smi':
            with open(file_path, 'w') as f:
                for mol in molecule_list:
                    f.write(f"{Chem.MolToSmiles(mol)}\t{mol.GetProp('_Name')}\n")
        
        print(f"已保存 {len(molecule_list)} 个分子到: {file_path}")
        return file_path
