"""
双/三切割 MMP 里"哪个连接点对应哪个不变片段"的追踪与规范化。

核心算法移植自 mmpdb（https://github.com/rdkit/mmpdb，
`mmpdblib/fragment_algorithm.py` 的 `get_symmetry_class` 和
`mmpdblib/index_algorithm.py` 的对称性枚举 + `cansmirks`），用来在两条
共享同一个"不变核心"的切割记录之间，找出让 R1、R2、Core 三方连接点正确
对齐的最小字典序原子编号映射，写出真正带 `[*:1]`/`[*:2]`（/`[*:3]`）的
SMIRKS——而不是像之前那样直接拼一个不考虑连接顺序、可能对不上号的占位符。

与 mmpdb 的差异：mmpdb 还有一张预计算的 `CANONICAL_ATTACHMENT_ORDER`
表，用来在持久化的 fragment 数据库里给同一个可变片段的多种等价切法去重；
我们这里每次都是现切现配对、不落盘成独立的 fragment 数据库，所以不需要
这张表——`get_smirks_order` 自己的暴力对称性枚举已经足以对任意一对切割
记录算出正确、双方一致的连接编号，不依赖片段是否已经被"规范化"过。
"""
import re

# ── 对称性判定（mmpdb fragment_algorithm.py::get_symmetry_class） ──────────

def get_symmetry_class(a, b, c=None):
    """
    判断 2 个（或 3 个）连接点/不变片段两两是否等价。
    2 个：返回 "11"（相同）或 "12"（不同）。
    3 个：返回 "111"/"112"/"121"/"122"/"123"，数字相同的位置表示等价。
    """
    if c is None:
        return "11" if a == b else "12"
    if a == b:
        return "111" if b == c else "112"
    if a == c:
        return "121"
    if b == c:
        return "122"
    return "123"


# ── 对称性枚举 + 最小字典序连接编号（mmpdb index_algorithm.py） ─────────────

def _sym11(p):
    return [(p[0], p[1]), (p[1], p[0])]


def _sym12(p):
    return [(p[0], p[1])]


def _sym111(p):
    return [
        (p[0], p[1], p[2]), (p[0], p[2], p[1]),
        (p[1], p[0], p[2]), (p[1], p[2], p[0]),
        (p[2], p[1], p[0]), (p[2], p[0], p[1]),
    ]


def _sym112(p):
    return [(p[0], p[1], p[2]), (p[1], p[0], p[2])]


def _sym121(p):
    return [(p[0], p[1], p[2]), (p[2], p[1], p[0])]


def _sym122(p):
    return [(p[0], p[1], p[2]), (p[0], p[2], p[1])]


def _sym123(p):
    return [(p[0], p[1], p[2])]


_SYMM_FUNCS = {
    "11": _sym11, "12": _sym12,
    "111": _sym111, "112": _sym112, "121": _sym121, "122": _sym122, "123": _sym123,
}


def _enumerate_symmetry(possibilities, symmetry_class):
    func = _SYMM_FUNCS[symmetry_class]
    result = []
    for p in possibilities:
        for q in func(p[-1]):
            result.append(p + (q,))
    return result


def _reorder(possibilities, order):
    result = []
    for p in possibilities:
        q = tuple(p[-1][offset] for offset in map(int, order))
        result.append(p + (q,))
    return result


def _invert_order(order):
    values = list(map(int, order))
    indices = list(range(len(values)))
    indices.sort(key=lambda i: values[i])
    return "".join(str(i) for i in indices)


_INVERT_TABLE = {s: _invert_order(s) for s in ("01", "10", "012", "021", "102", "120", "201", "210")}


def get_smirks_order(symmetry_class1, attachment_order1, constant_symmetry_class,
                      symmetry_class2, attachment_order2):
    """
    给定 R1（symmetry_class1/attachment_order1）、共享的不变核心
    （constant_symmetry_class）、R2（symmetry_class2/attachment_order2），
    在所有对称性等价的连接编号里选出字典序最小的一组，返回
    (r2_order, constant_order)——R1 本身固定用 [*:1],[*:2],(...) 的顺序，
    r2_order/constant_order 是 R2、Core 各自需要重新排列的连接点编号。
    """
    indices = tuple(range(len(symmetry_class1)))
    possibilities = [(indices,)]
    possibilities = _enumerate_symmetry(possibilities, symmetry_class1)
    possibilities = _reorder(possibilities, _INVERT_TABLE[attachment_order1])
    possibilities = _enumerate_symmetry(possibilities, constant_symmetry_class)
    possibilities = _reorder(possibilities, attachment_order2)
    possibilities = _enumerate_symmetry(possibilities, symmetry_class2)

    possibilities = [(p[5], p[3]) for p in possibilities]
    best = min(possibilities)
    return (
        "".join(str(i) for i in best[0]),
        "".join(str(i) for i in best[1]),
    )


# ── 按给定顺序给 SMILES 里的 * 打上 [*:N] 原子编号 ───────────────────────────

_WILDCARD_PAT = re.compile(re.escape("[*]") + "|" + re.escape("*"))


def relabel(smiles, order=None):
    """把 smiles 里第 k 个 * 换成 [*:order[k]+1]；order=None 时按原顺序编号。"""
    if order is None:
        order = list(range(smiles.count("*")))
    else:
        order = [int(c) for c in order]

    def _tag(_m):
        return "[*:%d]" % (order.pop(0) + 1)

    return _WILDCARD_PAT.sub(_tag, smiles)


class RelabelCache(dict):
    """缓存 relabel 结果，key 是 smiles 或 (smiles, order)。"""

    def __missing__(self, key):
        if isinstance(key, str):
            result = relabel(key)
        else:
            smiles, order = key
            result = relabel(smiles, order)
        self[key] = result
        return result


def build_smirks(frag1_smiles, symmetry_class1, attachment_order1,
                  constant_key, frag2_smiles, symmetry_class2, attachment_order2,
                  relabel_cache=None):
    """
    双/三切割专用：算出一条完整、连接点对齐一致的反应 SMIRKS
    ``[*:1]R1[*:2].[*:1]Core[*:2] >> [*:1]R2[*:2].[*:1]Core[*:2]``
    （3 切割则是 :1/:2/:3）。

    注意：R2 和 Core 各自需要的原子编号顺序（`new_order`/`constant_order`）
    不一定是原样的 0,1,(2)——两者必须配套使用才有意义，所以这里直接拼成
    一条完整字符串返回，不能只留下 "frag1>>frag2" 就把 Core 那部分的编号
    丢掉，否则 R2 里的 [*:N] 到底对应 Core 里哪个 [*:N] 就没法验证了。
    constant_symmetry_class 直接从 constant_key（'.'-连接的、已排序不变
    片段）反推——两条记录共享同一个 constant_key，对称性天然一致，不需要
    额外传参。
    """
    if relabel_cache is None:
        relabel_cache = RelabelCache()

    constant_terminals = constant_key.split('.')
    constant_symmetry_class = get_symmetry_class(*constant_terminals)

    new_order, constant_order = get_smirks_order(
        symmetry_class1, attachment_order1, constant_symmetry_class,
        symmetry_class2, attachment_order2,
    )

    labeled_frag1 = relabel_cache[frag1_smiles]
    labeled_frag2 = relabel_cache[frag2_smiles, new_order]
    labeled_constant = relabel_cache[constant_key, constant_order]

    return f"{labeled_frag1}.{labeled_constant}>>{labeled_frag2}.{labeled_constant}"


def get_orientation_tag(symmetry_class1, attachment_order1, constant_key,
                        symmetry_class2, attachment_order2):
    """
    双/三切割专用：算出这条记录"连接点朝向"的规范化 tag，用来在
    (core, R1, R2, cut_type) 分组时区分核心不对称时真正不同的连接朝向
    （否则会被静默合并成一条虚假规则），同时让对称等价的朝向收敛到同一个
    tag（不会被错误拆成两条）。

    直接复用 get_smirks_order 算出的 (new_order, constant_order)——它已经是
    "在所有对称性等价的连接编号里选出字典序最小的一组"，同一个真实变换
    不管从哪个分子对切出来都会得到相同的 tag，真正不同朝向的变换会得到
    不同的 tag。调用方把这个 tag 拼进 cut_type（例如 f'double#{tag}'），
    不需要新增列或改动任何下游 groupby 调用点。
    """
    constant_terminals = constant_key.split('.')
    constant_symmetry_class = get_symmetry_class(*constant_terminals)

    new_order, constant_order = get_smirks_order(
        symmetry_class1, attachment_order1, constant_symmetry_class,
        symmetry_class2, attachment_order2,
    )
    return f'{new_order}_{constant_order}'
