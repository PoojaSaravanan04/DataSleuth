"""Plain-assert tests: run with `python test_dsa.py` (no pytest needed)."""
import random
from pathlib import Path

import pandas as pd

import detectors as D
from dsa import MaxHeap, constant_hash, djb2_hash, find_duplicates


def naive_groups(rows):
    seen, groups = {}, {}
    for rid, rec in rows:
        if rec in seen:
            groups.setdefault(seen[rec], []).append(rid)
        else:
            seen[rec] = rid
    return groups


def test_hash_table_matches_naive():
    rnd = random.Random(1)
    rows = [(i, tuple(rnd.choice("abc") for _ in range(2))) for i in range(1, 300)]
    for nb in (1, 3, 16, 257):
        for fn in (djb2_hash, constant_hash):
            assert find_duplicates(rows, nb, fn).groups == naive_groups(rows)


def test_true_hash_collision_is_not_a_duplicate():
    same_len = lambda s: len(s)          # "ab" and "cd" get the SAME full hash
    rows = [(1, ("ab",)), (2, ("cd",)), (3, ("ab",))]
    res = find_duplicates(rows, 8, same_len)
    assert res.groups == {1: [3]}
    assert res.stats["hash_collisions"] >= 1
    assert res.steps[1]["duplicate_of"] is None and res.steps[1]["hash_collision"]


def test_heap_order():
    h = MaxHeap()
    vals = [random.Random(2).randint(0, 99) for _ in range(200)]
    for i, v in enumerate(vals):
        h.push((v, -i), i)
    out = [h.pop()[0] for _ in range(len(vals))]
    assert out == sorted(out, reverse=True)
    assert len(h.log) == 400


def test_sample_scan_and_no_mutation():
    raw = Path("sample_data.csv").read_bytes()
    import io
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, na_filter=False)
    df.index = pd.RangeIndex(1, len(df) + 1, name="row")
    before = df.copy()
    rules = D.default_rules(df)
    res = D.scan_dataset(df, rules)
    assert df.equals(before), "scan must not modify the dataset"
    kinds = {(i["type"], i["row"], i["column"]) for i in res["issues"]}
    assert ("Missing value", 6, "Age") in kinds
    assert ("Out-of-range value", 8, "Age") in kinds            # -5
    assert ("Out-of-range value", 9, "Quantity") in kinds       # -40
    assert ("Non-numeric value in numeric column", 23, "Age") in kinds
    assert ("Malformed email", 7, "Email") in kinds
    assert ("Placeholder-like category", 4, "Supplier") in kinds
    assert ("Inconsistent category spelling", 3, "City") in kinds
    assert ("Unexpected date format", 14, "OrderDate") in kinds
    assert ("Statistical outlier", 19, "Quantity") in kinds
    assert res["dup_groups"] == {5: [16, 30], 2: [25]}
    # Unspecified / N/A / Unknown are reported, never normalised
    assert df.at[4, "Supplier"] == "Unspecified"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
