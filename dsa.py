"""DataSleuth - hand-written data structures (standard library only).

* DuplicateHashTable / find_duplicates : chained hash table used for exact
  duplicate-record detection, with a full execution trace.
* MaxHeap                              : binary max-heap used to prioritise issues,
  with a log of every insertion / extraction.

Complexity (n rows, L = characters per record):
  hashing one record ........ O(L)
  insert / lookup (average) . O(L + chain length) ~ O(L)   (chains stay short)
  whole duplicate scan ...... O(n * L) expected
  worst case ................ O(n^2 * L): every record lands in one bucket, so each
                              new record is compared with every stored record.
  heap push / pop ........... O(log n);  building from n items O(n log n).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional, Sequence

_MASK32 = 0xFFFFFFFF
FIELD_SEP = "\x1f"  # unit separator: joins the cells of a record into one string


def djb2_hash(text: str) -> int:
    """Classic djb2 string hash (h = h*33 + c), truncated to 32 bits."""
    h = 5381
    for ch in text:
        h = (((h << 5) + h) + ord(ch)) & _MASK32
    return h


def constant_hash(text: str) -> int:
    """Deliberately terrible hash: every record collides (worst-case demo)."""
    return 42


def record_key(record: Sequence[str]) -> str:
    return FIELD_SEP.join(record)


class _Entry(NamedTuple):
    full_hash: int
    record: tuple
    row_id: int


class DuplicateHashTable:
    """Separate-chaining hash table keyed by whole records.

    Each bucket holds a list of (full_hash, record, first_row_id).  A new record is
    declared a duplicate ONLY when the full record tuple is equal to a stored one -
    sharing a bucket or even a hash value is never enough, so collisions cannot
    create false duplicates.
    """

    def __init__(self, num_buckets: int = 64, hash_fn: Callable[[str], int] = djb2_hash):
        if num_buckets < 1:
            raise ValueError("num_buckets must be >= 1")
        self.num_buckets = num_buckets
        self.hash_fn = hash_fn
        self.buckets: list[list[_Entry]] = [[] for _ in range(num_buckets)]
        self.size = 0
        self.comparisons = 0

    def chain_lengths(self) -> list[int]:
        return [len(b) for b in self.buckets]

    def process(self, row_id: int, record: tuple) -> dict:
        """Examine one record; insert it if new. Returns a trace dict for this step."""
        key = record_key(record)
        full_hash = self.hash_fn(key)
        bucket = full_hash % self.num_buckets
        chain = self.buckets[bucket]
        chain_before = [e.row_id for e in chain]

        examined = []
        duplicate_of: Optional[int] = None
        true_collision = False
        for entry in chain:
            self.comparisons += 1
            same_hash = entry.full_hash == full_hash
            same_record = same_hash and entry.record == record
            if not same_hash:
                verdict = "Different full hash (only shares the bucket) - skipped"
            elif same_record:
                verdict = "Identical record - DUPLICATE CONFIRMED"
            else:
                verdict = "Same hash value but different record - hash collision, NOT a duplicate"
                true_collision = True
            examined.append({
                "row": entry.row_id,
                "same_hash": same_hash,
                "same_record": same_record,
                "verdict": verdict,
                "record": " | ".join(entry.record),
            })
            if same_record:
                duplicate_of = entry.row_id
                break

        inserted = duplicate_of is None
        if inserted:
            chain.append(_Entry(full_hash, record, row_id))
            self.size += 1
            if chain_before:
                outcome = (f"New record - bucket collision ({len(chain_before)} other record(s) "
                           f"in bucket, none identical)")
            else:
                outcome = "New record - bucket was empty"
        else:
            outcome = f"DUPLICATE of row {duplicate_of}"

        return {
            "row": row_id,
            "record": " | ".join(record),
            "hash": full_hash,
            "bucket": bucket,
            "chain_before": chain_before,
            "examined": examined,
            "inserted": inserted,
            "duplicate_of": duplicate_of,
            "bucket_collision": inserted and bool(chain_before),
            "hash_collision": true_collision,
            "outcome": outcome,
            "chain_after": [e.row_id for e in chain],
        }


@dataclass
class DuplicateResult:
    table: DuplicateHashTable
    steps: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)   # first_row -> [later duplicate rows]
    stats: dict = field(default_factory=dict)


def find_duplicates(rows: Sequence[tuple], num_buckets: int = 64,
                    hash_fn: Callable[[str], int] = djb2_hash,
                    max_trace: int = 50000) -> DuplicateResult:
    """rows: sequence of (row_id, record_tuple_of_str). Pandas is NOT used here."""
    table = DuplicateHashTable(num_buckets, hash_fn)
    steps: list = []
    groups: dict = {}
    bucket_collisions = hash_collisions = 0
    for i, (row_id, record) in enumerate(rows, start=1):
        step = table.process(row_id, tuple(record))
        step["step"] = i
        if step["duplicate_of"] is not None:
            groups.setdefault(step["duplicate_of"], []).append(row_id)
        bucket_collisions += step["bucket_collision"]
        hash_collisions += step["hash_collision"]
        if i <= max_trace:
            steps.append(step)
    lengths = table.chain_lengths()
    stats = {
        "rows_examined": len(rows),
        "unique_records": table.size,
        "duplicate_rows": sum(len(v) for v in groups.values()),
        "duplicate_groups": len(groups),
        "comparisons": table.comparisons,
        "bucket_collisions": bucket_collisions,
        "hash_collisions": hash_collisions,
        "buckets": num_buckets,
        "used_buckets": sum(1 for x in lengths if x),
        "max_chain": max(lengths) if lengths else 0,
        "load_factor": round(table.size / num_buckets, 3),
        "trace_truncated": len(rows) > max_trace,
    }
    return DuplicateResult(table, steps, groups, stats)


class MaxHeap:
    """Binary max-heap of (key, payload); compares keys only. Logs every operation.

    Each log entry keeps its original fields and additionally records, for the
    visualisation, the exact key comparisons that were made ("comparisons") and the
    swaps as index pairs ("swap_pairs"). Nothing is recorded that did not happen.
    """

    def __init__(self):
        self._a: list = []
        self.log: list = []

    def __len__(self):
        return len(self._a)

    def peek(self):
        if not self._a:
            raise IndexError("peek from empty heap")
        return self._a[0]

    def snapshot(self) -> list:
        """Copy of the internal array in heap order (index i -> (key, payload)).

        It is NOT sorted: children of node i live at 2i+1 and 2i+2.
        """
        return list(self._a)

    def _record(self, op, payload, key, swaps, comparisons=None, extra=None):
        entry = {
            "step": len(self.log) + 1,
            "operation": op,
            "issue": payload,
            "key": key,
            "swaps": ", ".join(f"[{i}]<->[{j}]" for i, j in swaps) or "none",
            "n_swaps": len(swaps),
            "size": len(self._a),
            "top": self._a[0][1] if self._a else "",
            "array_head": " , ".join(str(p) for _, p in self._a[:10]),
            # --- additive fields (existing consumers ignore them) ---
            "swap_pairs": [[i, j] for i, j in swaps],
            "comparisons": comparisons or [],
        }
        if extra:
            entry.update(extra)
        self.log.append(entry)

    def push(self, key, payload):
        a = self._a
        a.append((key, payload))
        i = len(a) - 1
        swaps = []
        comparisons = []
        while i > 0:                       # sift-up
            parent = (i - 1) // 2
            greater = a[i][0] > a[parent][0]
            comparisons.append({"phase": "sift-up", "index": i, "against": parent,
                                "key": a[i][0], "against_key": a[parent][0], "greater": greater})
            if greater:
                a[i], a[parent] = a[parent], a[i]
                swaps.append((i, parent))
                i = parent
            else:
                break
        self._record("insert", payload, key, swaps, comparisons, {"final_index": i})

    def pop(self):
        a = self._a
        if not a:
            raise IndexError("pop from empty heap")
        top = a[0]
        last = a.pop()
        swaps = []
        comparisons = []
        extra = {"moved_to_root": None}
        if a:
            a[0] = last
            extra["moved_to_root"] = {"from_index": len(a), "key": last[0], "issue": last[1]}
            i, n = 0, len(a)
            while True:                    # sift-down
                left, right, largest = 2 * i + 1, 2 * i + 2, i
                if left < n:
                    greater = a[left][0] > a[largest][0]
                    comparisons.append({"phase": "sift-down", "index": left, "against": largest,
                                        "key": a[left][0], "against_key": a[largest][0], "greater": greater})
                    if greater:
                        largest = left
                if right < n:
                    greater = a[right][0] > a[largest][0]
                    comparisons.append({"phase": "sift-down", "index": right, "against": largest,
                                        "key": a[right][0], "against_key": a[largest][0], "greater": greater})
                    if greater:
                        largest = right
                if largest == i:
                    break
                a[i], a[largest] = a[largest], a[i]
                swaps.append((i, largest))
                i = largest
            extra["final_index"] = i
        self._record("extract-max", top[1], top[0], swaps, comparisons, extra)
        return top
