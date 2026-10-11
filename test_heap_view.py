"""Tests for the Max-Heap visualisation data. Run with `python test_heap_view.py` (no pytest needed)."""
import io
import random
import tempfile
from pathlib import Path

import detectors as D
import server
from dsa import MaxHeap

server.DB_PATH = Path(tempfile.mkdtemp()) / "test_history.sqlite3"   # never touch the real history DB


def parent_ge_children(keys):
    return all(not (keys[i] > keys[(i - 1) // 2]) for i in range(1, len(keys)))


def test_heap_log_keeps_old_fields_and_new_ones_replay_exactly():
    """The log's swap_pairs must reproduce the real heap array step by step (nothing invented)."""
    rnd = random.Random(7)
    h, shadow = MaxHeap(), []
    for step in range(300):
        if shadow and rnd.random() < 0.35:
            h.pop()
            entry = h.log[-1]
            assert entry["operation"] == "extract-max"
            last = shadow.pop()
            if shadow:
                shadow[0] = last
            for i, j in entry["swap_pairs"]:
                shadow[i], shadow[j] = shadow[j], shadow[i]
        else:
            key = (rnd.randint(0, 20), -rnd.randint(1, 50), -step)
            h.push(key, {"n": step})
            entry = h.log[-1]
            assert entry["operation"] == "insert"
            shadow.append((key, {"n": step}))
            for i, j in entry["swap_pairs"]:
                shadow[i], shadow[j] = shadow[j], shadow[i]
        assert shadow == h.snapshot(), f"log replay diverged at step {step}"
        assert parent_ge_children([k for k, _ in h.snapshot()])
        for old in ("step", "operation", "issue", "key", "swaps", "n_swaps", "size", "top", "array_head"):
            assert old in entry                      # existing contract preserved
        assert entry["n_swaps"] == len(entry["swap_pairs"])


def test_comparisons_are_real_comparisons():
    h = MaxHeap()
    for n, score in enumerate([10, 40, 30, 90, 20, 90]):
        h.push((score, -n, -n), n)
    for e in h.log:
        if e["operation"] != "insert":
            continue
        swaps = sum(c["greater"] for c in e["comparisons"])
        assert swaps == e["n_swaps"]                 # each sift-up swap came from a 'greater' comparison
        for c in e["comparisons"]:
            assert c["greater"] == (c["key"] > c["against_key"])
            assert c["against"] == (c["index"] - 1) // 2
    out = [h.pop()[0] for _ in range(len(h))]
    assert out == sorted(out, reverse=True)
    for e in h.log:
        if e["operation"] == "extract-max":
            for c in e["comparisons"]:
                assert c["greater"] == (c["key"] > c["against_key"])
                assert c["phase"] == "sift-down"


def test_snapshot_is_a_copy_not_sorted_and_not_the_internal_list():
    h = MaxHeap()
    for n, s in enumerate([90, 70, 80, 30, 60, 50]):     # the spec's example heap
        h.push((s, -n, -n), n)
    snap = h.snapshot()
    snap.clear()
    assert len(h) == 6                                   # mutating the copy cannot corrupt the heap
    assert [k[0] for k, _ in h.snapshot()][0] == 90


def sample_payload():
    server.WORKSPACES.clear()
    client = server.app.test_client()
    resp = client.get("/api/sample")
    assert resp.status_code == 200
    return client, resp.get_json()


def test_server_heap_view_matches_detector_scores_and_indices():
    _, d = sample_payload()
    hv, issues = d["dsa"]["heap_view"], d["issues"]
    arr = hv["array"]
    assert hv["size"] == d["issue_count"] == len(arr) and not hv["array_truncated"]
    keys = [tuple(n["key"]) for n in arr]
    assert parent_ge_children(keys)                                          # max-heap property
    assert arr[0]["score"] == max(i["score"] for i in issues)                # root = highest priority
    by_id = {i["id"]: i for i in issues}
    for idx, n in enumerate(arr):
        assert n["index"] == idx
        src = by_id[n["id"]]
        assert n["score"] == src["score"] == D.priority_score(src["severity"], src["type"])
        bd = n["breakdown"]
        assert bd["severity_weight"] + bd["type_weight"] == n["score"]        # explanation adds up
        assert (n["row"], n["column"], n["type"], n["severity"]) == (src["row"], src["column"], src["type"], src["severity"])
        assert n["parent"] == (None if idx == 0 else (idx - 1) // 2)
        assert n["children"] == [c for c in (2 * idx + 1, 2 * idx + 2) if c < len(arr)]
    # every issue appears exactly once
    assert sorted(n["id"] for n in arr) == sorted(by_id)


def test_server_extract_order_and_old_fields_still_present():
    _, d = sample_payload()
    dsa, hv = d["dsa"], d["dsa"]["heap_view"]
    for legacy in ("hash_steps", "hash_stats", "missing_steps", "heap_steps", "heap_array", "priority_order",
                   "hash_table", "heap", "rows_scanned", "duplicates_detected"):
        assert legacy in dsa
    assert all(e["operation"] == "insert" for e in dsa["heap_steps"])        # legacy meaning unchanged
    assert hv["inserts_total"] == hv["size"] == len(hv["insert_ops"])
    assert hv["extracts_total"] == len(hv["extract_ops"]) == len(dsa["priority_order"])
    keys = [tuple(o["key"]) for o in hv["extract_ops"]]
    assert keys == sorted(keys, reverse=True)                                # extract-max in descending order
    assert [o["issue"]["score"] for o in hv["extract_ops"]] == [x["score"] for x in dsa["priority_order"]]
    assert hv["extract_ops"][-1]["size"] == 0


def test_large_dataset_reports_truncation_instead_of_hiding_nodes():
    rows = ["OrderID,Age"] + [f"{i},-{i}" for i in range(1, 1501)]
    csv = ("\n".join(rows) + "\n").encode()
    server.WORKSPACES.clear()
    client = server.app.test_client()
    d = client.post("/api/upload", data={"file": (io.BytesIO(csv), "big.csv")},
                    content_type="multipart/form-data").get_json()
    hv = d["dsa"]["heap_view"]
    assert hv["size"] == d["issue_count"] > server.HEAP_VIEW_MAX_NODES
    assert hv["array_truncated"] and hv["array_returned"] == server.HEAP_VIEW_MAX_NODES
    assert parent_ge_children([tuple(n["key"]) for n in hv["array"]])        # returned prefix is a valid heap prefix


def test_history_and_cleaning_flow_unaffected():
    client, d = sample_payload()
    wid = d["workspace_id"]
    before_missing = [i for i in d["issues"] if i["type"] == "Missing value"]
    assert before_missing
    hist = client.get("/api/history").get_json()["history"]
    assert any(h["id"] == wid for h in hist)
    assert client.post("/api/history/pin", json={"id": wid, "pinned": True}).get_json()["history"][0]["pinned"]
    reopened = client.post("/api/history/open", json={"id": wid}).get_json()
    assert reopened["issue_count"] == d["issue_count"] and reopened["dsa"]["heap_view"]["size"] == d["issue_count"]
    prop = client.post("/api/clean/propose", json={}).get_json()
    assert prop["proposal"]["changes"] and prop["pending"]
    assert client.post("/api/cancel", json={}).get_json()["pending"] is None
    prop = client.post("/api/clean/propose", json={}).get_json()
    applied = client.post("/api/apply", json={}).get_json()
    assert "Applied" in applied["message"]
    after_missing = [i for i in applied["issues"] if i["type"] == "Missing value"]
    assert len(after_missing) < len(before_missing)                          # cleaning still works
    assert applied["dsa"]["heap_view"]["size"] == applied["issue_count"]     # heap rebuilt from the new issue list


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
