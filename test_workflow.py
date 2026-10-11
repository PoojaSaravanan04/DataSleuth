"""Regression checks for approval workflow and immutable original data."""
from pathlib import Path

import pandas as pd

import detectors as D
import workflow as W


def make_ws():
    raw = Path("sample_data.csv").read_bytes()
    return W.Workspace(raw, "sample_data.csv", W.parse_csv(raw))


def test_proposal_cancel_does_not_change_working_or_original():
    ws = make_ws()
    before_working = ws.working_df.copy(deep=True)
    before_original = ws.original_df.copy(deep=True)
    issue = next(i for i in D.scan_dataset(ws.working_df, ws.rules_df)["issues"]
                 if i["type"] == "Missing value" and i["column"] == "Age")
    ws.pending = W.build_proposal(ws, [issue], "custom", "21")
    W.cancel_pending(ws)
    assert ws.working_df.equals(before_working)
    assert ws.original_df.equals(before_original)


def test_confirmed_edit_changes_working_only_and_audits():
    ws = make_ws()
    before_original = ws.original_df.copy(deep=True)
    issue = next(i for i in D.scan_dataset(ws.working_df, ws.rules_df)["issues"]
                 if i["type"] == "Missing value" and i["column"] == "Age")
    ws.pending = W.build_proposal(ws, [issue], "custom", "21")
    ok, _ = W.apply_pending(ws)
    assert ok
    assert ws.working_df.at[issue["row"], "Age"] == "21"
    assert ws.original_df.equals(before_original)
    assert not W.audit_frame(ws).empty


def test_row_removal_requires_explicit_confirmation():
    ws = make_ws()
    issue = next(i for i in D.scan_dataset(ws.working_df, ws.rules_df)["issues"]
                 if i["type"] == "Out-of-range value" and i["column"] == "Age")
    row_count = len(ws.working_df)
    ws.pending = W.build_proposal(ws, [issue], "remove")
    ok, _ = W.apply_pending(ws, confirm_removal=False)
    assert not ok and len(ws.working_df) == row_count
    ok, _ = W.apply_pending(ws, confirm_removal=True)
    assert ok and len(ws.working_df) == row_count - 1
    assert len(ws.original_df) == row_count


def test_working_csv_does_not_include_index_by_default():
    ws = make_ws()
    content = W.working_csv(ws).decode("utf-8")
    assert content.splitlines()[0].startswith("OrderID,")
    assert not content.splitlines()[0].startswith("row,")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
