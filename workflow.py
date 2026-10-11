"""State and approval workflow for DataSleuth. Original upload bytes/data are immutable."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import math
import re
import statistics
from typing import Any

import pandas as pd

import detectors as D

LABELS = ["Unreviewed", "Accept as valid", "Confirmed issue", "Placeholder", "Needs investigation"]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_csv(raw: bytes) -> pd.DataFrame:
    if not raw:
        raise ValueError("The uploaded file is empty.")
    try:
        # Preserve cell text; detector rules, not pandas' NA inference, identify missing markers.
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, na_filter=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", dtype=str,
                             keep_default_na=False, na_filter=False)
        except Exception as exc:
            raise ValueError(f"Could not parse this CSV: {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV has no header or data rows.") from exc
    except Exception as exc:
        raise ValueError(f"Could not parse this CSV: {exc}") from exc
    if len(df.columns) == 0:
        raise ValueError("The CSV has no columns.")
    if any(str(c).strip() == "" for c in df.columns):
        raise ValueError("Every column must have a name. Please fix blank column headers and upload again.")
    if df.columns.duplicated().any():
        raise ValueError("Column names must be unique. Please rename duplicate headers and upload again.")
    df.index = pd.RangeIndex(1, len(df) + 1, name="row")
    return df


def show_val(v: Any) -> str:
    if D.is_missing(v):
        return "(missing)" if str(v).strip() else "(empty)"
    return str(v)


@dataclass
class Workspace:
    raw: bytes
    name: str
    original_df: pd.DataFrame
    working_df: pd.DataFrame = field(init=False)
    rules_init: pd.DataFrame = field(init=False)
    rules_df: pd.DataFrame = field(init=False)
    analysed: bool = False
    pending: dict | None = None
    audit: list[dict] = field(default_factory=list)
    review: dict = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    changes: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.original_df = self.original_df.copy(deep=True)
        self.working_df = self.original_df.copy(deep=True)
        self.rules_init = D.default_rules(self.original_df)
        self.rules_df = self.rules_init.copy(deep=True)

    def rules(self) -> dict:
        return D.resolve_rules(self.rules_df)

    def changes_applied(self) -> int:
        return sum(1 for x in self.audit if x.get("status") == "applied" and x.get("operation") not in ("reset",))

    def reset(self):
        self.working_df = self.original_df.copy(deep=True)
        self.pending = None
        self.review.clear()
        self.audit.append({"Timestamp": now_iso(), "Operation": "reset", "Issue ID": "", "Row": "",
                           "Column": "", "Original value": "", "Proposed value": "",
                           "Action": "Reset working copy", "Status": "applied", "Details": "Restored from original upload"})
        self.changes.append({"operation": "reset", "timestamp": now_iso()})


def actions_for(issue_type: str, numeric: bool = False) -> list[tuple[str, str]]:
    if issue_type == "Missing value":
        acts = [("leave", "Leave unchanged"), ("mark_missing", "Mark as missing (review label only)")]
        if numeric:
            acts += [("mean", "Manual Clean · Fill with mean"), ("median", "Manual Clean · Fill with median")]
        acts += [("mode", "Manual Clean · Fill with mode"), ("custom", "User Clean · Enter custom value"),
                 ("remove", "Remove row (requires confirmation)")]
        return acts
    if issue_type == "Duplicate record":
        return [("leave", "Keep both records"), ("reviewed", "Mark as reviewed"),
                ("remove", "Propose removing this row (requires confirmation)")]
    if issue_type in ("Placeholder-like category", "Inconsistent category spelling", "Rare category"):
        return [("accept", "Accept as valid"), ("placeholder", "Mark as confirmed placeholder"),
                ("blank", "Mark cell as missing · replace with blank"),
                ("custom", "User Clean · Replace with custom value"), ("leave", "Leave unresolved")]
    return [("leave", "Leave unchanged"), ("custom", "User Clean · Enter corrected value"),
            ("blank", "Mark cell as missing · replace with blank"),
            ("mark_missing", "Record a missing-value review label only"),
            ("investigate", "Flag for further investigation"),
            ("remove", "Remove row (requires confirmation)")]


def _proposed_stat(ws: Workspace, col: str, action: str):
    vals = [v for v in ws.working_df[col].tolist() if not D.is_missing(v)]
    if action in ("mean", "median"):
        nums = [D.parse_number(v) for v in vals]
        nums = [x for x in nums if x is not None]
        if not nums:
            return None
        n = statistics.fmean(nums) if action == "mean" else statistics.median(nums)
        return str(int(n)) if float(n).is_integer() else str(round(n, 6))
    if action == "mode":
        if not vals:
            return None
        return str(pd.Series(vals, dtype=str).mode().iloc[0])
    return None


def validate_value(ws: Workspace, col: str, value: str) -> list[str]:
    errors = []
    rule = ws.rules().get(col, D.DEFAULT_RULE)
    if D.is_missing(value):
        return errors
    exp = rule.get("expected", "text")
    if exp == "numeric":
        n = D.parse_number(value)
        if n is None:
            errors.append(f"'{value}' is not numeric for column '{col}'.")
        else:
            if rule.get("min") is not None and n < rule["min"]:
                errors.append(f"Value is below the configured minimum ({rule['min']}).")
            if rule.get("max") is not None and n > rule["max"]:
                errors.append(f"Value is above the configured maximum ({rule['max']}).")
    elif exp == "email" and not D.EMAIL_RE.match(value):
        errors.append("This does not match the configured email format.")
    elif exp == "date" and D.parse_date(value) is None:
        errors.append("This date is not in a recognized date format.")
    return errors


def build_proposal(ws: Workspace, issues: list[dict], action: str, custom: str = "") -> dict:
    changes, errors, warnings, rows, cols = [], [], [], [], []
    issue_ids = []
    kind = "edit"
    action_label = action.replace("_", " ").title()
    for issue in issues:
        row, col = int(issue["row"]), issue["column"]
        issue_ids.append(issue["id"])
        if row not in rows: rows.append(row)
        if col not in cols: cols.append(col)
        if action in ("leave", "accept", "reviewed", "investigate", "placeholder", "mark_missing"):
            kind = "meta"
            continue
        if action == "remove":
            kind = "remove"
            changes.append({"row": row, "column": "(entire row)", "original": " | ".join(map(str, ws.working_df.loc[row].tolist())) if row in ws.working_df.index else "(row not found)", "proposed": "(remove row)"})
            continue
        if row not in ws.working_df.index or col not in ws.working_df.columns:
            errors.append(f"Row {row}, column '{col}' no longer exists.")
            continue
        original = ws.working_df.at[row, col]
        if action in ("mean", "median", "mode"):
            proposed = _proposed_stat(ws, col, action)
            if proposed is None:
                errors.append(f"Cannot calculate {action} for '{col}' because it has no usable values.")
                continue
        elif action == "custom":
            proposed = str(custom)
        elif action == "blank":
            proposed = ""
        else:
            errors.append(f"Unsupported action: {action}")
            continue
        if action == "custom" and str(custom) == "":
            errors.append("Enter a value before previewing this change.")
        errors.extend(validate_value(ws, col, str(proposed)))
        changes.append({"row": row, "column": col, "original": original, "proposed": str(proposed)})
    return {"changes": changes, "errors": list(dict.fromkeys(errors)), "warnings": warnings,
            "rows": rows, "columns": cols, "issue_ids": issue_ids, "kind": kind,
            "action": action, "action_label": action_label, "created_at": now_iso()}


def build_edit_proposal(ws: Workspace, edited_df: pd.DataFrame) -> dict:
    changes, errors = [], []
    common_cols = [c for c in ws.working_df.columns if c in edited_df.columns]
    for row in ws.working_df.index:
        if row not in edited_df.index:
            continue
        for col in common_cols:
            old, new = ws.working_df.at[row, col], edited_df.at[row, col]
            old_s, new_s = str(old), str(new)
            if old_s != new_s:
                errors.extend(validate_value(ws, col, new_s))
                changes.append({"row": int(row), "column": col, "original": old, "proposed": new_s})
    return {"changes": changes, "errors": list(dict.fromkeys(errors)), "warnings": [],
            "rows": sorted({c["row"] for c in changes}), "columns": list(dict.fromkeys(c["column"] for c in changes)),
            "issue_ids": [], "kind": "edit", "action": "user_edit", "action_label": "User Clean · Manual cell edits",
            "created_at": now_iso()}



def build_group_replacement_proposal(ws: Workspace, column: str, group_key: str, replacement: str) -> dict:
    """Propose one canonical value for every casing/spacing variant in a category group."""
    if column not in ws.working_df.columns:
        return {"changes": [], "errors": [f"Column '{column}' no longer exists."], "warnings": [],
                "rows": [], "columns": [column], "issue_ids": [], "kind": "edit",
                "action": "group_replace", "action_label": "Manual Clean · Standardize category group", "created_at": now_iso()}
    replacement = str(replacement).strip()
    if not replacement:
        error = ["Enter a non-empty canonical category value."]
        changes = []
    else:
        error, changes = [], []
        for row, value in ws.working_df[column].items():
            if not D.is_missing(value) and D.normalize_key(value) == group_key and str(value) != replacement:
                changes.append({"row": int(row), "column": column, "original": value, "proposed": replacement,
                                "reason": f"Standardized every spelling/case variant in this group to '{replacement}'.",
                                "confidence": "User chosen"})
                error.extend(validate_value(ws, column, replacement))
    return {"changes": changes, "errors": list(dict.fromkeys(error)), "warnings": [],
            "rows": sorted({c["row"] for c in changes}), "columns": [column], "issue_ids": [],
            "kind": "edit", "action": "group_replace", "action_label": "Manual Clean · Standardize category group",
            "created_at": now_iso()}


def build_smart_clean_proposal(ws: Workspace, issues: list[dict]) -> dict:
    """Build an explainable, conservative cleaning plan; changes remain proposals.

    Numeric missing values use the median, categorical missing values use the mode,
    and free-text/email/date missing values are labeled "missing" rather than guessed.
    Category variants are unified to an observed canonical spelling when possible.
    Invalid values, outliers, and duplicate rows are never guessed or removed.
    """
    changes_by_cell: dict[tuple[int, str], dict] = {}
    warnings: list[str] = []
    rules = ws.rules()

    def propose(row, col, original, proposed, reason, confidence="High"):
        key = (int(row), str(col))
        # Do not let a later, lower-quality recommendation overwrite an earlier one.
        changes_by_cell[key] = {"row": int(row), "column": str(col), "original": original,
                                "proposed": str(proposed), "reason": reason, "confidence": confidence}

    # Handle each missing cell using the column's inferred/explicit type.
    for issue in issues:
        row, col = int(issue["row"]), issue["column"]
        if issue["type"] != "Missing value" or row not in ws.working_df.index or col not in ws.working_df.columns:
            continue
        original = ws.working_df.at[row, col]
        expected = rules.get(col, D.DEFAULT_RULE).get("expected", "text")
        valid_values = [v for v in ws.working_df[col].tolist() if not D.is_missing(v)]
        if expected == "numeric":
            rule = rules.get(col, D.DEFAULT_RULE)
            numbers = [D.parse_number(v) for v in valid_values]
            numbers = [v for v in numbers if v is not None
                       and (rule.get("min") is None or v >= rule["min"])
                       and (rule.get("max") is None or v <= rule["max"])]
            if numbers:
                mean = statistics.fmean(numbers)
                proposed = str(int(mean)) if float(mean).is_integer() else str(round(mean, 6))
                propose(row, col, original, proposed,
                        f"Filled with the mean of {len(numbers)} valid numeric value(s) in this column ({proposed}). Review whether mean imputation suits this dataset.", "Medium")
            else:
                propose(row, col, original, "missing", "No usable numeric values were available, so the cell is labeled 'missing' instead of inventing a number.", "High")
        elif expected == "categorical":
            category_values = [str(v).strip() for v in valid_values
                               if str(v).strip().lower() not in D.PLACEHOLDERS]
            if category_values:
                counts = Counter(category_values)
                proposed = counts.most_common(1)[0][0]
                propose(row, col, original, proposed,
                        f"Filled with the most frequent category (mode: '{proposed}'). Check that this default makes sense for your use case.", "Medium")
            else:
                propose(row, col, original, "missing", "No valid categories were available; labeled as 'missing' rather than guessing.", "High")
        else:
            propose(row, col, original, "missing",
                    f"This is a {expected} / free-text field. There is no safe value to infer, so it is labeled 'missing' instead of fabricated.", "High")

    # Canonicalize grouped category variants, preferring an existing title-case form
    # (e.g. Chennai) over whichever spelling happens to appear most often.
    grouped = {}
    for issue in issues:
        if issue["type"] == "Inconsistent category spelling" and issue.get("group_key"):
            grouped.setdefault((issue["column"], issue["group_key"]), []).append(issue)
    for (col, group_key), members in grouped.items():
        if col not in ws.working_df.columns:
            continue
        values = [str(v).strip() for v in ws.working_df[col].tolist()
                  if not D.is_missing(v) and D.normalize_key(v) == group_key]
        if not values:
            continue
        forms = Counter(values)
        title_candidates = [v for v in forms if v == v.title()]
        if title_candidates:
            canonical = sorted(title_candidates, key=lambda v: (-forms[v], len(v)))[0]
        else:
            # If variants are purely casing differences, create a readable title-case form.
            stripped = [re.sub(r"\s+", " ", v).strip() for v in values]
            if len({D.normalize_key(v) for v in stripped}) == 1 and all(re.fullmatch(r"[A-Za-z .'-]+", v) for v in stripped):
                canonical = max(stripped, key=len).title()
            else:
                canonical = forms.most_common(1)[0][0]
        for issue in members:
            row = int(issue["row"])
            if row not in ws.working_df.index:
                continue
            original = ws.working_df.at[row, col]
            if str(original).strip() != canonical:
                propose(row, col, original, canonical,
                        f"Unified a case/spacing variant to the canonical category '{canonical}'.", "High")

    unresolved = sorted({f"{i['type']} in {i['column']} (row {i['row']})" for i in issues
                         if i["type"] not in ("Missing value", "Inconsistent category spelling")})
    if unresolved:
        shown = unresolved[:8]
        warnings.append("Needs human review, not a guessed correction: " + "; ".join(shown)
                        + (f"; and {len(unresolved) - len(shown)} more." if len(unresolved) > len(shown) else ""))
    changes = list(changes_by_cell.values())
    return {"changes": changes, "errors": [], "warnings": warnings,
            "rows": sorted({c["row"] for c in changes}), "columns": list(dict.fromkeys(c["column"] for c in changes)),
            "issue_ids": [i["id"] for i in issues if (int(i["row"]), i["column"]) in changes_by_cell],
            "kind": "edit", "action": "smart_clean", "action_label": "DataSleuth Clean · explainable recommendations",
            "created_at": now_iso()}


def apply_pending(ws: Workspace, confirm_removal: bool = False) -> tuple[bool, str]:
    p = ws.pending
    if not p:
        return False, "There is no pending proposal."
    if p.get("errors"):
        return False, "Fix validation errors before applying this proposal."
    if p["kind"] == "remove" and not confirm_removal:
        return False, "Explicit row-removal confirmation is required."
    stamp = now_iso()
    if p["kind"] == "meta":
        for iid in p["issue_ids"]:
            label = {"accept": "Accept as valid", "reviewed": "Accept as valid", "placeholder": "Placeholder",
                     "investigate": "Needs investigation", "mark_missing": "Placeholder", "leave": "Needs investigation"}.get(p["action"], "Needs investigation")
            ws.review[iid] = {"label": label, "note": p["action_label"], "reviewed": True, "timestamp": stamp}
        ws.audit.append({"Timestamp": stamp, "Operation": "review", "Issue ID": ", ".join(p["issue_ids"]),
                         "Row": ", ".join(map(str, p["rows"])), "Column": ", ".join(p["columns"]),
                         "Original value": "", "Proposed value": "", "Action": p["action_label"],
                         "Status": "applied", "Details": "Review metadata only; cell values unchanged"})
    else:
        if p["kind"] == "remove":
            for row in p["rows"]:
                if row in ws.working_df.index:
                    original = " | ".join(map(str, ws.working_df.loc[row].tolist()))
                    ws.working_df = ws.working_df.drop(index=row)
                    ws.audit.append({"Timestamp": stamp, "Operation": "remove_row", "Issue ID": ", ".join(p["issue_ids"]),
                                     "Row": row, "Column": "(entire row)", "Original value": original,
                                     "Proposed value": "(removed)", "Action": p["action_label"], "Status": "applied",
                                     "Details": "User explicitly confirmed row removal"})
        else:
            for c in p["changes"]:
                row, col = c["row"], c["column"]
                if row not in ws.working_df.index or col not in ws.working_df.columns:
                    continue
                ws.working_df.at[row, col] = c["proposed"]
                ws.audit.append({"Timestamp": stamp, "Operation": "edit_cell", "Issue ID": ", ".join(p["issue_ids"]),
                                 "Row": row, "Column": col, "Original value": str(c["original"]),
                                 "Proposed value": str(c["proposed"]), "Action": p["action_label"],
                                 "Status": "applied", "Details": "Confirmed by user"})
        ws.changes.append({"operation": p["action"], "timestamp": stamp, "changes": p["changes"]})
    n = len(p["changes"])
    ws.pending = None
    return True, f"Applied the confirmed operation. {n} cell/row change(s) recorded."


def cancel_pending(ws: Workspace):
    p = ws.pending
    if p:
        ws.audit.append({"Timestamp": now_iso(), "Operation": "proposal", "Issue ID": ", ".join(p.get("issue_ids", [])),
                         "Row": ", ".join(map(str, p.get("rows", []))), "Column": ", ".join(p.get("columns", [])),
                         "Original value": "", "Proposed value": "", "Action": p.get("action_label", ""),
                         "Status": "cancelled", "Details": "Proposal cancelled; working dataset unchanged"})
    ws.pending = None


def set_label(ws: Workspace, issue: dict, label: str, note: str = ""):
    ws.review[issue["id"]] = {"label": label, "note": note, "reviewed": True, "timestamp": now_iso()}
    ws.audit.append({"Timestamp": now_iso(), "Operation": "review", "Issue ID": issue["id"], "Row": issue["row"],
                     "Column": issue["column"], "Original value": str(issue["value"]), "Proposed value": "",
                     "Action": label, "Status": "applied", "Details": note})


def issues_frame(ws: Workspace, issues: list[dict], pretty: bool = True) -> pd.DataFrame:
    rows = []
    for i in issues:
        rows.append({"Issue ID": i["id"], "Row": i["row"], "Column": i["column"],
                     "Value": show_val(i["value"]), "Issue type": i["type"], "Severity": i["severity"],
                     "Reason": i["reason"], "Suggested action": i.get("suggestion", ""),
                     "Your label": ws.review.get(i["id"], {}).get("label", "Unreviewed"),
                     "Note": ws.review.get(i["id"], {}).get("note", ""), "Priority score": i.get("score", 0)})
    return pd.DataFrame(rows)


def audit_frame(ws: Workspace) -> pd.DataFrame:
    cols = ["Timestamp", "Operation", "Issue ID", "Row", "Column", "Original value", "Proposed value", "Action", "Status", "Details"]
    return pd.DataFrame(ws.audit, columns=cols)


def changed_cells(ws: Workspace):
    out = set()
    for row in ws.working_df.index:
        if row not in ws.original_df.index:
            continue
        for col in ws.working_df.columns:
            if col in ws.original_df.columns and str(ws.working_df.at[row, col]) != str(ws.original_df.at[row, col]):
                out.add((row, col))
    removed = [r for r in ws.original_df.index if r not in ws.working_df.index]
    return out, removed


def working_csv(ws: Workspace, include_original_row: bool = False) -> bytes:
    df = ws.working_df.copy()
    if include_original_row:
        df.insert(0, "original_row", df.index)
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")
