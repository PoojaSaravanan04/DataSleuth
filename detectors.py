"""DataSleuth - explainable detection rules. Read-only: never modifies the DataFrame."""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from datetime import datetime

import pandas as pd

from dsa import find_duplicates

ENTIRE_ROW = "(entire row)"
MISSING_MARKERS = {"", "null", "nan", "none", "nil", "na", "n/a", "#n/a", "(null)", "missing"}
PLACEHOLDERS = {"unknown", "unspecified", "n/a", "na", "tbd", "undefined",
                "not available", "not specified"}
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")
EXPECTED_TYPES = ["auto", "numeric", "categorical", "text", "email", "date"]

SEV_CONFIRMED = "Confirmed rule violation"
SEV_POTENTIAL = "Potential anomaly"
SEV_FACT = "Factual finding"
SEVERITY_WEIGHT = {SEV_CONFIRMED: 50, SEV_FACT: 30, SEV_POTENTIAL: 20}
TYPE_WEIGHT = {
    "Out-of-range value": 30, "Invalid date": 30, "Malformed email": 25,
    "Non-numeric value in numeric column": 25, "Missing value": 20,
    "Duplicate record": 15, "Unexpected date format": 15,
    "Placeholder-like category": 10, "Inconsistent category spelling": 10,
    "Rare category": 5, "Statistical outlier": 5,
}
DEFAULT_RULE = {"expected": "text", "explicit": False, "min": None, "max": None, "inferred": "text"}


def priority_score(severity: str, issue_type: str) -> int:
    """Documented priority rule: severity weight + issue-type weight."""
    return SEVERITY_WEIGHT.get(severity, 0) + TYPE_WEIGHT.get(issue_type, 0)


# ---------------------------------------------------------------- parsing helpers
def is_missing(v) -> bool:
    return v is None or str(v).strip().lower() in MISSING_MARKERS


def parse_number(v):
    s = str(v).strip()
    if not s or "_" in s:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def fmt_num(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


DATE_FORMATS = {
    "9999-99-99": ["%Y-%m-%d"], "9999/99/99": ["%Y/%m/%d"],
    "99/99/9999": ["%d/%m/%Y", "%m/%d/%Y"], "99-99-9999": ["%d-%m-%Y", "%m-%d-%Y"],
    "99.99.9999": ["%d.%m.%Y"],
}


def date_shape(v: str) -> str:
    return re.sub(r"\d", "9", str(v).strip())


def parse_date(v: str):
    for fmt in DATE_FORMATS.get(date_shape(v), []):
        try:
            return datetime.strptime(str(v).strip(), fmt)
        except ValueError:
            pass
    return None


def dominant_date_shape(values) -> str:
    shapes = Counter(date_shape(v) for v in values if not is_missing(v))
    return shapes.most_common(1)[0][0] if shapes else ""


def normalize_key(v: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def email_problem(v: str) -> str:
    if v != v.strip() or " " in v:
        return "contains spaces"
    if v.count("@") == 0:
        return "has no '@'"
    if v.count("@") > 1:
        return "has more than one '@'"
    local, domain = v.split("@")
    if not local:
        return "has nothing before the '@'"
    if not domain or "." not in domain:
        return "the domain part has no '.' (e.g. 'example.com')"
    return "does not match the pattern name@domain.tld"


def missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda s: s.str.strip().str.lower().isin(MISSING_MARKERS))


# ---------------------------------------------------------------- type inference / rules
def infer_column_type(name: str, values) -> str:
    vals = [v for v in values if not is_missing(v)]
    if not vals:
        return "text"
    lname = name.lower()
    if "email" in lname or "e-mail" in lname:
        return "email"
    n = len(vals)
    if sum(parse_number(v) is not None for v in vals) / n >= 0.8:
        return "numeric"
    date_ratio = sum(parse_date(v) is not None for v in vals) / n
    if date_ratio >= 0.6 or ("date" in lname and date_ratio >= 0.3):
        return "date"
    uniq = len(set(vals))
    if uniq <= 12 or (uniq / n <= 0.3 and uniq <= 50):
        return "categorical"
    return "text"


def default_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Editable example rules. They only apply to numeric columns and can be changed."""
    rows = []
    for col in df.columns:
        tokens = re.findall(r"[a-z]+", str(col).lower())
        mn = mx = float("nan")
        if "age" in tokens:
            mn, mx = 0.0, 120.0
        elif any(t in tokens for t in ("quantity", "qty", "units")):
            mn = 0.0
        rows.append({"Column": col, "Inferred type": infer_column_type(str(col), df[col].tolist()),
                     "Expected type": "auto", "Min": mn, "Max": mx})
    return pd.DataFrame(rows)


def _num(x):
    if x is None or pd.isna(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def resolve_rules(rules_df: pd.DataFrame) -> dict:
    out = {}
    for _, r in rules_df.iterrows():
        exp = str(r.get("Expected type", "auto") or "auto")
        inf = str(r.get("Inferred type", "text"))
        out[r["Column"]] = {"expected": inf if exp == "auto" else exp, "explicit": exp != "auto",
                            "min": _num(r.get("Min")), "max": _num(r.get("Max")), "inferred": inf}
    return out


# ---------------------------------------------------------------- profiling
def profile_columns(df: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
    rules = resolve_rules(rules_df)
    rows = []
    for col in df.columns:
        vals = df[col].tolist()
        nonmiss = [v for v in vals if not is_missing(v)]
        c = Counter(nonmiss)
        rows.append({"Column": col, "Inferred / expected type": rules.get(col, DEFAULT_RULE)["expected"],
                     "Non-missing": len(nonmiss), "Missing": len(vals) - len(nonmiss),
                     "Unique values": len(c),
                     "Most common": ", ".join(f"{k} ({n})" for k, n in c.most_common(3))})
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
    rules = resolve_rules(rules_df)
    rows = []
    for col in df.columns:
        if rules.get(col, DEFAULT_RULE)["expected"] != "numeric":
            continue
        raw = [v for v in df[col] if not is_missing(v)]
        nums = [x for x in (parse_number(v) for v in raw) if x is not None]
        if not nums:
            continue
        rows.append({"Column": col, "Numeric values": len(nums), "Non-numeric text": len(raw) - len(nums),
                     "Mean": round(statistics.fmean(nums), 4), "Median": round(statistics.median(nums), 4),
                     "Min": min(nums), "Max": max(nums),
                     "Std dev": round(statistics.stdev(nums), 4) if len(nums) > 1 else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scan
def make_issue(code, row, col, value, itype, severity, reason, suggestion, extra=None):
    d = {"id": f"{code}-{int(row)}-{col}", "code": code, "row": int(row), "column": col, "value": value,
         "type": itype, "severity": severity, "reason": reason, "suggestion": suggestion,
         "score": priority_score(severity, itype)}
    if extra:
        d.update(extra)
    return d


def scan_dataset(df: pd.DataFrame, rules_df: pd.DataFrame, num_buckets: int = 101,
                 flag_rare: bool = False) -> dict:
    """Read-only scan. Row numbers are df.index labels (1-based data rows)."""
    rules = resolve_rules(rules_df)
    rows = [int(r) for r in df.index]
    issues: list = []

    for col in df.columns:
        rule = rules.get(col, DEFAULT_RULE)
        exp = rule["expected"]
        values = df[col].tolist()
        nonmissing = []
        for r, v in zip(rows, values):
            if is_missing(v):
                what = f"holds the missing-value marker '{v}'" if v.strip() else "is empty"
                issues.append(make_issue(
                    "MISS", r, col, v, "Missing value", SEV_FACT,
                    f"The cell {what}. It is only reported; nothing has been filled in.",
                    "Decide whether to leave it, supply a value you choose, or mark it as intentionally missing."))
            else:
                nonmissing.append((r, v))
        if not nonmissing:
            continue

        flagged_rows: set = set()

        if exp == "numeric":
            parsed = [(r, v, parse_number(v)) for r, v in nonmissing]
            pct = round(100 * sum(x is not None for _, _, x in parsed) / len(parsed))
            valid = []
            for r, v, x in parsed:
                if x is None:
                    sev = SEV_CONFIRMED if rule["explicit"] else SEV_POTENTIAL
                    how = "configured as numeric" if rule["explicit"] else f"inferred to be numeric ({pct}% of its values parse as numbers)"
                    issues.append(make_issue(
                        "TYPE", r, col, v, "Non-numeric value in numeric column", sev,
                        f"'{v}' cannot be read as a number, but column '{col}' is {how}.",
                        "Check the source. Correct it, mark it missing, or leave it unchanged."))
                    flagged_rows.add(r)
                elif rule["min"] is not None and x < rule["min"]:
                    issues.append(make_issue(
                        "RANGE", r, col, v, "Out-of-range value", SEV_CONFIRMED,
                        f"{v} is below the configured minimum of {fmt_num(rule['min'])} for '{col}'. "
                        "This violates the rule as configured; it does not tell us what the correct value is.",
                        "Check the source. Enter a corrected value, mark it missing, or leave it unchanged."))
                    flagged_rows.add(r)
                elif rule["max"] is not None and x > rule["max"]:
                    issues.append(make_issue(
                        "RANGE", r, col, v, "Out-of-range value", SEV_CONFIRMED,
                        f"{v} is above the configured maximum of {fmt_num(rule['max'])} for '{col}'. "
                        "This violates the rule as configured; it does not tell us what the correct value is.",
                        "Check the source. Enter a corrected value, mark it missing, or leave it unchanged."))
                    flagged_rows.add(r)
                else:
                    valid.append((r, v, x))
            if len(valid) >= 8:                      # potential anomalies: extreme IQR outliers
                nums = [x for _, _, x in valid]
                q1, _, q3 = statistics.quantiles(nums, n=4, method="inclusive")
                iqr = q3 - q1
                if iqr > 0:
                    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
                    for r, v, x in valid:
                        if x < lo or x > hi:
                            issues.append(make_issue(
                                "OUT", r, col, v, "Statistical outlier", SEV_POTENTIAL,
                                f"{v} lies outside the extreme fences [{fmt_num(lo)}, {fmt_num(hi)}] "
                                f"(Q1={fmt_num(q1)}, Q3={fmt_num(q3)}, 3 x IQR). Unusual is not the same as wrong.",
                                "Verify against the source; accept it if it is genuine."))

        elif exp == "email":
            for r, v in nonmissing:
                if not EMAIL_RE.match(v):
                    issues.append(make_issue(
                        "MAIL", r, col, v, "Malformed email", SEV_CONFIRMED,
                        f"'{v}' does not look like a valid email address: {email_problem(v)}.",
                        "Enter a corrected address, mark it missing, or leave it unchanged."))

        elif exp == "date":
            dom = dominant_date_shape([v for _, v in nonmissing])
            for r, v in nonmissing:
                shape = date_shape(v)
                if parse_date(v) is None:
                    if shape in DATE_FORMATS:
                        issues.append(make_issue(
                            "DATE", r, col, v, "Invalid date", SEV_CONFIRMED,
                            f"'{v}' has a date layout but is not a real calendar date (e.g. month 13 or day 32).",
                            "Check the source; enter a corrected date or leave it unchanged."))
                    else:
                        issues.append(make_issue(
                            "FMT", r, col, v, "Unexpected date format", SEV_POTENTIAL,
                            f"'{v}' is not in any recognised date layout (most values use '{dom}', where 9 = a digit).",
                            "Check the source; enter the date in the column's usual format or leave it."))
                elif shape != dom:
                    amb = " Layouts like dd/mm/yyyy and mm/dd/yyyy are ambiguous, so it was not converted." if len(DATE_FORMATS[shape]) > 1 else ""
                    issues.append(make_issue(
                        "FMT", r, col, v, "Unexpected date format", SEV_POTENTIAL,
                        f"'{v}' uses layout '{shape}' while most values in '{col}' use '{dom}'.{amb}",
                        "Review whether the mixed formats are intended; nothing was converted."))

        else:  # categorical / text
            for r, v in nonmissing:
                if v.strip().lower() in PLACEHOLDERS:
                    issues.append(make_issue(
                        "PLACE", r, col, v, "Placeholder-like category", SEV_POTENTIAL,
                        f"The value '{v}' may represent a missing or unknown entry, or it may be a "
                        "legitimate category. Its meaning cannot be determined from the value alone.",
                        "Accept it as valid, mark it as a placeholder under your own rule, or replace it - your call."))
                    flagged_rows.add(r)
            if exp == "categorical":
                groups: dict = {}
                for r, v in nonmissing:
                    if v.strip().lower() in PLACEHOLDERS:
                        continue
                    key = normalize_key(v)
                    if key:
                        groups.setdefault(key, []).append((r, v))
                for key, members in groups.items():
                    forms = Counter(v for _, v in members)
                    if len(forms) < 2:
                        continue
                    dominant = forms.most_common(1)[0][0]
                    summary = ", ".join(f"'{f}' x{n}" for f, n in forms.most_common())
                    for r, v in members:
                        # Highlight every member of the variant group, including the most common
                        # spelling, so the user can compare the complete set in the editor.
                        issues.append(make_issue(
                            "CAT", r, col, v, "Inconsistent category spelling", SEV_POTENTIAL,
                            f"'{v}' belongs to a set of values that differ only in case, spacing or punctuation "
                            f"({summary}). They may represent the same category - or may not.",
                            "Compare the variants; keep, or choose a replacement yourself.",
                            extra={"group_key": key, "dominant_value": dominant}))
                        flagged_rows.add(r)
                if flag_rare and len(nonmissing) >= 10:
                    freq = Counter(v for _, v in nonmissing)
                    for r, v in nonmissing:
                        if freq[v] == 1 and r not in flagged_rows:
                            issues.append(make_issue(
                                "RARE", r, col, v, "Rare category", SEV_POTENTIAL,
                                f"'{v}' appears only once among {len(nonmissing)} values in '{col}'. "
                                "It may be a typo or a legitimate rare category.",
                                "Verify against the source; accept it if it is genuine."))

    # ---- exact duplicates through the custom hash table (no pandas duplicated())
    records = [(r, tuple(rec)) for r, rec in zip(rows, df.values.tolist())]
    res = find_duplicates(records, num_buckets)
    for first, dups in res.groups.items():
        members = [first] + dups
        by_row = dict(records)
        text = " | ".join(by_row[first])
        for m in members:
            others = [x for x in members if x != m]
            issues.append(make_issue(
                "DUP", m, ENTIRE_ROW, text, "Duplicate record", SEV_POTENTIAL,
                f"Row {m} is an exact copy of row(s) {', '.join(map(str, others))} (every column matches). "
                "A repeat may be a data-entry error or a genuine repeated event.",
                "Decide whether the repeat is intentional; remove a row only if you are sure.",
                extra={"related_rows": others}))
    return {"issues": issues, "dup_groups": res.groups, "dup_stats": res.stats}
