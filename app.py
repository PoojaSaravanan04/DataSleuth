"""DataSleuth: simple ChatGPT-style CSV cleaning with user-approved changes."""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import detectors as D
import workflow as W
from dsa import MaxHeap, constant_hash, djb2_hash, find_duplicates

st.set_page_config(page_title="DataSleuth", page_icon="⌕", layout="wide", initial_sidebar_state="expanded")
ss = st.session_state
SAMPLE_PATH = Path(__file__).parent / "sample_data.csv"
COLORS = {"missing": "#245ca6", "confirmed": "#a8323e", "potential": "#a66b12", "duplicate": "#6f42a6", "changed": "#237a52"}
SEV_KIND = {D.SEV_CONFIRMED: "confirmed", D.SEV_POTENTIAL: "potential", D.SEV_FACT: "missing"}

CSS_PATH = Path(__file__).parent / "styles.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

for key, value in {
    "ws": None, "history": [], "epoch": 0, "upload_sig": None, "flash": None,
    "active_issue": None, "page": "Workspace", "selected_column": None, "mode": None, "smart_report": None,
    "processed": False, "pinned": set(), "smart_done": False, "smart_proposal": None,
}.items():
    if key not in ss:
        ss[key] = value


def archive_current():
    ws = ss.get("ws")
    if ws is None:
        return
    key = hashlib.sha1(ws.raw + ws.name.encode("utf-8", errors="ignore")).hexdigest()
    ss.history[:] = [h for h in ss.history if h["key"] != key]
    ss.history.insert(0, {"key": key, "name": ws.name, "created_at": ws.created_at, "workspace": ws, "pinned": key in ss.pinned})
    del ss.history[12:]


def load_file(raw: bytes, name: str, archive: bool = True):
    try:
        df = W.parse_csv(raw)
    except Exception as exc:
        ss.flash = ("error", str(exc))
        return False
    if archive and ss.ws is not None:
        archive_current()
    ss.ws = W.Workspace(raw, name, df)
    archive_current()  # make the current dataset available in Recent and Pin controls
    ss.ws.analysed = True  # detection is read-only, so issues can be shown immediately
    ss.epoch += 1
    ss.upload_sig = hashlib.sha1(raw + name.encode("utf-8", errors="ignore")).hexdigest()
    ss.active_issue = None
    ss.selected_column = None
    ss.mode = None
    ss.smart_report = None
    ss.processed = False
    ss.smart_done = False
    ss.smart_proposal = None
    ss.page = "Workspace"
    ss.flash = ("success", f"Loaded {name}. The original data has not been changed.")
    return True


def scan(ws):
    rare = bool(ss.get(f"rare_{ss.epoch}", False))
    return D.scan_dataset(ws.working_df, ws.rules_df, flag_rare=rare), D.scan_dataset(ws.original_df, ws.rules_df, flag_rare=rare)


def styled_table(df, issues, changed=None):
    cell_kind, dup_rows = {}, set()
    rank = {"missing": 1, "potential": 2, "confirmed": 3}
    for issue in issues:
        row, col = issue["row"], issue["column"]
        if col == D.ENTIRE_ROW:
            dup_rows.add(row)
            continue
        kind = SEV_KIND.get(issue["severity"], "missing")
        k = (row, col)
        if k not in cell_kind or rank[kind] > rank[cell_kind[k]]:
            cell_kind[k] = kind
    changed = changed or set()

    def paint(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for row in dup_rows:
            if row in styles.index:
                styles.loc[row, :] = f"background-color:{COLORS['duplicate']};color:#fff"
        for (row, col), kind in cell_kind.items():
            if row in styles.index and col in styles.columns:
                styles.at[row, col] = f"background-color:{COLORS[kind]};color:#fff"
        for row, col in changed:
            if row in styles.index and col in styles.columns:
                styles.at[row, col] = f"background-color:{COLORS['changed']};color:#fff"
        return styles
    return df.style.apply(paint, axis=None)


def column_summary(df, issues):
    rows = []
    for col in df.columns:
        subset = [i for i in issues if i["column"] == col]
        missing = sum(i["type"] == "Missing value" for i in subset)
        anomaly = sum(i["type"] != "Missing value" and i["type"] != "Duplicate record" for i in subset)
        rows.append({"Column": str(col), "Missing values": missing, "Anomalies / variants": anomaly})
    return pd.DataFrame(rows)


def render_pending(ws):
    p = ws.pending
    if not p:
        return
    st.markdown("#### Review before applying")
    st.warning("These are proposed changes only. The working dataset is unchanged until you confirm.")
    preview = pd.DataFrame([{"Row": c["row"], "Column": c["column"], "Original value": W.show_val(c["original"]),
                             "Proposed value": W.show_val(c["proposed"]),
                             **({"Why this suggestion": c["reason"], "Confidence": c.get("confidence", "Review")}
                                if c.get("reason") else {})} for c in p["changes"][:1000]])
    st.caption(f"{p['action_label']} · {len(p['changes'])} proposed cell change(s)")
    if not preview.empty:
        st.dataframe(preview, hide_index=True, use_container_width=True, height=min(340, 38 * len(preview) + 40))
    if len(p["changes"]) > 1000:
        st.caption(f"Showing the first 1,000 of {len(p['changes'])} proposed changes.")
    for warning in p.get("warnings", []):
        st.info(warning)
    for error in p.get("errors", []):
        st.error(error)
    remove_ok = False
    if p["kind"] == "remove":
        remove_ok = st.checkbox(f"I approve removing row(s): {', '.join(map(str, p['rows']))}", key=f"remove_confirm_{ss.epoch}")
    a, b, _ = st.columns([1.4, 1, 4])
    can_apply = not p.get("errors") and (bool(p["changes"]) or p["kind"] == "meta") and (p["kind"] != "remove" or remove_ok)
    if a.button("Confirm and apply", type="primary", disabled=not can_apply, key=f"apply_{ss.epoch}"):
        was_smart = p.get("action") == "smart_clean"
        ok, message = W.apply_pending(ws, confirm_removal=remove_ok)
        if ok and was_smart:
            ss.smart_done = True
        ss.flash = ("success" if ok else "error", message)
        st.rerun()
    if b.button("Cancel", key=f"cancel_{ss.epoch}"):
        W.cancel_pending(ws)
        ss.flash = ("info", "Cancelled. No proposed data changes were applied.")
        st.rerun()


def render_cleaning_visual(ws, issues, proposal, already_applied=False):
    """Show how the automatic cleaning proposal was formed without changing data."""
    st.markdown("#### How DataSleuth reaches its recommendations")
    stages = [
        ("01", "Profile", "Infer column types and calculate valid-value statistics."),
        ("02", "Detect", "Find missing markers, placeholders, category variants and validation issues."),
        ("03", "Recommend", "Prepare a reasoned proposal for issues that can be handled conservatively."),
        ("04", "Review", "Compare original values with proposed values before approval."),
        ("05", "Apply", "Update only the working copy after the user confirms."),
    ]
    stage_cols = st.columns(5, gap="small")
    for box, (number, title, detail) in zip(stage_cols, stages):
        with box:
            st.markdown(f'<div class="ds-pipeline-step"><span>{number}</span><b>{title}</b><small>{detail}</small></div>', unsafe_allow_html=True)

    base_df = ws.working_df.copy(deep=True)
    proposed_df = ws.working_df.copy(deep=True)
    if already_applied:
        # Reconstruct the pre-cleaning values from the proposal for an honest comparison.
        for change in proposal.get("changes", []):
            row, col = int(change["row"]), str(change["column"])
            if row in base_df.index and col in base_df.columns:
                base_df.at[row, col] = change["original"]
        proposed_scan = D.scan_dataset(proposed_df, ws.rules_df)
        base_scan = D.scan_dataset(base_df, ws.rules_df)
        before_issues = base_scan["issues"]
    else:
        for change in proposal.get("changes", []):
            row, col = int(change["row"]), str(change["column"])
            if row in proposed_df.index and col in proposed_df.columns:
                proposed_df.at[row, col] = str(change["proposed"])
        before_issues = issues
        proposed_scan = D.scan_dataset(proposed_df, ws.rules_df)
    current_counts = Counter(i["type"] for i in before_issues)
    proposed_counts = Counter(i["type"] for i in proposed_scan["issues"])
    kinds = sorted(set(current_counts) | set(proposed_counts))
    if kinds:
        comparison = pd.DataFrame([{"Issue type": k, "Before proposal": current_counts.get(k, 0),
                                    "After proposal (simulation)": proposed_counts.get(k, 0)} for k in kinds])
        st.markdown("#### Issue counts: current data vs proposed result")
        st.caption("The after-proposal values are simulated on a copy. Nothing is changed until you confirm.")
        fig = px.bar(comparison, x="Issue type", y=["Before proposal", "After proposal (simulation)"],
                     barmode="group", template="plotly_dark", color_discrete_sequence=["#7397ff", "#51c99a"])
        fig.update_layout(height=330, margin=dict(l=8, r=8, t=18, b=8), legend_title_text="",
                          xaxis_tickangle=-25, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    changes = proposal.get("changes", [])
    if changes:
        st.markdown("#### What would change")
        preview = pd.DataFrame([{"Row": c["row"], "Column": c["column"], "Before": W.show_val(c["original"]),
                                 "Proposed": W.show_val(c["proposed"]), "Reason": c.get("reason", ""),
                                 "Confidence": c.get("confidence", "Review")} for c in changes[:100]])
        st.dataframe(preview, hide_index=True, use_container_width=True, height=min(330, 42 * len(preview) + 45))
        if len(changes) > 100:
            st.caption(f"Showing 100 of {len(changes)} proposed changes.")
        changed = {(int(c["row"]), str(c["column"])) for c in changes}
        rows = sorted({r for r, _ in changed})[:12]
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("**Before · current working data**")
            st.dataframe(styled_table(base_df.loc[base_df.index.isin(rows)], before_issues),
                         hide_index=False, use_container_width=True, height=260)
        with right:
            st.markdown("**After · simulated proposal**")
            st.dataframe(styled_table(proposed_df.loc[proposed_df.index.isin(rows)], proposed_scan["issues"], changed),
                         hide_index=False, use_container_width=True, height=260)
    else:
        st.info("No safe automatic edits were proposed. Review flagged issues manually; ambiguous values are not guessed.")


def render_dsa(ws, issues):
    st.caption("Duplicate detection uses the custom chained hash table. The max-heap orders issues by priority. Hash collisions alone are not treated as duplicates.")
    c1, c2, c3 = st.columns(3)
    source = c1.radio("Dataset", ["Working copy", "Original"], horizontal=True, key=f"dsa_source_{ss.epoch}")
    buckets = c2.slider("Hash buckets", 4, 128, 16, key=f"dsa_buckets_{ss.epoch}")
    worst = c3.checkbox("Demonstrate worst-case collisions", key=f"dsa_worst_{ss.epoch}")
    df = ws.working_df if source == "Working copy" else ws.original_df
    records = [(int(r), tuple(map(str, row))) for r, row in zip(df.index, df.values.tolist())]
    res = find_duplicates(records, buckets, constant_hash if worst else djb2_hash)
    st.write(f"Duplicate records found: **{res.stats['duplicate_rows']}** · comparisons: **{res.stats['comparisons']}** · longest chain: **{res.stats['max_chain']}**")
    if res.steps:
        skey = f"dsa_step_{ss.epoch}"
        ss[skey] = min(max(1, ss.get(skey, 1)), len(res.steps))
        ss[skey] = st.slider("Step through hashing", 1, len(res.steps), ss[skey], key=f"{skey}_slider")
        step = res.steps[ss[skey] - 1]
        st.caption(f"Step {step['step']} · row {step['row']} · hash {step['hash']} · bucket {step['bucket']}")
        st.code(step["record"])
        (st.success if step["duplicate_of"] is not None else st.info)(step["outcome"])
        if step["examined"]:
            st.dataframe(pd.DataFrame([{"Stored row": e["row"], "Same hash": e["same_hash"],
                                        "Same complete record": e["same_record"], "Verdict": e["verdict"]}
                                       for e in step["examined"]]), hide_index=True, use_container_width=True)
        counts = [0] * buckets
        for item in res.steps[:step["step"]]:
            if item["inserted"]: counts[item["bucket"]] += 1
        fig = px.bar(pd.DataFrame({"Bucket": list(range(buckets)), "Stored records": counts}), x="Bucket", y="Stored records", template="plotly_dark")
        fig.update_layout(height=260, margin=dict(l=8, r=8, t=20, b=8))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Issue priority")
    if issues:
        heap = MaxHeap()
        for issue in issues:
            heap.push((int(issue.get("score", 0)), -int(issue["row"])), issue["id"])
        ordered = []
        by_id = {i["id"]: i for i in issues}
        while len(heap):
            _, iid = heap.pop()
            i = by_id[iid]
            ordered.append({"Priority": i["score"], "Issue": i["type"], "Column": i["column"], "Row": i["row"]})
        st.dataframe(pd.DataFrame(ordered), hide_index=True, use_container_width=True, height=280)


# Sidebar: ChatGPT-style sessions and pinning
with st.sidebar:
    st.markdown('<div class="ds-brand"><span class="ds-mark">⌕</span> DataSleuth</div><div class="ds-sub">Your data-cleaning workspace</div>', unsafe_allow_html=True)
    if st.button("＋  New chat", use_container_width=True, type="primary", key="new_chat"):
        archive_current()
        ss.ws = None
        ss.upload_sig = None
        ss.selected_column = None
        ss.mode = None
        ss.processed = False
        ss.smart_done = False
        ss.smart_report = None
        ss.smart_proposal = None
        ss.epoch += 1
        ss.page = "Workspace"
        st.rerun()
    st.markdown('<div class="ds-side-label">WORKSPACE</div>', unsafe_allow_html=True)
    for label, icon, page in [("Dataset", "▦", "Workspace"), ("Visualisation", "◫", "Visualisation"),
                              ("Audit & export", "↻", "Audit & export"), ("DSA explorer", "⌘", "DSA explorer")]:
        if st.button(f"{icon}  {label}", use_container_width=True, key=f"nav_{page}"):
            ss.page = page
            st.rerun()
    st.markdown('<div class="ds-side-label">PINNED</div>', unsafe_allow_html=True)
    pinned_items = [h for h in ss.history if h["key"] in ss.pinned]
    if not pinned_items:
        st.caption("Pin important datasets here.")
    for idx, item in enumerate(pinned_items):
        c1, c2 = st.columns([5, 1])
        with c1:
            if st.button(f"📌 {item['name'][:22]}", key=f"pinned_open_{idx}_{item['key']}", use_container_width=True):
                archive_current()
                ss.ws = item["workspace"]
                ss.page = "Workspace"
                ss.processed = True
                ss.epoch += 1
                ss.mode = None
                st.rerun()
        with c2:
            if st.button("×", key=f"unpin_{idx}_{item['key']}", help="Unpin"):
                ss.pinned.discard(item["key"])
                for h in ss.history:
                    if h["key"] == item["key"]: h["pinned"] = False
                st.rerun()
    st.markdown('<div class="ds-side-label">RECENT</div>', unsafe_allow_html=True)
    if not ss.history:
        st.caption("Your dataset chats will appear here.")
    for idx, item in enumerate(ss.history):
        c1, c2 = st.columns([5, 1])
        with c1:
            if st.button(f"▤  {item['name'][:23]}", key=f"history_{idx}_{item['key']}", use_container_width=True, help=item["created_at"]):
                archive_current()
                ss.ws = item["workspace"]
                ss.page = "Workspace"
                ss.processed = True
                ss.epoch += 1
                ss.selected_column = None
                ss.mode = None
                st.rerun()
        with c2:
            pinned = item["key"] in ss.pinned
            if st.button("📌" if pinned else "☆", key=f"pin_{idx}_{item['key']}", help="Unpin" if pinned else "Pin this dataset"):
                if pinned: ss.pinned.discard(item["key"])
                else: ss.pinned.add(item["key"])
                item["pinned"] = not pinned
                st.rerun()
    st.divider()
    st.markdown('<div class="ds-side-label">ADD DATA</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload CSV", type=["csv"], key="sidebar_upload")
    if up is not None:
        sig = hashlib.sha1(up.getvalue() + up.name.encode("utf-8", errors="ignore")).hexdigest()
        if sig != ss.upload_sig and load_file(up.getvalue(), up.name):
            st.rerun()
    if ss.ws is not None and st.button("↺  Reset working copy", use_container_width=True):
        ss.ws.reset()
        ss.smart_done = False
        ss.flash = ("success", "Working copy restored to the original upload.")
        st.rerun()

if ss.flash:
    kind, message = ss.flash
    getattr(st, kind, st.info)(message)
    ss.flash = None

ws = ss.ws
if ws is None:
    st.markdown("## What are we cleaning today?")
    st.markdown('<p class="ds-muted">Upload a CSV to highlight missing values, invalid values, inconsistent categories and duplicates. Your original file stays untouched.</p>', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 2.1, 1])
    with center:
        st.markdown('<div class="ds-panel"><h3>Start with a CSV</h3><p class="ds-muted">Upload your file. DataSleuth will inspect it without changing any cells.</p></div>', unsafe_allow_html=True)
        main_up = st.file_uploader("Upload your CSV dataset", type=["csv"], key="main_upload")
        if main_up is not None and load_file(main_up.getvalue(), main_up.name):
            st.rerun()
        if st.button("Load sample dataset", use_container_width=True):
            if SAMPLE_PATH.exists() and load_file(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name):
                st.rerun()
    st.stop()

ws = ss.ws
scan_w, scan_o = scan(ws)
issues, issues_original = scan_w["issues"], scan_o["issues"]
issue_by_id = {i["id"]: i for i in issues}
changed_cells, removed_rows = W.changed_cells(ws)

# Dataset header and Process action
h1, h2 = st.columns([4.4, 1.1], vertical_alignment="center")
with h1:
    st.markdown(f'<div class="ds-hero"><div class="ds-eyebrow">DATASET WORKSPACE</div><h1>{ws.name}</h1><p class="ds-muted">Upload first. Process when you are ready. Your original CSV is always preserved.</p><div class="ds-chips"><span class="ds-chip">CSV loaded</span><span class="ds-chip">Original protected</span><span class="ds-chip">Changes require approval</span></div></div>', unsafe_allow_html=True)
with h2:
    replace_up = st.file_uploader("Replace dataset", type=["csv"], key=f"replace_{ss.epoch}", label_visibility="collapsed")
    if replace_up is not None:
        sig = hashlib.sha1(replace_up.getvalue() + replace_up.name.encode("utf-8", errors="ignore")).hexdigest()
        if sig != ss.upload_sig and load_file(replace_up.getvalue(), replace_up.name):
            st.rerun()
    if st.button("⚙  Process dataset", type="primary", use_container_width=True, key=f"process_{ss.epoch}"):
        ss.processed = True
        ss.mode = None
        ss.smart_done = False
        st.rerun()

if ss.page == "Workspace":
    # Always-on detection visualisation: users can see what was found immediately after upload,
    # before selecting either cleaning workflow.
    st.markdown("## Detection visualisation")
    st.caption("This is the inspection pass. It reads the uploaded dataset, identifies suspicious cells, and highlights them without changing any values.")
    detect_steps = [
        ("01", "Read CSV", "Load rows and columns into a working view."),
        ("02", "Profile", "Infer column types and inspect value patterns."),
        ("03", "Detect", "Find missing markers, placeholders, invalid values, variants and duplicates."),
        ("04", "Highlight", "Map each finding to its row and cell so you can inspect it."),
    ]
    detect_cols = st.columns(4, gap="small")
    for box, (number, title, detail) in zip(detect_cols, detect_steps):
        with box:
            st.markdown(f'<div class="ds-pipeline-step"><span>{number}</span><b>{title}</b><small>{detail}</small></div>', unsafe_allow_html=True)

    detection_counts = Counter(i["type"] for i in issues)
    if detection_counts:
        count_df = pd.DataFrame([{"Detected issue": k, "Cells / records": v} for k, v in detection_counts.items()])
        vis_left, vis_right = st.columns([1.05, 1.95], gap="large")
        with vis_left:
            st.markdown("#### What the scan found")
            st.dataframe(column_summary(ws.working_df, issues), hide_index=True, use_container_width=True, height=270)
            fig_detect = px.bar(count_df, x="Detected issue", y="Cells / records", template="plotly_dark", color="Detected issue")
            fig_detect.update_layout(showlegend=False, height=270, margin=dict(l=8, r=8, t=12, b=8), xaxis_tickangle=-25,
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_detect, use_container_width=True)
        with vis_right:
            st.markdown("#### Where the problems are highlighted")
            st.caption("Blue = missing · red = confirmed violation · amber = potential issue · purple = duplicate · green = approved change.")
            st.dataframe(styled_table(ws.working_df.head(80), issues, changed_cells), use_container_width=True, height=470)
            st.caption("Showing the first 80 rows in this overview. Use Manual Clean filters to isolate every matching issue across the dataset.")
    else:
        st.success("The scan did not find issues covered by the current rules. You can still inspect the dataset or add validation rules.")
        st.dataframe(ws.working_df.head(30), hide_index=True, use_container_width=True)

    st.markdown('<div class="ds-explain-box"><b>What this visualisation means</b><p>DataSleuth does not change data during detection. It checks values against missing-marker rules, category-normalisation patterns, column validation rules and duplicate-record checks. The highlighted table shows the locations of findings; the charts summarise their types and columns. Choose <b>Process dataset</b> above to move to automatic recommendations or manual cleaning.</p></div>', unsafe_allow_html=True)
    st.divider()

    if not ss.processed:
        st.markdown('<div class="ds-welcome"><div class="ds-welcome-icon">✦</div><h2>Your data, ready to investigate.</h2><p>Press <b>Process dataset</b> to scan for missing values, placeholder entries, inconsistent spellings, invalid values and duplicates. Then choose automatic recommendations or spreadsheet-style manual cleaning.</p><div class="ds-step-row"><span>01 &nbsp; Upload</span><span>02 &nbsp; Process</span><span>03 &nbsp; Clean</span><span>04 &nbsp; Download</span></div></div>', unsafe_allow_html=True)
        st.stop()

    st.markdown('<div class="ds-center-title"><div class="ds-eyebrow">STEP 02</div><h2>How would you like to clean this dataset?</h2><p class="ds-muted">DataSleuth has inspected the file. Choose a workflow to continue.</p></div>', unsafe_allow_html=True)
    summary = column_summary(ws.working_df, issues)
    with st.expander("Review detected issues by column", expanded=False):
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.caption("Empty cells and standard missing markers are detected separately from placeholder values such as 'unspecified'.")
    mode_a, mode_b = st.columns(2, gap="large")
    with mode_a:
        st.markdown('<div class="ds-mode-card ds-mode-smart"><div class="ds-mode-icon">✦</div><div class="ds-eyebrow">RECOMMENDED WORKFLOW</div><h3>DataSleuth Clean</h3><p>Review explainable cleaning suggestions. Numeric gaps can use the mean or median; category gaps can use the mode; city names can be standardized. Placeholder values are surfaced for a decision, not silently deleted.</p><div class="ds-mode-tags"><span>Suggestions</span><span>Reasoning</span><span>Before / after</span></div></div>', unsafe_allow_html=True)
        if st.button("Clean with DataSleuth  →", type="primary", use_container_width=True, key=f"smart_clean_{ss.epoch}"):
            proposal = W.build_smart_clean_proposal(ws, issues)
            ss.mode = "smart"
            ss.smart_done = False
            ss.smart_proposal = proposal
            ss.smart_report = {"changes": len(proposal["changes"]), "warnings": proposal.get("warnings", [])}
            if proposal["changes"]:
                ws.pending = proposal
            else:
                ws.pending = None
                ss.flash = ("info", "No safe automatic recommendations were found. Try Manual Clean for placeholder decisions and custom corrections.")
            st.rerun()
    with mode_b:
        st.markdown('<div class="ds-mode-card ds-mode-manual"><div class="ds-mode-icon">▦</div><div class="ds-eyebrow">FULL CONTROL</div><h3>Manual Clean</h3><p>Open a spreadsheet-style workspace with every column, issue highlighting, column filters, editable cells and bulk actions. Editing one city spelling can standardize its whole variant group.</p><div class="ds-mode-tags"><span>Spreadsheet</span><span>Filters</span><span>Bulk edit</span></div></div>', unsafe_allow_html=True)
        if st.button("Open spreadsheet editor  →", use_container_width=True, key=f"user_clean_{ss.epoch}"):
            ss.mode = "user"
            ss.smart_done = False
            st.rerun()

    if ss.mode == "smart":
        st.divider()
        st.markdown("### DataSleuth Clean · proposed corrections")
        st.markdown('<div class="ds-explain-box"><b>How it decides</b><p>Numeric missing values: mean by default (you can review the value). Categorical missing values: most frequent category. Free text and dates: mark as <code>missing</code> rather than inventing a value. City/category spelling groups: standardize case and spacing, preferring a readable existing form such as <code>Chennai</code>. Values like <code>unspecified</code> are flagged for your decision. Invalid ages, malformed emails and duplicates are never guessed or deleted.</p></div>', unsafe_allow_html=True)
        active_proposal = ws.pending if ws.pending and ws.pending.get("action") == "smart_clean" else (ss.smart_proposal or {"changes": []})
        render_cleaning_visual(ws, issues, active_proposal, already_applied=bool(ss.smart_done and not ws.pending and active_proposal.get("changes")))
        if ws.pending:
            render_pending(ws)
        elif ss.smart_done:
            st.success("The approved cleaning changes are now in the working dataset.")
            st.download_button("⬇ Download cleaned CSV", data=W.working_csv(ws), file_name=f"{Path(ws.name).stem}_cleaned.csv", mime="text/csv", type="primary", use_container_width=True, key=f"smart_download_{ss.epoch}")
            st.markdown("#### Cleaned data preview")
            st.dataframe(ws.working_df, hide_index=True, use_container_width=True, height=360)
        if ss.smart_report:
            for warning in ss.smart_report.get("warnings", []): st.warning(warning)

    if ss.mode == "user":
        st.markdown('<div class="ds-manual-top"><div><div class="ds-eyebrow">MANUAL CLEANING WORKSPACE</div><h2>Spreadsheet editor</h2><p class="ds-muted">Filter, inspect highlighted cells, edit values, and apply changes to one cell or a whole spelling group.</p></div><div class="ds-sheet-badge">▦ GRID MODE</div></div>', unsafe_allow_html=True)
        all_cols = list(ws.working_df.columns)
        filt1, filt2, filt3 = st.columns([1.1, 1.1, 1.6])
        with filt1:
            filter_col = st.selectbox("Filter by column", ["All columns"] + all_cols, key=f"manual_col_filter_{ss.epoch}")
        with filt2:
            filter_kind = st.selectbox("Show rows", ["All rows", "Issues only", "Missing values", "Placeholders", "Spelling variants", "Duplicates"], key=f"manual_filter_kind_{ss.epoch}")
        with filt3:
            find_text = st.text_input("Find value", placeholder="Search across the sheet…", key=f"manual_find_{ss.epoch}")
        relevant = issues
        if filter_col != "All columns": relevant = [i for i in relevant if i["column"] == filter_col or i["column"] == D.ENTIRE_ROW]
        if filter_kind == "Missing values": relevant = [i for i in relevant if i["type"] == "Missing value"]
        elif filter_kind == "Placeholders": relevant = [i for i in relevant if i["type"] == "Placeholder-like category" or (i["type"] == "Missing value" and str(i["value"]).strip().lower() in D.PLACEHOLDERS)]
        elif filter_kind == "Spelling variants": relevant = [i for i in relevant if i["type"] == "Inconsistent category spelling"]
        elif filter_kind == "Duplicates": relevant = [i for i in relevant if i["type"] == "Duplicate record"]
        elif filter_kind == "Issues only": pass
        else: relevant = issues if filter_col == "All columns" else [i for i in issues if i["column"] == filter_col or i["column"] == D.ENTIRE_ROW]
        if filter_kind != "All rows":
            show_rows = {int(i["row"]) for i in relevant}
            sheet_df = ws.working_df.loc[ws.working_df.index.isin(show_rows)].copy()
        else:
            sheet_df = ws.working_df.copy(deep=True)
        if find_text:
            mask = sheet_df.astype(str).apply(lambda c: c.str.contains(find_text, case=False, regex=False)).any(axis=1)
            sheet_df = sheet_df.loc[mask]
        st.caption("Blue = missing · red = confirmed violation · amber = possible issue · purple = duplicate · green = changed. Click a highlighted cell to open its correction panel. One table, no duplicate preview.")
        sheet_key = f"issue_sheet_{ss.epoch}_{filter_col}_{filter_kind}"
        selection_event = st.dataframe(
            styled_table(sheet_df, relevant, changed_cells),
            use_container_width=True,
            height=520,
            on_select="rerun",
            selection_mode="single-cell",
            key=sheet_key,
        )
        selected_cells = []
        try:
            selected_cells = selection_event.selection.cells
        except Exception:
            try:
                selected_cells = selection_event["selection"]["cells"]
            except Exception:
                selected_cells = []

        if selected_cells:
            selected = selected_cells[0]
            try:
                pos_row = int(selected["row"] if isinstance(selected, dict) else selected[0])
                pos_col = int(selected["column"] if isinstance(selected, dict) else selected[1])
                row_id = int(sheet_df.index[pos_row])
                col_name = str(sheet_df.columns[pos_col])
                cell_value = ws.working_df.at[row_id, col_name]
                cell_issues = [i for i in issues if int(i["row"]) == row_id and str(i["column"]) == col_name]
                if cell_issues:
                    issue = cell_issues[0]
                    st.markdown('<div class="ds-cell-correction">', unsafe_allow_html=True)
                    st.markdown(f"#### Correct cell · {col_name} · row {row_id}")
                    st.markdown(f"**Current value:** `{W.show_val(cell_value)}`  \n\n**Detected:** {issue['type']}")
                    st.caption(issue.get("reason", "This value was flagged by the data-quality scan." ) + " Suggested next step: " + issue.get("suggestion", "Review the source value."))
                    numeric_col = ws.rules().get(col_name, {}).get("expected") == "numeric" or all(D.parse_number(v) is not None for v in ws.working_df[col_name].tolist() if not D.is_missing(v))
                    choices = ["Enter a value", "Mark as missing"]
                    if numeric_col:
                        choices = ["Fill with mean", "Fill with median", "Enter a value", "Mark as missing"]
                    if issue["type"] == "Inconsistent category spelling":
                        choices = ["Standardize this spelling group", "Enter a value", "Mark as missing"]
                    chosen_action = st.selectbox("Correction", choices, key=f"cell_action_{ss.epoch}_{row_id}_{col_name}")
                    custom_value = ""
                    if chosen_action == "Enter a value":
                        custom_value = st.text_input("Replacement value", key=f"cell_custom_{ss.epoch}_{row_id}_{col_name}", placeholder="For example: 0, missing, or Chennai")
                    elif chosen_action == "Standardize this spelling group":
                        custom_value = st.text_input("Canonical value for all variants", value=str(issue.get("dominant_value") or str(cell_value).title()), key=f"cell_canonical_{ss.epoch}_{row_id}_{col_name}")
                    scope_all = st.checkbox("Apply to all similar flagged cells in this column", value=True, key=f"cell_scope_{ss.epoch}_{row_id}_{col_name}")
                    st.caption("When enabled, the correction is proposed for every matching issue in this column, such as all negative ages or all missing ages. You will preview and approve the changes before they are applied.")
                    if st.button("Preview correction", type="primary", key=f"cell_preview_{ss.epoch}_{row_id}_{col_name}"):
                        target_issues = [issue]
                        if scope_all:
                            if issue["type"] == "Inconsistent category spelling" and issue.get("group_key"):
                                target_issues = [i for i in issues if i["column"] == col_name and i["type"] == issue["type"] and i.get("group_key") == issue.get("group_key")]
                            elif issue["type"] == "Out-of-range value":
                                target_issues = [i for i in issues if i["column"] == col_name and i["type"] == "Out-of-range value" and ((D.parse_number(i["value"]) or 0) < 0) == ((D.parse_number(cell_value) or 0) < 0)]
                            else:
                                target_issues = [i for i in issues if i["column"] == col_name and i["type"] == issue["type"]]
                        if chosen_action == "Standardize this spelling group":
                            group_key = issue.get("group_key")
                            if group_key:
                                ws.pending = W.build_group_replacement_proposal(ws, col_name, group_key, custom_value)
                            else:
                                ws.pending = W.build_edit_proposal(ws, ws.working_df.assign(**{col_name: ws.working_df[col_name]}))
                        else:
                            action_map = {"Fill with mean": "mean", "Fill with median": "median", "Enter a value": "custom", "Mark as missing": "custom"}
                            custom = "missing" if chosen_action == "Mark as missing" else custom_value
                            # Build one proposal that applies the chosen value/statistic to the selected issue group.
                            changes, errors = [], []
                            stat_values = []
                            target_rows = {int(i["row"]) for i in target_issues}
                            if chosen_action in ("Fill with mean", "Fill with median"):
                                stat_values = [D.parse_number(ws.working_df.at[r, col_name]) for r in ws.working_df.index if int(r) not in target_rows and D.parse_number(ws.working_df.at[r, col_name]) is not None]
                                if stat_values:
                                    import statistics as _statistics
                                    number = sum(stat_values) / len(stat_values) if chosen_action == "Fill with mean" else _statistics.median(stat_values)
                                    custom = str(int(number)) if float(number).is_integer() else str(round(number, 6))
                                else:
                                    errors.append(f"No valid values are available to calculate the {chosen_action.lower()}.")
                            if chosen_action == "Enter a value" and not custom_value.strip():
                                errors.append("Enter a replacement value first.")
                            for target in target_issues:
                                r, c = int(target["row"]), str(target["column"])
                                if r in ws.working_df.index and c in ws.working_df.columns:
                                    old = ws.working_df.at[r, c]
                                    changes.append({"row": r, "column": c, "original": old, "proposed": custom,
                                                    "reason": f"User-selected correction applied to {len(target_issues)} similar flagged cell(s).", "confidence": "User approved"})
                            ws.pending = {"changes": changes, "errors": errors, "warnings": [], "rows": sorted({c["row"] for c in changes}),
                                          "columns": [col_name], "issue_ids": [i["id"] for i in target_issues], "kind": "edit",
                                          "action": "cell_group_correction", "action_label": f"Manual Clean · {chosen_action} · {col_name}", "created_at": W.now_iso()}
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info(f"Selected {col_name}, row {row_id}: `{W.show_val(cell_value)}`. This cell has no detected issue.")
            except (IndexError, KeyError, TypeError, ValueError):
                st.caption("Select a cell in the table to inspect or correct it.")
        else:
            st.info("Click a cell in the table. If it has a detected issue, a correction panel will appear here.")

        st.markdown("#### Bulk actions")
        bulk_tab1, bulk_tab2 = st.tabs(["Missing / placeholder values", "Standardize spelling variants"])
        with bulk_tab1:
            eligible = [c for c in all_cols if any(D.is_missing(v) or str(v).strip().lower() in D.PLACEHOLDERS for v in ws.working_df[c].tolist())]
            if eligible:
                bc1, bc2, bc3 = st.columns([1.2, 1.4, 1.8])
                with bc1: bulk_col = st.selectbox("Column", eligible, key=f"bulk_missing_col_{ss.epoch}")
                with bc2: bulk_scope = st.selectbox("Cells to handle", ["Missing markers only", "Placeholders only", "Both"], key=f"bulk_missing_scope_{ss.epoch}")
                with bc3: bulk_action = st.selectbox("Action", ["Replace with missing", "Replace with blank", "Fill with mean", "Fill with median", "Fill with mode", "Custom value"], key=f"bulk_missing_action_{ss.epoch}")
                custom = ""
                if bulk_action == "Custom value": custom = st.text_input("Replacement value", key=f"bulk_missing_custom_{ss.epoch}")
                if st.button("Preview bulk replacement", key=f"bulk_missing_preview_{ss.epoch}", type="primary"):
                    targets = []
                    for issue in issues:
                        if issue["column"] != bulk_col: continue
                        val = ws.working_df.at[int(issue["row"]), bulk_col]
                        missing = D.is_missing(val)
                        placeholder = str(val).strip().lower() in D.PLACEHOLDERS
                        if (bulk_scope == "Missing markers only" and missing) or (bulk_scope == "Placeholders only" and placeholder) or (bulk_scope == "Both" and (missing or placeholder)):
                            if issue["type"] in ("Missing value", "Placeholder-like category") and issue not in targets: targets.append(issue)
                    action_map = {"Replace with missing": "custom", "Replace with blank": "blank", "Fill with mean": "mean", "Fill with median": "median", "Fill with mode": "mode", "Custom value": "custom"}
                    custom_value = "missing" if bulk_action == "Replace with missing" else custom
                    proposal = W.build_proposal(ws, targets, action_map[bulk_action], custom_value)
                    proposal["action_label"] = f"Manual Clean · {bulk_action} · {bulk_col}"
                    ws.pending = proposal
                    st.rerun()
            else: st.success("No missing markers or placeholder-like values found in the dataset.")
        with bulk_tab2:
            groups = {}
            for issue in issues:
                if issue["type"] == "Inconsistent category spelling" and issue.get("group_key"):
                    groups.setdefault((issue["column"], issue["group_key"]), set()).add(str(issue["value"]))
            if groups:
                group_keys = list(groups)
                labels = [f"{col}: " + " / ".join(sorted(vals)[:5]) for (col, _), vals in group_keys]
                chosen = st.selectbox("Variant group", range(len(group_keys)), format_func=lambda i: labels[i], key=f"variant_group_{ss.epoch}")
                gcol, gkey = group_keys[chosen]
                values = sorted(groups[(gcol, gkey)])
                canonical_default = next((v for v in values if v == v.title()), values[0])
                st.caption(f"These values normalize to the same category key: {', '.join(values)}. Choosing a replacement updates every matching variant in this column, not just one row.")
                replacement = st.text_input("Canonical value for the whole group", value=canonical_default, key=f"variant_replacement_{ss.epoch}_{chosen}")
                if st.button("Preview standardize all variants", type="primary", key=f"variant_apply_{ss.epoch}"):
                    ws.pending = W.build_group_replacement_proposal(ws, gcol, gkey, replacement)
                    st.rerun()
            else: st.success("No spelling variant groups were detected.")

        if ws.pending:
            st.divider()
            render_pending(ws)
        st.markdown("#### Download cleaned dataset")
        st.download_button("⬇ Download current working CSV", data=W.working_csv(ws), file_name=f"{Path(ws.name).stem}_cleaned.csv", mime="text/csv", use_container_width=True, key=f"manual_download_{ss.epoch}")

    if ss.mode is None:
        st.markdown("#### Detected issues")
        st.dataframe(column_summary(ws.working_df, issues), hide_index=True, use_container_width=True)
        st.markdown("#### Dataset preview")
        st.dataframe(styled_table(ws.working_df, issues, changed_cells), use_container_width=True, height=340)

elif ss.page == "Visualisation":
    st.markdown("## Visualisation")
    st.caption("Visualisations reflect the current working copy and refresh after confirmed changes.")
    miss = pd.DataFrame([{"Column": str(c), "Missing values": sum(D.is_missing(v) for v in ws.working_df[c].tolist())} for c in ws.working_df.columns])
    left, right = st.columns(2)
    with left:
        st.markdown("#### Missing values by column")
        fig = px.bar(miss, x="Column", y="Missing values", template="plotly_dark")
        fig.update_layout(height=320, margin=dict(l=8, r=8, t=15, b=8), xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Issues by type")
        counts = Counter(i["type"] for i in issues)
        issue_df = pd.DataFrame([{"Issue": k, "Count": v} for k, v in counts.items()])
        if not issue_df.empty:
            fig = px.bar(issue_df, x="Issue", y="Count", template="plotly_dark")
            fig.update_layout(height=320, margin=dict(l=8, r=8, t=15, b=8), xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
    viz_col = st.selectbox("Explore column", list(ws.working_df.columns), key=f"viz_col_{ss.epoch}")
    rule = ws.rules().get(viz_col, D.DEFAULT_RULE)
    if rule.get("expected") == "numeric":
        vals = [D.parse_number(v) for v in ws.working_df[viz_col].tolist() if not D.is_missing(v)]
        vals = [v for v in vals if v is not None]
        if vals:
            l, r = st.columns(2)
            with l:
                fig = px.histogram(pd.DataFrame({viz_col: vals}), x=viz_col, nbins=25, template="plotly_dark", title=f"Distribution · {viz_col}")
                fig.update_layout(height=320, margin=dict(l=8, r=8, t=35, b=8))
                st.plotly_chart(fig, use_container_width=True)
            with r:
                fig = px.box(pd.DataFrame({viz_col: vals}), y=viz_col, points="outliers", template="plotly_dark", title=f"Spread · {viz_col}")
                fig.update_layout(height=320, margin=dict(l=8, r=8, t=35, b=8))
                st.plotly_chart(fig, use_container_width=True)
        numeric_cols = [c for c in ws.working_df.columns if ws.rules().get(c, D.DEFAULT_RULE).get("expected") == "numeric"]
        if len(numeric_cols) > 1:
            nums = ws.working_df[numeric_cols].apply(lambda s: s.map(D.parse_number)).astype(float)
            corr = nums.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1, template="plotly_dark", title="Numeric correlations")
            fig.update_layout(height=max(300, min(600, 45 * len(numeric_cols))), margin=dict(l=8, r=8, t=35, b=8))
            st.plotly_chart(fig, use_container_width=True)
    else:
        vals = ws.working_df[viz_col].astype(str)
        vals = vals[~vals.map(D.is_missing)]
        freq = vals.value_counts().head(20).rename_axis("Value").reset_index(name="Count")
        fig = px.bar(freq, x="Value", y="Count", template="plotly_dark", title=f"Common values · {viz_col}")
        fig.update_layout(height=340, margin=dict(l=8, r=8, t=35, b=8), xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    comparison = []
    for c in ws.original_df.columns:
        comparison.append({"Column": c, "Original missing": sum(D.is_missing(v) for v in ws.original_df[c].tolist()),
                           "Working missing": sum(D.is_missing(v) for v in ws.working_df[c].tolist()) if c in ws.working_df.columns else 0})
    fig = px.bar(pd.DataFrame(comparison), x="Column", y=["Original missing", "Working missing"], barmode="group", template="plotly_dark", title="Original vs working copy")
    fig.update_layout(height=300, margin=dict(l=8, r=8, t=35, b=8), xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

elif ss.page == "Audit & export":
    st.markdown("## Audit & export")
    st.caption("The original download is byte-for-byte identical to the uploaded file. The working download contains approved changes only.")
    stem = Path(ws.name).stem
    a, b = st.columns(2)
    with a:
        st.markdown("#### Original CSV")
        st.download_button("Download original", data=ws.raw, file_name=f"{stem}_original.csv", mime="text/csv", use_container_width=True)
    with b:
        st.markdown("#### Approved working CSV")
        st.download_button("Download working copy", data=W.working_csv(ws), file_name=f"{stem}_cleaned.csv", mime="text/csv", use_container_width=True)
    st.markdown("#### Issue report")
    report = W.issues_frame(ws, issues, pretty=False)
    st.download_button("Download issue report", data=report.to_csv(index=False).encode("utf-8"), file_name=f"{stem}_issues.csv", mime="text/csv")
    st.markdown("#### Audit log")
    audit = W.audit_frame(ws)
    if audit.empty:
        st.caption("No cleaning decisions recorded yet.")
    else:
        st.dataframe(audit, hide_index=True, use_container_width=True, height=320)
    st.download_button("Download audit log", data=audit.to_csv(index=False).encode("utf-8"), file_name=f"{stem}_audit.csv", mime="text/csv")

elif ss.page == "DSA explorer":
    st.markdown("## DSA explorer")
    render_dsa(ws, issues)
