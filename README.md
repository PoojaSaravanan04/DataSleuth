# DataSleuth — Interactive Data Quality Investigator

DataSleuth is a local web application with a custom HTML/CSS/JavaScript frontend and a Flask/Python backend. Streamlit is no longer used by the main application.

## Features
- ChatGPT-inspired sidebar, dataset sessions, and pin controls (browser-session prototype).
- CSV upload and read-only detection visualisation, including an animated row-by-row scan replay that reveals highlights as each row is inspected.
- Cell-level issue highlighting and filters for missing values, violations, anomalies, duplicates, and approved changes.
- DataSleuth Clean: explainable cleaning proposal with a five-stage pipeline, before/proposed issue chart calculated by running the detectors on a temporary copy, and a reviewable change table.
- Data visualizer: choose X and Y columns and create bar, line, scatter, pie, or histogram charts; export chart data as CSV.
- Preview and explicit confirmation before changes are applied.
- Cleaned CSV download and audit view.
- Custom hash table and max-heap remain in `dsa.py`.

## Run on Windows
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python server.py
```
Open http://127.0.0.1:5000 in your browser.

## Run tests
```bat
python test_dsa.py
python test_workflow.py
```

## Notes
- The current recommendation engine is rule/statistics based; it does not call an LLM API.
- Original CSV bytes and original DataFrame are retained; proposal generation does not change the working data.
- This is a local single-user prototype. The in-memory workspace store is not intended for multi-user public hosting.
- Charts are drawn locally with the browser canvas API; no external chart CDN is required. The row-by-row display is a replay of the backend scan results, not a claim that the Python detector executes one network request per row.

## Algorithm trace and saved investigations

The main Data Analysis workflow includes an expandable **How DataSleuth detects issues** panel:
- **Duplicate detection:** shows operations from the custom separate-chaining hash table, including bucket selection, comparisons, collisions, and duplicate confirmation.
- **Missing values:** shows the direct cell-by-cell missing-marker checks. Missing values are not detected with hashing.
- **Issue priority:** shows operations from the custom binary max-heap using the detector's documented priority score.

Investigation history is stored locally in `datasleuth_history.sqlite3`. Saved workspaces include the original upload and current workspace/audit state, and can be reopened after refreshing or restarting the local Flask server. Pin status is also stored. Keep the database file if you want to retain history; deleting it deletes the saved investigations.
