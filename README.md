# TreeScreen — Interpretable Decision Tree for Depression, Anxiety & Stress Screening

Major project: *An Interpretable, Multi-Condition Decision Tree Model for Early Detection and Risk Stratification of Depression, Anxiety, and Stress.*
**Screening / decision support only — not a diagnosis.** Not for admissions, employment, discipline or other consequential decisions.

## Files
```
server.py        entire backend: API, ML pipeline, safety flow, CLI
test_server.py   backend tests (pytest)
requirements.txt
frontend/        index.html, package.json, vite.config.ts, tsconfig.json,
                 src/main.tsx (whole React app), src/main.test.tsx, src/index.css
```
`data/` and `artifacts/` are created automatically (datasets, trained runs).

## Status — read first
| Item | State |
|---|---|
| App, API, ML pipeline, safety flow, admin | Implemented and tested |
| Model produced by the demo command | Trained on **SYNTHETIC** data: proves the software works, says nothing about real students |
| Supervisor dataset | **Pending.** Pipeline is ready (see "Real dataset") |
| 8 supplementary questions | Proposed by me; unvalidated, not supervisor-approved. Edit them in `server.py` (`_SUPPLEMENTARY`) |
| PHQ-9 / GAD-7 | Standard wording and 0-3 scale; **confirm licence terms** before redistributing |
| Phase-I PDF | Was not available while building; objectives follow your brief. Paste the report's references into `PROJECT_INFO["references"]` in `server.py` |
| Helpline numbers | Entered from public info; **verify before real use** (or point `CRISIS_RESOURCES_FILE` at your own JSON) |

## Run
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ADMIN_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"   # Windows: set ADMIN_API_KEY=<16+ chars>
python server.py train --synthetic --activate            # ~2 min, SYNTHETIC demo model
uvicorn server:app --port 8000                           # API docs: http://localhost:8000/docs

cd frontend && npm install && npm run dev                # http://localhost:5173  (proxies /api to :8000)
```
Tests: `pytest -q` (≈3 min, trains real models on synthetic data) · `cd frontend && npm test`.
Production frontend: `npm run build` → serve `frontend/dist` behind a proxy that forwards `/api` to the backend.
Optional env vars: `CORS_ORIGINS`, `CRISIS_RESOURCES_FILE`, `MAX_UPLOAD_MB`, `PREDICT_RATE_LIMIT_PER_MIN`, `TRAIN_N_JOBS`.

## Real dataset
CSV, one row per respondent: 24 item columns (integers 0-3; ids in `server.py` — `phq9_1..9`, `gad7_1..7`, `sup_sleep_quality`, `sup_sleep_irregular`, `sup_academic_overwhelm`, `sup_academic_worry`, `sup_social_avoidance`, `sup_routine_change`, `sup_physical_inactivity`, `sup_unable_cope`), plus at least one of `depression_risk`, `anxiety_risk`, `stress_risk`, `overall_risk` (Low/Moderate/High), optional `participant_id` (repeats ⇒ participant-grouped split). Other columns are dropped; identifier-like columns trigger a warning.
```bash
python server.py validate data.csv
python server.py train --data data.csv --name "Supervisor dataset v1" --provenance "source, date, how labels were defined" --activate
python server.py evaluate                      # comparison table; --format csv --out cmp.csv
```
If labels were *computed from PHQ-9/GAD-7 totals*, add `--no-derived-totals` (the validator flags this leakage). Only conditions that have labels are predicted; absent ones are never faked. If your questionnaire differs from the 8 proposed supplementary items, edit them in `server.py` to match your columns. The Admin page does the same via upload/train/activate/export (key from `ADMIN_API_KEY`).

## How it works (methodology)
- **Split/tuning:** stratified held-out test split (participant-grouped if needed); `GridSearchCV` on training data only, refit on macro-F1 (or macro-recall / balanced accuracy); macro precision/recall/balanced accuracy recorded for every candidate.
- **Preprocessing:** median imputation fitted on training data inside the pipeline; derived PHQ-9/GAD-7/supplementary totals are deterministic sums; scaling for logistic regression only. Imbalance: tuned `class_weight='balanced'`; no resampling, so the test split is never altered.
- **Primary model:** pruned CART tuned over criterion, depth 3-6, leaf size, class weight, cost-complexity alpha. **Benchmarks:** Random Forest, XGBoost, LightGBM, Logistic Regression on the same split/protocol; unavailable libraries are reported as skipped.
- **Metrics:** accuracy, balanced accuracy, macro precision/recall/F1, one-vs-rest ROC-AUC (probabilities uncalibrated), per-class P/R/F1, confusion matrix; JSON/CSV export; runs are append-only.
- **Explanations** come from the fitted tree (`decision_path`, rules, node counts) with a built-in check that the traced leaf equals the prediction. Leaf class shares are raw training counts, not calibrated probabilities. Association, not causation.
- **"Highest level"** on results is the max of the condition levels (labelled derived).
- **Safety/privacy:** any non-zero PHQ-9 item-9 answer shows support/helpline guidance regardless of the model (works with no model too). No identifiers requested; answers never stored or logged; no third-party calls. Admin: bearer key (≥16 chars, otherwise admin is disabled), lockout after failed attempts, rate limits, 5 MB `.csv`-only uploads. Single-process design (in-memory limiter/jobs): run one worker.

## API (OpenAPI at `/docs`)
Public: `GET /api/{health,project-info,questionnaire}`, `/api/model/{status,features,metrics,tree}`, `POST /api/screening/predict` (`{"answers": {24 ints 0-3}}`, unknown keys rejected).
Admin (`Authorization: Bearer $ADMIN_API_KEY`): `POST /api/admin/dataset/{validate,upload,synthetic}`, `GET /api/admin/datasets`, `POST /api/admin/model/train` (background job), `GET /api/admin/jobs[/id]`, `GET /api/admin/evaluations[/id[/export?format=json|csv]]`, `POST /api/admin/model/activate/{run_id}`.

## Phase-I objectives → implementation
Pruned CART (`train_and_evaluate`) · 24 items PHQ-9/GAD-7/supplementary (`SECTIONS`) · sleep/academic/behaviour items (supplementary section) · Low/Moderate/High per condition (targets) · if-then paths (`explain` section, Results + Explainable AI pages) · imbalance + recall (balanced weights, per-class recall) · fair benchmarks · full metrics · low-cost deployment (one process, no DB/GPU) · privacy + non-diagnostic framing. The About page shows the same mapping from the API.

## Test results (actual runs)
See the final message for the latest numbers; not verified: Docker (none provided), automated accessibility audit, screen-reader testing, any real-data performance.
