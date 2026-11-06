# Credit Risk Portfolio Analytics

A credit-risk monitoring dashboard with two embedded AI features — an LLM
"Ask the Portfolio" Q&A assistant and anomaly detection — built from 5
synthetic bank tables (customers, loan accounts, bureau credit, repayment
history, collections events).

**Python version:** 3.11.6

## Run instructions (3 commands)

```bash
pip install -r requirements.txt
python -m src.build          # ingest, clean, engineer features, train model, score anomalies
streamlit run app/app.py     # launch the dashboard
```

On macOS, XGBoost also requires the OpenMP runtime: `brew install libomp`
(one-time, not a Python dependency).

## LLM API key

The "Ask the Portfolio" feature uses an OpenAI model if a key is available,
and otherwise falls back to a deterministic rule-based query engine over the
same grounded aggregate data (still fully functional, no hallucinated
numbers, just no free-text generation).

- **Local:** copy `.env.example` to `.env` and set `OPENAI_API_KEY`, or
  `export OPENAI_API_KEY=...` before running Streamlit.
- **Hosted (Streamlit Community Cloud):** add `OPENAI_API_KEY` under
  *App settings → Secrets*.
- **If no key is set:** the dashboard shows "LLM mode: OFF — using
  rule-based fallback" and still answers questions from the real numbers.

## Repository layout (brief maps to the assignment)

| Brief item | Location |
|---|---|
| Data ingestion & cleaning, DQ decisions | `src/ingestion.py` |
| Feature engineering (account/customer aggregates) | `src/features.py` |
| Portfolio analytics & insights | `src/analytics.py` |
| Default model (Logistic Regression + XGBoost), metrics | `src/model.py` |
| Anomaly detection | `src/anomaly.py` |
| "Ask the Portfolio" LLM feature | `src/llm_qa.py` |
| End-to-end build pipeline | `src/build.py` |
| Dashboard (portfolio view, account view, filters, both AI features) | `app/app.py` |
| Architecture diagram | `docs/architecture.pdf` |
| Solution overview (insights, model results, assumptions) | `docs/solution_overview.pdf` |
| Raw data | `data/raw/*.csv` |
| Tests | `tests/` |

## Regenerating the one-page PDFs (optional, not required to run the app)

`docs/architecture.pdf` and `docs/solution_overview.pdf` are pre-built and
committed. To regenerate them: `pip install reportlab && python scripts/make_docs.py`
(reportlab is a doc-authoring tool only, not an app runtime dependency).

## Notes

- `data/processed/` and `artifacts/` are build outputs (gitignored for size);
  regenerate them with `python -m src.build`.
- No secrets are committed; `.env` is gitignored, only `.env.example` is tracked.
