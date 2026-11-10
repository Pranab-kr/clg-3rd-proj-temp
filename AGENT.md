# Agent guidance

## Project context

This repository contains an educational third-year semester project: a cold-start-aware electronics recommendation system. The intended demonstration compares a popularity/history-based baseline with an explainable content/value-based recommender that can surface suitable new-brand products with no interaction history.

This is a local, reproducible prototype, not an Amazon clone or a production marketplace. Do not claim access to Amazon's private data or algorithm. The initial dataset is synthetic and must be labelled as synthetic in code, output, and report. The project should demonstrate both new-item and new-user cold start without external credentials or paid services.

## Source-of-truth documents

Read these before changing the implementation:

- `PROJECT_PROGRESS.md` — current phase, handoff status, decisions, and next steps.
- `docs/superpowers/specs/2026-11-10-cold-start-electronics-recommender-design.md` — approved project intent, scope, constraints, and success criteria.
- `docs/superpowers/plans/2026-11-10-cold-start-electronics-recommender-plan.md` — implementation tasks, interfaces, test expectations, and execution order.
- `docs/report-and-presentation-outline.md` — report, presentation, and viva content that must match the actual implementation.

Update `PROJECT_PROGRESS.md` after each completed phase with the date, files created, commands run, test results, and deferred work. Never invent experiment values, screenshots, or claims in the report.

## Architecture direction

Keep the core path modular and replaceable:

`synthetic data generator -> validation -> feature preparation -> baseline/proposed recommenders -> evaluation -> Streamlit UI`

The planned Python package separates configuration, data generation/validation, feature encoding, classification, clustering, recommendation scoring, explanation text, and evaluation. `app.py` should remain a thin Streamlit entry point that calls those modules rather than containing model-training logic. The baseline ranks established products from documented popularity/rating signals. The proposed recommender uses product metadata, user preferences or interaction profiles, value/quality signals, and a bounded cold-start exposure adjustment; it must return human-readable reasons.

The intended stack is Python 3.11+, pandas, NumPy, scikit-learn, Streamlit, pytest, and Matplotlib/Seaborn. Use a fixed random seed and stable feature-column order so generated data, recommendations, and metrics are reproducible.

## Working commands

The implementation is being built in phases. Once the project skeleton exists, the standard commands are:

```bash
# Create and activate a Windows virtual environment
python -m venv .venv
source .venv/Scripts/activate

# Install dependencies
python -m pip install -r requirements.txt

# Run the focused Phase 1 test
python -m pytest tests/test_config.py -v

# Run all tests
python -m pytest -q

# Generate the synthetic CSV snapshot (after Phase 2)
python scripts/generate_data.py --seed 42

# Run experiments (after the recommender/evaluation phases)
python scripts/run_experiments.py --seed 42

# Launch the demo (after the Streamlit phase)
streamlit run app.py
```

Use `python -m pytest path/to/test_file.py -v` for a single test file and `python -m pytest path/to/test_file.py::test_name -v` for one test. Commands that depend on files from a later phase should not be run or documented as currently available until those files exist.

## Current implementation phase

Phase 3 is complete. The repository has a reproducible synthetic dataset generator and validation layer in `src/data_generation.py` and `src/data_validation.py`, plus stable product/user feature preparation and deterministic K-Means clustering in `src/features.py` and `src/clustering.py`. The default snapshot contains 120 products, 80 users, and 1,200 interactions. New-item products are deliberately excluded from generated historical interactions, and new-user profiles are explicitly marked.

The next phase is the baseline and cold-start-aware recommender. Before changing implementation, run the full suite with `.venv/Scripts/python.exe -m pytest -q`, reuse the fitted encoder from `src/features.py`, and follow the plan's test-first sequence for `src/recommenders.py` and `src/explanations.py`. Generate local data with `.venv/Scripts/python.exe scripts/generate_data.py --seed 42` when needed; CSV outputs are ignored by Git.

Phase 1 established the configuration contract:

- `RANDOM_SEED` is an integer, initially `42`.
- `TOP_K` is `5`.
- Categories include `Television`, `Phone`, and `Laptop`.
- Data and results paths are project-relative.
- The environment and commands are documented without requiring external services.

Phase 2 contract:

- Products, users, and interactions have explicit required schemas.
- Validation reports missing columns and invalid ranges without hiding independently invalid values.
- Same seed and parameters produce equivalent tables.
- At least one new item has zero historical interactions, and at least one user is marked as new.
- The CLI works when invoked as `python scripts/generate_data.py --seed 42` from the repository root.

## Important constraints

- Keep the controlled new-brand/zero-history scenario reproducible; do not hard-code a particular product or brand into recommender logic.
- Support a preference-driven new user with no interaction rows.
- Treat invalid category/budget and missing optional metadata with explicit validation or documented safe fallbacks.
- Keep exposure adjustment small, transparent, bounded, and subordinate to relevance.
- Classification labels are proxy engagement/interaction labels, not purchase intent.
- Deep learning is optional and must not block the core classical solution.
- Do not add scraping, authentication, payments, private customer data, paid services, or proprietary branding.
- Keep synthetic records clearly separate from any future public dataset.

## Handoff expectations

Before ending work, run the most relevant verification commands, report failures honestly, and update `PROJECT_PROGRESS.md`. If a later document conflicts with the progress log, treat the latest explicit user decision and the approved design as authoritative, then document the resolution.
