# Cold-Start-Aware Electronics Recommendation System

This repository contains an educational, reproducible prototype for a third-year semester project. It compares a popularity/history-based electronics recommender with an explainable content/value-based approach for new products and new users.

The initial dataset will be synthetic and clearly labelled. The project does not scrape Amazon, use private customer data, reproduce proprietary algorithms, or require external credentials or paid services.

## Phase 1 status

Phase 1 establishes the Python package, shared configuration, dependency manifest, and test layout. The later phases will add synthetic data generation, validation, feature preparation, recommenders, evaluation, and a Streamlit demonstration.

## Setup

Use Python 3.11 or newer in a local Windows environment:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

The project uses a fixed random seed (`42`) and a default recommendation size of five items. Data and experiment outputs are kept under the project-relative `data/` and `results/` paths when those phases are implemented.

## Verification and development commands

```bash
# Focused Phase 1 test
python -m pytest tests/test_config.py -v

# One test by name
python -m pytest tests/test_config.py::test_configuration_defines_reproducible_project_defaults -v

# Full test suite
python -m pytest -q

# Planned commands after the corresponding phases exist
python scripts/generate_data.py --seed 42
python scripts/run_experiments.py --seed 42
streamlit run app.py
```

For project scope, architecture, constraints, and the current handoff, read `AGENT.md`, `PROJECT_PROGRESS.md`, and the design and implementation plan under `docs/superpowers/`.
