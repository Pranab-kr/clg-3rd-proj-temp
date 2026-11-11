# Demo script: Cold-Start-Aware Electronics Recommender

Audience: project team and HOD. Length: about 8 minutes. All numbers below were produced by the commands in this script on **synthetic data, seed 42**. If you change the seed or code, re-run and update them.

## 0. Setup (before the audience arrives)

From the repository root, in Git Bash:

```bash
.venv/Scripts/python.exe scripts/generate_data.py --seed 42     # 120 products, 80 users, 1,200 interactions
.venv/Scripts/python.exe scripts/run_experiments.py --seed 42   # writes results/
.venv/Scripts/python.exe -m pytest -q                            # expect: 72 passed
.venv/Scripts/python.exe -m streamlit run app.py                 # opens http://localhost:8501
```

## 1. The problem (1 min)

- A history-based recommender ranks by what people already clicked, rated, and reviewed. A **brand-new product has no history**, so it can never rank, however good it is. A **new user** has no history either.
- This prototype compares that baseline with an explainable recommender that scores **product metadata + stated preferences + value/quality**, plus a small, bounded boost for suitable zero-history items.
- Point at the yellow banner: **everything is synthetic**. No Amazon data or algorithm is used.

## 2. New-user scenario: Television (3 min)

Sidebar: Scenario = *New user (preferences)*, Category = *Television*, leave the budget at the full range, no features selected.

What to show:

- **Baseline (left):** `HomeBright Television 016`, `BrightTech Phone 041`, `NovaCompute Laptop 063`, `NovaCell Phone 059`, `Northstar Laptop 039`. It ignores the user's category (Phones and Laptops appear) because it only knows catalog-wide popularity (e.g. "18 recorded interactions, 4.8 average rating, 3,898 reviews"). **No new item appears.**
- **Proposed (right):** `NovaLaunch10`, `NovaLaunch4`, `NovaLaunch7`, `NovaLaunch1` (all Televisions, tagged 🆕 "New item (no history)"), then `VisionArc Television 109`. Read one reason aloud, e.g. NovaLaunch10: *"Television with 83% preference similarity, priced at 454.09, 2-year warranty, 25% discount, strong value score, new item with no assumed interaction history."*
- Note the banner under the tables: new zero-history items in the top 5: baseline 0, proposed 4.

Then add features *4K* and *HDR* and set the budget to 300–1000 to show the controls respond (proposed top 5 is still `NovaLaunch10, 7, 4, 1`, then `NovaView Television 055`).

**Say plainly:** NovaLaunch4 (1,432) and NovaLaunch7 (1,298) are priced above that 300–1000 budget. Price fit is a soft penalty, not a filter, and these new items score highly on discount, warranty, and rating. This is a known limitation (see section 5).

## 3. Existing synthetic user (1 min)

Scenario = *Existing synthetic user*, pick `u0001` (prefers Television, budget 776–1,073, features 4K|HDR; marked as a new user in the data).

- Baseline is the same catalog-wide list as above.
- Proposed: `NovaLaunch7`, `NovaLaunch10`, `NovaLaunch4`, `NovaLaunch1`, `VisionArc Television 064`.

## 4. Baseline vs proposed numbers (2 min)

Scroll to **Saved experiment comparison** (from `results/recommender_comparison.csv`; leave-one-out, K=5, the same 80 users and training history for both methods):

| method | Precision@5 | Recall@5 | catalog coverage | new-item exposure | relevant new items surfaced | avg rating | avg value |
|---|---|---|---|---|---|---|---|
| baseline | 0.0150 | 0.0750 | 0.092 | 0.000 | 0.00 | 4.486 | 0.610 |
| proposed | 0.0025 | 0.0125 | 0.217 | 0.6875 | 1.00 | 4.446 | 0.686 |

Talking points:

- The proposed method **wins on the cold-start goals**: coverage (0.22 vs 0.09), new-item exposure (0.69 vs 0), relevant new items surfaced (1.00 vs 0), with average rating almost unchanged and a higher average value score.
- It **loses on Precision@5 and Recall@5**. Do not hide this. The synthetic interactions are random over non-new products, so history carries no preference signal and popularity is the best guess for a random held-out item. New items can never be held-out positives because they have no interactions by design.
- Optional extras in `results/`: classifier comparison (accuracy 0.51–0.56, F1 0.23–0.40, Random Forest best F1 0.404; proxy engagement label, near chance because labels are random relative to features) and K-Means user clusters (k=3, silhouette 0.304; clusters split by category/budget).

## 5. Limitations to state (1 min)

1. Data is synthetic; results show the mechanism, not commercial performance.
2. Proposed loses on Precision/Recall for the reason above; offline top-K metrics do not predict real satisfaction.
3. New-item exposure (0.69) is high. The exposure weight is bounded (default 0.10, max 0.25), but synthetic new items are also generated with high discount/rating/warranty, so they rank well on relevance and quality alone. Not tuned.
4. Budget is a soft penalty, so over-budget new items can appear (section 2).
5. Classification labels are proxy engagement, not purchase intent. No deep learning was added (optional, deferred).
6. Only one held-out item per user, so precision/recall are coarse.

## Likely questions

- *"Why not use real data?"* No dataset was supplied, and the cold-start cases must be controlled and reproducible. Synthetic data is labelled as such everywhere.
- *"Is the boost unfair?"* It applies only to zero-history new items, is capped at 0.25, and the sidebar exposure slider shows its effect: at weight 0 (Television, 300–1000, 4K/HDR) the top 5 contains 2 new items (`NovaLaunch10`, `NovaLaunch7`) instead of 4, so most of the new-item presence comes from relevance and quality, not the boost.
- *"How is a recommendation explained?"* Reasons are built from the product's actual metadata and score components; new-item text says no history is assumed.

## Reset between runs

Stop Streamlit with Ctrl+C. Data and results regenerate deterministically from the seed (steps 0), so no manual cleanup is needed.
