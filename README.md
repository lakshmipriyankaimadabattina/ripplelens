# RippleLens: change impact prediction (ENGR 5570, Group 2)

File-level change impact prediction on Apache Commons / PDFBox history. Given a seed file from a real commit,
rank all other Java files by how likely they changed in the same commit. Methods compared: random, same-package,
static (import graph), static+ (tie-broken import graph), co-change history, a learned combined model, and a small LLM pilot.

Authors: Jagadeeswara Reddy Putluri, Lakshmi Priyanka Imadabattina, Siddartha Reddy Challa (Group 2).

## Layout
| Path | What it is |
|---|---|
| `src/ripplelens_m3.py` | Pipeline: mining, filters, rankers, combined model, metrics (one repo per run) |
| `src/analyze_m3.py` | Pools results: tables, Wilcoxon + Holm, change type, time thirds, ablations |
| `src/ripplelens_baseline_a2.py` | Milestone 2 baseline script (3 methods) |
| `repos.lock` | Pinned repositories and full commit SHAs |
| `results/m3/` | Per-commit results (`*_results.jsonl`), per-repo summaries, `analysis.json`, figures |
| `results/a2/` | Milestone 2 summaries and results |
| `llm_pilot/` | 40-commit LLM pilot: prompts, answers, scoring |
| `view/` | Ripple view prototype (`ripplelens_view.html`, self-contained) and its DOM test |
| `docs/` | Milestone reports |

## Reproduce
Requirements: Python 3.12, git, ~2 GB disk for the clones. `pip install -r requirements.txt`

```
make check-analysis   # fast: re-run only the analysis on the committed per-commit results
make reproduce        # clone pinned repos, rerun the whole pipeline (~2 min in parallel), compare
```
`verify.py` compares regenerated numbers with `results/m3/analysis.json` (tolerance 1e-9). Seeds are fixed (42; per-commit seeds from the commit SHA).
Environment used: Linux, Python 3.12.3, scikit-learn 1.9.1, SciPy 1.17.1, git 2.43.0.

## Method summary
- Test set: latest 30% of non-merge commits; 2 to 15 Java files; rename-dominated and bulk-subject commits dropped; up to 3 random seed files per commit; truth = other pre-existing changed files.
- All statistics for a commit come from strictly older commits. The combined model (HistGradientBoosting) is trained only on the history window.
- Sample: up to 300 commits per repository (seed 42) plus every eligible commit after 2026-07-01.

## LLM pilot (read before citing)
40 commits, one seed each, Claude Sonnet 5.5 via the claude.ai chat interface on 2026-11-08 (sampling settings not controllable, so not exactly reproducible).
The prompt contains the seed path, the commit subject and an alphabetically sorted candidate list (union of top-10 of four methods); truth was not shown.
`llm_pilot/llm_eval.json` holds the scores. A scripted, logged API run is planned.

## Known limitations
Ground truth is "changed in the same commit", not "really affected"; change-type labels are keyword heuristics; all repositories are Apache Java projects.

## License
Add a license of your choice (for example MIT) before making the repository public.
