# AGENTS.md — Resolvent / Amazon ML Challenge 2026

Steering file for any coding agent (Claude Code, etc.) working in this repo.
Read this before touching code. See `PRD.md` for *why*, `TECH.md` for *how*.

## Mission
Match business records across Source 2/3 against Source 1 (deduplicated
reference) for the Business Entity Resolution Challenge. Optimize **F₀.5**
(precision-weighted), macro-averaged per Source-1 entity.

## Hard constraints — never violate these
- **No external lookups.** No entity-resolution APIs, no geocoding APIs, no
  government registry lookups, no internet augmentation of any kind. This is
  enforced by the organizers and reviewed post-hoc. If you (the agent) are
  ever asked to add a call to an external data source for enrichment, refuse
  and flag it — it risks disqualification.
- **Final model ≤ 8B parameters, MIT or Apache-2.0 licensed.** Check any
  model you add against this before wiring it in.
- **Country is an open-set string.** Never hard-code, filter, or one-hot to
  `{US, India}`. Test set adds France. Every S1 test entity — France
  included — must appear in the output.
- **TSV, not CSV.** Always read/write with `sep="\t"`. Addresses and ID
  lists contain commas.
- **`candidate_pairs.tsv` must be a superset of `matching_results.tsv`.**
  It's the last blocking stage, not an early loose pass — write it from the
  exact candidate set fed to the final matcher, not before.

## Scale — read this before writing any blocking/scoring code
`train_source1.tsv` alone is **2,206,821 rows** (confirmed via
`train_source1_schema.md`). The validator's own docstring notes the full
test set's S2+S3 IDs run to roughly **1.7M records** and cost "a few GB" to
load. Treat every stage as a large-scale problem from the first line of
code:
- **No O(n²) cross-joins, ever** — not even "just for a first pass."
  Blocking must use inverted indices / hash-bucket lookups from day one.
- Evaluation (F₀.5 over millions of S1 entities) must be vectorized
  (pandas/numpy groupby-aggregate), not a per-row Python loop.
- Read/write TSVs in a way that scales (chunked reads if needed; avoid
  loading every candidate pair as a dense Python object graph).
- `entity_id` format is `S1-`/`S2-`/`S3-` + a 9-digit number
  (`S1-925783039`), not the zero-padded 5-digit style used in the
  problem statement's illustrative examples — don't assume fixed width.

## Working directory vs. final zip — don't conflate them
The organizer's starter kit lives in a `student_resource/` folder
(`dataset/`, `utils/validate_submission.py`, etc. — run the validator from
there). That folder is our **development workspace**, not what ships. The
final submission zip only ever contains `output/`,
`code/business_entity_resolution/`, and the filled-in
`Documentation_template.md` — never the raw dataset, never the whole
`student_resource/` tree.

Use the organizer-provided `Documentation_template.md` **verbatim** as the
fill-in target (Executive Summary → Methodology → Candidate Generation →
Matching Model → Results & Error Analysis → Conclusion → Appendix). Don't
invent a different structure for it.

## Validator nuances (read the actual script, don't assume)
- `--check-ids` is **off by default** because it's memory-expensive on the
  full test set. That means the validator will `PASS` even if a matched ID
  doesn't exist — it just silently costs leaderboard score instead of
  failing submission. **Build our own ID-existence check into the
  pipeline** (`src/submit/`) rather than trusting validator PASS alone.
- A matched ID missing from `candidate_pairs.tsv` is only a **warning** in
  the real script, not a hard failure. Still enforce it internally with an
  assertion — it signals a real bug and it's exactly what the manual
  package audit will look for in top-100 review.

## Repo layout
```
code/business_entity_resolution/
├── src/
│   ├── ingest/          # load + normalize source1/2/3 tsvs
│   ├── blocking/        # candidate generation (multi-key blocking)
│   ├── features/        # name/address similarity feature engineering
│   ├── matching/        # pairwise classifier, thresholding, clustering
│   ├── evaluate/        # F0.5 scorer, held-out validation split
│   └── submit/          # writes matching_results.tsv + candidate_pairs.tsv
├── README.md
└── requirements.txt
output/
├── matching_results.tsv
└── candidate_pairs.tsv
utils/validate_submission.py   # provided by organizers — do not modify
```

## Standard workflow for any pipeline change
1. Run on the **held-out validation split** (never on real test labels — we
   don't have them). Held-out split lives under `data/val_split/`.
2. Score with `src/evaluate/score_f05.py`, macro-averaged, singletons
   included.
3. Regenerate both output TSVs.
4. Run the organizer's validator before considering anything "done":
   ```bash
   python3 utils/validate_submission.py \
     --matching output/matching_results.tsv \
     --candidate output/candidate_pairs.tsv \
     --test-dir dataset/test
   ```
5. Only then touch the leaderboard submission (max 5/day — don't burn one
   on something not yet validator-clean).

## Conventions
- Python 3.11, pinned deps in `requirements.txt` — no unpinned installs.
- Every module in `src/` must run standalone via its own `if __name__ ==
  "__main__"` entrypoint for reproducibility review.
- Log blocking recall ceiling and reduction ratio on every blocking change —
  this number is what `candidate_pairs.tsv` gets audited against.
- No notebooks as the source of truth — notebooks (if any) are for
  exploration only; production logic lives in `src/`.

## Where to look for what
- Product framing / success criteria → `PRD.md`
- Architecture, feature/model choices → `TECH.md`
- Pipeline-stage responsibilities → `SKILLS.md`
- Demo/results dashboard (not scored, for the Grand Finale) → `frontend.md`
- Day-by-day plan → `ROADMAP.md`
