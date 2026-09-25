# PRD.md — Resolvent: Business Entity Resolution

## Objective
Build the highest-F₀.5, fully compliant entity-resolution pipeline for the
Amazon ML Challenge 2026, and produce a submission package + presentation
that stands out on rigor at the Grand Finale.

## Success metrics (in priority order)
1. **Leaderboard F₀.5** (public → private) — the only scored number during
   the challenge window
2. **Validator PASS** on every submission — a rejected format never scores,
   regardless of model quality
3. **Top-100 qualification** → unlocks the methodology/architecture review
4. **Methodology document quality** — explicitly reviewed for top teams;
   clarity and depth are rewarded over brevity per the organizers' own
   instructions
5. **Grand Finale presentation credibility** — not separately scored on the
   leaderboard, but is the actual moment of differentiation in front of
   Amazon Scientists

## In scope
- Full pipeline: ingestion → blocking → feature engineering → matching →
  thresholding/clustering → submission packaging
- Held-out validation harness (no test ground truth is provided — we build
  our own)
- Generalization to an **unseen country (France)** at test time — this is
  the single biggest risk in the whole problem and must be treated as a
  first-class design constraint, not an edge case
- Internal demo dashboard for the Grand Finale (see `frontend.md`)
- Full submission zip: `output/`, `code/business_entity_resolution/`,
  `Documentation_template.md`

## Out of scope / explicitly prohibited
- Any external API/database lookup for entity resolution, geocoding, or
  business registry verification — instant disqualification risk
- Any model over 8B parameters or without an MIT/Apache-2.0 license
- Anything resembling a "pitch a novel product idea" exercise — there is no
  track selection or sponsor-tech bonus scoring in this competition; the
  deliverable is the fixed entity-resolution task

## Constraints (hard, from organizer docs)
- TSV I/O only, exact column names, no duplicate IDs/rows, S2/S3-only match
  IDs, every S1 test entity present
- ≤8B params, MIT/Apache-2.0 license on the final model
- Max 5 leaderboard submissions/day, 3-day window
- No external data augmentation of any kind

## Functional requirements
| Module | Requirement |
|---|---|
| Ingestion | Parses all three sources with `sep="\t"`; normalizes names/addresses without destroying entity-distinguishing signal |
| Blocking | Multi-key, country-agnostic (must generalize to France); logs recall ceiling + reduction ratio every run |
| Feature engineering | Name + address similarity features; no feature implicitly hard-coded to {US, India} |
| Matching model | Pairwise classifier meeting the license/size constraint; calibrated for precision given F₀.5's 2:1 precision weighting |
| Clustering/output | Correct one-to-many handling, singleton correctness prioritized (full 1.0 credit, cheap to protect) |
| Validation | Wraps organizer's `validate_submission.py`; blocks any leaderboard upload that doesn't PASS |
| Dashboard | Read-only, local, presentation-ready (see `frontend.md`) |

## Methodology doc — deliverable mapping
The organizer's `Documentation_template.md` (provided, use verbatim) has 6
numbered sections + appendix. Map our work to it directly rather than
writing a separate doc: Executive Summary → §1; EDA/noise findings → §2.1;
approach + **named core innovation** → §2.2; blocking strategy + recall
protection → §3; features/model/threshold method → §4; validation F₀.5 +
error gallery → §5; wrap-up → §6; code structure + entry points → Appendix
A; extra charts (can reuse the `frontend.md` dashboard views) → Appendix B.

## Timeline
See `ROADMAP.md` for the day-by-day plan across the 3-day window.

## Risks
- **Scale** — `train_source1.tsv` is 2.2M rows; test S2/S3 combined is
  ~1.7M IDs per the validator's own docstring. Anything not built as an
  indexed/vectorized pipeline from day one will not finish in the window.
  Confirmed via `train_source1_schema.md` and `validate_submission.py` —
  this is now a fact, not a guess, and Day 1 hour 0 is a scale check.
- **France generalization** — no training examples; mitigate by
  synthetically holding out one training country during dev to
  stress-test true out-of-distribution behavior before test data proves it
  either way
- **Over-fitting blocking to US/India address formats** — mitigate with
  component-based (not pattern-based) address features
- **Burning submissions** — always validate + score on held-out split
  first; treat the 5/day cap as scarce
- **Validator PASS ≠ correct.** ID-existence checking is off by default in
  the provided validator (memory-cost tradeoff) — build our own check
  rather than trusting a green `PASS` alone.
