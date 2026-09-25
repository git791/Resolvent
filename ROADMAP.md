# ROADMAP.md — 3-Day Execution Plan

Window: Sept 25, 12:00 AM IST → Sept 27, 11:59 PM IST. Today is Day 1.
Max 5 leaderboard submissions/day — treat as scarce, not routine.

## Day 1 — Foundation + first submission
- **Hrs 0–1: Scale sanity check, before anything else.** Confirmed sizes:
  `train_source1.tsv` = 2.2M rows, test S2+S3 ≈ 1.7M IDs. Load a full
  source file and time it; prototype the blocking index on a small sample
  first, then run it against the *full* file and confirm it finishes in
  minutes, not hours, before building anything on top of it. A pipeline
  that "works" on 10k rows but is O(n²) will not finish on the real data —
  find that out now, not on Day 3.
- **Hrs 1–3:** Data recon. Load all sources with `sep="\t"`, sanity-check
  schema against `train_source1_schema.md` (and verify source2/3 actually
  match the same 4-column shape — that's an assumption, not yet confirmed),
  look at name/address noise patterns firsthand, check train label
  distribution (singleton rate, avg matches/entity) — this shapes every
  later decision.
- **Hrs 3–5:** Build the **F₀.5 evaluation harness first** (vectorized,
  not a row-wise loop — see `TECH.md` §6), on a held-out split from
  training data. Non-negotiable before any modeling — without it every
  later decision is a guess.
- **Hrs 5–9:** Naive baseline end-to-end, run on the **full** file sizes
  from the start (not a toy sample): simplest possible blocking
  (exact/near-exact name match, indexed) → simplest matcher (threshold on
  one similarity score) → full output generation → our own ID-existence
  self-check → **run the validator** → **first leaderboard submission**.
  Goal: prove the whole pipeline works *at real scale*, not to be
  competitive yet.
- **Hrs 9–11:** Write `code/business_entity_resolution/README.md` and
  `requirements.txt` now, while the pipeline is simple — far cheaper than
  reconstructing it later under time pressure.
- **Stretch:** Start the multi-key blocking pass (Section 2 of `TECH.md`).

## Day 2 — Where the score is actually won
- **Hrs 0–4:** Multi-key blocking to convergence. Track recall ceiling +
  reduction ratio on every change; this bounds everything downstream.
- **Hrs 4–8:** Full feature engineering (name + address similarity suite),
  swap naive threshold for the GBT pairwise classifier, calibrate the
  threshold **by sweeping for max F₀.5** on the validation split, not
  accuracy.
- **Hrs 8–10:** Run the **held-out-country stress test** (train on one
  country, validate on another) — this is your only real signal on France
  generalization before test scoring reveals it. If blocking or features
  silently depend on US/India-specific patterns, this is where it shows up.
- **Hrs 10–12:** Second/third leaderboard submissions once validator-clean
  and beating Day 1's baseline meaningfully. Log every submission's config
  (this doubles as version history the organizers ask you to maintain).
- **Stretch:** Error gallery — pull actual worst false merges and worst
  misses from the validation set; these become both debugging signal *and*
  Grand Finale material.

## Day 3 — Precision polish, packaging, presentation
- **Hrs 0–4:** Precision-focused refinement only — F₀.5 punishes false
  merges 2× recall, so spend this block tightening thresholds and pruning
  weak candidate signals, not chasing more recall.
- **Hrs 4–6:** Final leaderboard submissions (respect the 5/day cap —
  don't spend them on marginal tweaks; batch validated improvements).
- **Hrs 6–8:** Assemble the submission zip: `output/`, full `code/` tree,
  filled `Documentation_template.md` written for clarity/depth (organizers
  explicitly reward this, no page limit).
- **Hrs 8–10:** Build the internal dashboard (`frontend.md`) — scorecard +
  blocking funnel + entity explorer with the error gallery already in
  hand from Day 2.
- **Hrs 10–12:** Dry-run the Grand Finale presentation against the actual
  dashboard; anticipate the two questions Amazon Scientists will ask first
  — "how do you handle the unseen country" and "how did you avoid false
  merges" — and make sure the deck/demo answers both directly.

## What "standing out" actually means here
Given the real judging structure (leaderboard F₀.5 + reviewed methodology
+ live presentation to Amazon Scientists — not tracks/sponsor-tech/pitch
scoring), the differentiators are:
1. A **held-out validation methodology that's genuinely rigorous**
   (including the unseen-country stress test) — most teams will skip this
   and get surprised by France.
2. **Precision discipline** — actively resisting the urge to chase recall,
   because F₀.5 doesn't reward it symmetrically.
3. A **methodology doc that reads like a technical report**, not a
   README — this is explicitly what's reviewed for top teams.
4. A **presentation-ready, inspectable pipeline** (the dashboard) that
   makes your reasoning legible under live questioning, instead of a wall
   of code someone has to reverse-engineer on the spot.
