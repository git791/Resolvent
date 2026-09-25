# SKILLS.md — Pipeline Skill Catalog

Each "skill" below is a self-contained pipeline stage: a clear input, output,
and success measure. Treat each as independently testable.

## 1. Ingestion & Normalization
- **Input:** raw `*_source{1,2,3}.tsv`
- **Output:** normalized records — lowercased, whitespace-collapsed name and
  address, legal-suffix map applied (Corp/Corporation, Pvt/Private,
  Ltd/Limited → canonical tokens), `&`/`and` unified, country kept as raw
  open-set string
- **Success measure:** spot-check normalization doesn't destroy legitimate
  distinguishing info (e.g. don't merge "Pvt Ltd" away if it changes entity
  identity signals — legal-suffix normalization is for *matching*, not for
  the stored canonical name)

## 2. Blocking / Candidate Generation
- **Input:** normalized S1 records + normalized S2/S3 pools
- **Output:** `candidate_pairs.tsv` (S1 → set of plausible S2/S3 IDs)
- **Strategy:** multi-key blocking — union of candidates from (a) normalized
  name n-gram/token overlap, (b) phonetic code (Soundex/Metaphone) on name,
  (c) address token overlap, partitioned by country so blocking logic never
  assumes a fixed country set
- **Success measure:** **recall ceiling** (fraction of true matches present
  in the candidate set) and **reduction ratio** (candidates generated vs.
  full cross-product) — log both on every change. This determines the upper
  bound of leaderboard score; be more generous here than in the matcher.
  **Must also run on the real ~2.2M-record scale in tractable time** — an
  inverted-index/hash-bucket implementation, never a nested-loop
  cross-product (see `TECH.md`).

## 3. Feature Engineering
- **Input:** each (S1, candidate) pair
- **Output:** feature vector — name similarity (Jaro-Winkler, token Jaccard,
  TF-IDF cosine, normalized edit distance), address similarity (token
  overlap, component-wise match, abbreviation-aware edit distance),
  structural signals (country match, name length ratio)
- **Success measure:** feature importances make sense; no feature silently
  encodes a train-only artifact (e.g. don't let a feature implicitly assume
  only US/India ever appear)

## 4. Matching Model
- **Input:** feature vectors + training labels (from ground truth)
- **Output:** match probability per pair
- **Constraint check:** ≤8B params, MIT/Apache-2.0 license — verify before
  adopting any pretrained component
- **Success measure:** validation F₀.5, with explicit precision/recall
  breakdown (F₀.5 punishes false merges 2× — watch precision specifically)

## 5. Thresholding & Clustering
- **Input:** pairwise probabilities
- **Output:** final S1 → {matched IDs} groups, one-to-many resolved
  consistently (no candidate double-counted, no cycle inconsistencies)
- **Success measure:** singleton accuracy (correctly predicting "no match")
  — this is worth a full 1.0 per entity and is cheap precision to protect

## 6. Evaluation Harness
- **Input:** predictions + held-out validation ground truth
- **Output:** per-entity F₀.5, macro-average, confusion breakdown, error
  gallery (worst false merges, worst missed matches)
- **Success measure:** this is the tool the whole team trusts before ever
  spending one of the 5 daily leaderboard submissions. Must score millions
  of S1 entities via vectorized groupby, not a per-row Python loop.

## 7. Submission Packaging & Validation
- **Input:** final matches + full candidate set actually scored
- **Output:** `output/matching_results.tsv`, `output/candidate_pairs.tsv`,
  passing `utils/validate_submission.py`
- **Success measure:** `PASS`, exit 0, every time before upload. Note the
  provided validator doesn't check ID-existence by default (memory cost)
  — our own submit step must independently verify matched/candidate IDs
  are real, since a `PASS` alone doesn't guarantee that.
