# TECH.md — Technical Design

## Scale (confirmed, not estimated)
`train_source1.tsv` = 2,206,821 rows. Validator docstring confirms the
full test set's S2+S3 IDs run ~1.7M records, "a few GB" in memory. Every
design decision below assumes this scale, not a sample-sized dev set.

## Pipeline overview
```
raw TSVs → normalize → block (candidate_pairs) → featurize
         → pairwise classify → threshold + cluster → matching_results
                                                     → validate → submit
```

## 1. Normalization
- Lowercase, collapse whitespace/punctuation
- Legal-suffix canonicalization table (Corp/Corporation, Pvt/Private,
  Ltd/Limited, Inc, LLC, etc.) — applied for *matching features only*;
  keep raw fields untouched in the record store
- Address abbreviation table (Rd/Road, St/Street, Ave/Avenue, ...)
- Keep `country` as-is, raw string — no enum, no allowlist

## 2. Blocking strategy (candidate generation)
At ~2.2M S1 records against a multi-million S2/S3 pool, blocking is an
**indexing problem, not a comparison problem** — every pass below must
build a hash/dict-based inverted index (key → list of entity_ids) and look
up candidates in O(1) per key, never iterate the full cross-product. Union
of independent blocking passes, deduped, partitioned by country so no
single blocking key structurally excludes an unseen country:
- **Name-token blocking:** index by sorted significant-token sets
  (stopwords/legal suffixes stripped); candidates share ≥1 token
- **Phonetic blocking:** Soundex/Metaphone code on the primary business
  name token, to catch transliteration/typo variants token-blocking misses
- **Address-token blocking:** shared street-name or locality tokens after
  abbreviation normalization
- **Character n-gram blocking (backstop):** trigram index on
  name+address concatenation, for cases the above miss (typos, reordering)

Log recall ceiling (against training ground truth) and reduction ratio on
every change — this is the number that upper-bounds leaderboard score, and
it's exactly what `candidate_pairs.tsv` gets audited on. Prefer generous
recall here; precision is the matcher's job.

## 3. Feature engineering (per S1–candidate pair)
- **Name:** Jaro-Winkler, normalized Levenshtein, token Jaccard, TF-IDF
  cosine (char n-gram, to be robust to transliteration), abbreviation-
  normalized exact-match flag
- **Address:** token-set overlap, component-position similarity (street
  number, street name, locality) where components are parseable, fallback
  whole-string similarity where they're not (handles missing-PIN, landmark-
  based addresses)
- **Structural:** country match flag (never used to *filter*, only as a
  feature — must not become a de facto allowlist), name-length ratio,
  token-count difference

## 3.5 Training-pair sampling (scale-driven, not optional)
Even after blocking, the full candidate set is far too large and far too
imbalanced (true matches are a tiny fraction) to train a classifier on
directly. Sample negatives **per S1 entity, from its own candidate bucket**
(hard negatives — things blocking thought were plausible but aren't true
matches) rather than uniformly at random from the whole pool — uniform
random negatives are near-zero-similarity and teach the model nothing
useful about the actual decision boundary it has to draw.

## 4. Matching model
Two viable model shapes, both trivially satisfy the ≤8B/MIT-Apache
constraint:
- **Option A (recommended default): gradient-boosted trees** (LightGBM or
  CatBoost, both open-license) on the engineered feature vector. Fast to
  train/iterate within a 3-day window, highly interpretable (feature
  importances → good methodology-doc material), no licensing risk at all.
- **Option B: small open sentence-embedding model** (e.g. an
  MIT/Apache-licensed sentence-transformer in the tens-of-millions of
  params, nowhere near the 8B cap) to produce a semantic similarity feature
  for name/address, fed into the same GBT classifier as an *additional*
  feature rather than replacing the feature-engineering approach. Use this
  only if pure string-similarity features plateau — added complexity has to
  earn its keep given the tight timeline.

Either way: no LLM prompting-based matching. It's unnecessary complexity
against a 2:1 precision-weighted metric where calibrated probability
thresholds on structured features are easier to control and audit.

## 5. Thresholding & clustering
- Calibrate the probability threshold on the **held-out validation split**
  by directly optimizing F₀.5 (not accuracy, not F1) — sweep thresholds,
  pick the max.
- Enforce one-to-many correctly: an S1 entity can match multiple S2/S3
  records; but reject inconsistent structures (e.g. don't let clustering
  logic accidentally merge two candidates into each other).
- Default to *no match* on ambiguous low-confidence pairs — singletons are
  worth full credit and cost nothing to protect; F₀.5's precision weighting
  means an uncertain guess is a bad trade.

## 6. Evaluation harness
- Build the F₀.5 macro-average scorer exactly per spec (per-entity, then
  averaged, singletons scoring 1.0/0.0) before any modeling starts — it's
  the only feedback loop until leaderboard submissions.
- Implement it as a **vectorized groupby-aggregate** over predicted vs.
  true match sets, not a Python `for` loop over millions of S1 rows — at
  this scale a naive loop-based scorer becomes the bottleneck in every
  iteration cycle.
- Stress-test generalization: **hold out one training country entirely**
  during a dev pass (e.g. train only on US, validate on India-only) to
  simulate the real France situation, before trusting that the pipeline
  generalizes.

## 7. Submission generation
- `candidate_pairs.tsv` written from the literal candidate set passed into
  the final matcher's inference call — not an earlier/looser blocking pass.
- `matching_results.tsv` is a subset of that by construction — enforce this
  with an assertion, not just a hope (the real validator only *warns* on
  this, it doesn't fail — so our own assertion is the actual gate).
- Every write goes through `utils/validate_submission.py` before upload.
  Note its `--check-ids` existence-check is off by default (memory cost on
  the ~1.7M-ID test set) — run our own lightweight ID-existence check
  in `src/submit/` regardless, since the validator will happily `PASS` a
  file containing nonexistent IDs that then just scores worse.

## Reproducibility
`code/business_entity_resolution/README.md` must let anyone regenerate both
output files end-to-end from raw data using only what's in `src/` +
`requirements.txt` — this is explicitly what gets audited for top teams.
