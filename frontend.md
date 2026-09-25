# frontend.md — Demo/Results Dashboard

## Is a frontend actually in scope?
**Not for scoring.** The leaderboard only looks at `matching_results.tsv`.
But per AWS Builder Center's own description of the event, the format
"Includes a 72-hour intensive ML hackathon **followed by a Grand Finale
where top teams present their solutions to Amazon Scientists**." A
polished internal dashboard is what turns "here's our F₀.5 number" into a
credible, inspectable presentation — that's the actual differentiator here,
not a scored requirement. Build it *after* the pipeline is solid, never
before.

## Scope
- Read-only, static/local. Reads the team's own output files
  (`matching_results.tsv`, `candidate_pairs.tsv`, validation-split scores) —
  no live inference, no external calls (same no-external-lookup constraint
  applies to anything the dashboard touches).
- Three views:
  1. **Scorecard** — validation F₀.5 (macro), precision/recall, singleton
     accuracy, broken out by country (US / India / synthetic-holdout for
     "unseen country" stress test)
  2. **Blocking funnel** — cross-product size → candidates generated →
     final matches, with recall ceiling annotated at each stage
  3. **Entity match explorer** — pick an S1 record, see its candidates side
     by side with similarity scores and the final decision; include a
     filtered "worst false merges" and "worst missed matches" gallery for
     error analysis during the presentation
- Build as a single self-contained HTML file (or Streamlit if the team
  prefers Python-native) — no deployment needed, runs from a laptop at the
  Grand Finale.

## Design system

### Color palette
Grounded in current (2026) "Professional / B2B" dashboard palette trends —
an indigo + emerald system reads as trustworthy/analytical and, usefully,
maps directly onto the domain: emerald for correct/confirmed matches,
amber/red reserved specifically for false merges (the thing F₀.5 punishes
hardest), so the color language itself reinforces the precision story.

| Role | Color | Hex |
|---|---|---|
| Primary (brand/nav/headers) | Indigo 600 | `#4F46E5` |
| Primary dark (text on light, dark-mode surface) | Slate 900 | `#0F172A` |
| Secondary / authority accents | Slate 600 | `#475569` |
| Success / correct match | Emerald 600 | `#059669` |
| Warning / borderline confidence | Amber 500 | `#F59E0B` |
| Error / false merge | Red 500 | `#EF4444` |
| Background | Slate 50 | `#F8FAFC` |
| Surface / cards | White | `#FFFFFF` |
| Borders/dividers | Slate 200 | `#E2E8F0` |

Dark mode: swap background/surface to `#0F172A` / `#1E293B`, keep the
indigo/emerald/amber/red accents unchanged (they hold contrast on dark).

### Type
| Role | Font | Notes |
|---|---|---|
| UI / body / labels | **Inter** | current default for SaaS & analytics dashboards; excellent at small sizes, wide weight range |
| Headlines / section titles / stat callouts | **Space Grotesk** | geometric sans with just enough personality for a "product," pairs cleanly with Inter without competing |
| Data / IDs / code / TSV values | **JetBrains Mono** | tabular figures, disambiguates `S1-`/`S2-`/`S3-` IDs at a glance, standard for anything showing raw data values |

Load via Google Fonts (Inter, Space Grotesk) + JetBrains Mono (Google Fonts
also serves it). All three are open-license, consistent with the challenge's
own model-licensing spirit.

### Layout notes
- Numbers-first: every view leads with the metric, not the chrome.
- Use monospace exclusively for entity IDs and raw field values — never for
  prose — so the eye immediately distinguishes "data" from "explanation."
- Keep it boring in the best sense: no gratuitous animation. The wow factor
  here is *clarity under scrutiny* — Amazon Scientists will be looking for
  rigor, not flash.
