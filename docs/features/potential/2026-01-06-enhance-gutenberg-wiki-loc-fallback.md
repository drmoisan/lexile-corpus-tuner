# enhance-gutenberg-wiki-loc-fallback (Potential)

- Date captured: 2026-01-06
- Author: Dan Moisan
- Status: Draft

## Problem / Why

The original publication year enrichment pipeline uses Open Library as the source of truth.
Some Gutenberg titles have noisy metadata or lack Open Library coverage, resulting in `original_pub_year=null`.

This potential feature adds optional fallback lookups (Wikidata and/or the Library of Congress) so enrichment can improve coverage while keeping provenance and confidence explicit.

## Proposed Behavior

When enabled, attempt secondary lookups for publication year if Open Library yields no acceptable match.

High-level behavior:
- Primary source remains Open Library Search (`https://openlibrary.org/search.json`) using title + author matching.
- If and only if the primary lookup returns `pub_year_confidence=none`, optionally attempt:
	- Wikidata lookup (when `--enable-wikidata` is set)
	- LOC lookup (when `--enable-loc` is set)
- Fallback lookups should only run when an appropriate identifier is available for the row (e.g., ISBN/LCCN), to reduce false positives.
- Enrichment outputs remain:
	- `original_pub_year` (int or null)
	- `pub_year_confidence` (`high`/`low`/`none`)
	- `original_pub_source` (e.g., `openlibrary`, `wikidata`, `loc`)
- `issued_date` remains untouched.

## Acceptance Criteria (early draft)

- [ ] When `--enable-wikidata` is set and the row has a usable identifier, the pipeline attempts Wikidata lookup after Open Library yields no acceptable match.
- [ ] When `--enable-loc` is set and the row has a usable identifier, the pipeline attempts LOC lookup after Open Library yields no acceptable match.
- [ ] Fallbacks never run when identifiers are missing (to reduce false positives).
- [ ] The selected fallback year is written to `original_pub_year` and provenance is recorded in `original_pub_source`.
- [ ] `pub_year_confidence` remains explicit; fallback matches should be conservative (typically `low` unless a stronger rule exists).
- [ ] CLI flags are documented and have deterministic behavior.

## Constraints & Risks

List notable constraints (performance, compatibility, scope) or risks.

- Identifier availability: Gutenberg metadata may not reliably include ISBN/LCCN; a clear identifier strategy is required.
- API rate limits and reliability: Wikidata/LOC endpoints may have different quotas and failure modes than Open Library.
- False positives: fallback matching must remain conservative and auditable via confidence + provenance.
- Complexity creep: avoid turning enrichment into a general bibliographic reconciliation system.

## Test Conditions to Consider

- [ ] Unit coverage areas
- [ ] Integration scenarios
- [ ] CLI/API examples

Suggested test conditions:
- Unit: identifier extraction / gating behavior.
- Unit: Wikidata response parsing into a year.
- Unit: LOC response parsing into a year.
- Integration (mocked): when Open Library yields none and an identifier is present, fallback runs and writes `original_pub_source` correctly.
- Integration (mocked): when identifiers are missing, fallback is skipped even when enabled.
- CLI: flags parse and toggle behavior deterministically.

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/enhance-gutenberg-wiki-loc-fallback/` folder from the template

