# ADR 0000: Record Architecture Decisions

## Status

**Accepted**

## Context

`yt-ingest` is growing from a one-off script into a maintained CLI tool for
personal video-to-notes use. Without an explicit record of the design decisions
made along the way, future readers (and future me) will have to reverse-engineer
why each non-obvious choice exists. Design rationale currently lives in the
original superpowers specs, which are implementation-plan-oriented rather than
decision-oriented, and which mix aspiration with what was actually shipped. The
specs are also scheduled for deletion, so the decisions need a new home.

A lightweight decision log makes it possible to answer **why** without reading
the full superpowers history, and makes it possible to supersede old decisions
cleanly when the tool's context changes (e.g. a better LLM provider, a new
transcript source).

## Decision

Adopt the **Michael Nygard ADR format** for architectural decisions that are:

1. **Hard to reverse** — e.g. choosing a transcript fetcher abstraction, picking
   an LLM provider, or drawing an architectural boundary.
2. **Surprising without context** — decisions where the obvious alternative
   would have been chosen by a new reader who doesn't know the constraint that
   forced the current shape.
3. **Result of a real trade-off** — rejected alternatives are worth explaining.

Do not write ADRs for trivial choices (library versions, one-off bugfixes,
cosmetic changes).

### File format

- One file per decision in `docs/adr/`.
- Filename pattern: `NNNN-kebab-slug.md` (e.g. `0001-transcript-fetching-chain-of-responsibility.md`).
- Numbering is strict, sequential, 4-digit, no gaps (`0000`, `0001`, `0002`, ...).
- Structure inside each file: **Title / Status / Context / Decision / Consequences**.

### Status lifecycle

A newly written ADR starts as `proposed` if the decision is not yet applied, or
`accepted` if it documents what is already in the code. Status transitions:

- `accepted` → `deprecated` when the decision is no longer current but nothing
  has replaced it (e.g. a removed feature).
- `accepted` → `superseded` when a newer ADR replaces it. The superseded ADR
  gains a `Superseded by` link at the top of Context; the new ADR links back
  with `Supersedes`.

Other statuses are not used.

### Cross-reference convention

- Inline prose: `ADR-00NN` (short form, e.g. "the model choice described in
  ADR-0002").
- When a reader needs to go look up the file: `docs/adr/00NN-kebab-slug.md`.
- When an ADR relates to non-decision documentation (current state, schemas,
  diagrams), point at `docs/architecture.md` as the canonical home, rather
  than duplicating detail inside the ADR.

### Non-decision documentation

Current project state — what things **are** today, not why they were chosen —
lives in `docs/architecture.md`. The ADR set explains the historical fork
points; `architecture.md` describes the resulting shape.

## Consequences

- Every future "non-obvious" design change needs an ADR. The bar above (hard to
  reverse, surprising, real trade-off) filters out noise.
- ADRs are a permanent record. Even deprecated or superseded ADRs remain in the
  `docs/adr/` directory so the rationale survives.
- `docs/architecture.md` is the live document; ADRs are the fossil record.
- Superpowers-style spec files are no longer maintained for `yt-ingest` and the
  previous `docs/superpowers/` directory has been removed.
