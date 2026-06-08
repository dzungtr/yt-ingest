# ADR 0005: Study note as primary output contract

## Status

**Accepted**

## Context

The per-video LLM pass could produce several different kinds of output
depending on who the intended reader is: a blog post polished for publication,
a RAG-shaped chunk set optimised for retrieval, a transcript summary for an
agent to consume further, or dense study notes for a single reader to skim.
Each target shape implies different prompt structure, length, tone, and which
content to preserve or drop.

Mixing these use cases in one prompt leads either to a compromise output that
satisfies none of them, or to a menu of output modes in the CLI that the
owner does not currently want to maintain.

## Decision

The primary and only supported use case for the per-video LLM output is
**personal study notes**. A reader opens `notes/<video_id>.md` to absorb the
video's content faster than watching it. The notes are:

- **Skimmable and structured for later review**, rather than polished for
  publication.
- **Not optimised for RAG retrieval as the primary consumer.** The FAISS
  index reads the notes after the fact, but the note's shape is chosen for
  a human reader, not for an embedding model.
- **Not a second pass by an external agent.** When a coding agent rather
  than a human wants the transcript, it uses `--agent` mode (ADR-0007) and
  gets raw transcript text, bypassing the note entirely.

Concretely the notes have:

- A **short TL;DR at the top** (3-7 lines after the merge stage).
- A **free body** with section headings drawn from the actual content.
- **Third-person, neutral voice** — the speaker is referred to as "the
  speaker" or by name; no first-person narration.
- **Prose default**; bullets and numbered lists only where they genuinely aid
  scanning (enumerations, steps, comparisons).
- A **soft density target** of roughly 20-30% of the input transcript's
  word count.

### Strip / preserve rules

**Always keep:**
- Concrete numbers, dates, named tools, people, companies, products.
- Analogies and metaphors used to explain concepts.
- Caveats, counterarguments, stated limitations.
- Step-by-step procedures and how-tos.
- The speaker's opinions, recommendations, and predictions.

**Keep only if load-bearing (it carries the argument, not just illustrates it):**
- Specific examples and case studies.
- Verbatim quotes.

**Always drop:**
- Personal anecdotes from the speaker.
- Sponsor reads, channel plugs, intros, outros, calls to subscribe.
- Filler, verbal tics, hedging, hype language, dramatic build-up.
- Repetition of the same point.

These rules live in the extraction prompt (see `src/yt_ingest/extract.py`)
rather than in configuration, because changing them is infrequent and the
prompt is where they have to be to take effect.

## Consequences

- **The reader gets a consistent shape** regardless of video topic.
- **The "blog post" framing is explicitly rejected.** If a user later wants
  publication-polish output, that is a separate decision requiring separate
  prompts.
- **The note template (`src/yt_ingest/templates/note.md.j2`) is a Jinja
  envelope for model output**, not a rigid 7-section structure. The original
  spec envisaged fixed body sections (Summary / Key claims / Frameworks /
  Definitions / Worth rewatching / Counterpoints / Open questions) — that
  structure is not realised in the current template, which renders title,
  channel, duration, video id, URL, and then the model's free body.
- **Clickable YouTube deep-link timestamps** (the `[mm:ss](https://youtu.be/…) `
  pattern mentioned in the original spec) are **not rendered** by the
  current template; the body is whatever the model produced.
- **Prompt changes require prompt work, not config work.** There are no CLI
  flags or config knobs for these rules.
