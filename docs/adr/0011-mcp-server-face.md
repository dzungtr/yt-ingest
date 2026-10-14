# ADR 0011: MCP server face

## Status

**Proposed** — implementation tracked in issues #27–#30 (spec #26).

**Supersedes ADR-0004** (CLI-only architectural boundary). ADR-0007's three
invocation modes are unchanged; this decision adds a fourth, server-shaped
face on top of them.

## Context

ADR-0004 drew a hard boundary: `yt-ingest` is a CLI-only tool, explicitly
not an API, not a container, not a deployable. The boundary served its
purpose — it kept the tool easy to throw away and prevented silent scope
growth. But the context changed: the owner now runs an agent harness whose
agents live on multiple tailnet machines, and the homelab already hosts two
MCP servers (graphiti-mcp, signoz-mcp) behind tailscale ingresses with the
tailnet ACL as the perimeter. The pressure is not "expose yt-ingest to the
world" — it is "let any agent on my tailnet produce a study note from a URL
without shelling out to a local CLI install and per-host API keys."

ADR-0004's own consequences section anticipated this: "Sharing with others
requires a fresh decision." This is that decision, made explicitly rather
than eroded silently.

## Decision

`yt-ingest` gains a **server face**: a single-tool MCP server
(`write_study_note(url) -> note text`), deployed as a stateless container on
the homelab cluster, reachable over the tailnet. The boundary narrows, it
does not dissolve. The tool is still explicitly **not**:

- a web UI or multi-user product,
- a REST API outside the MCP protocol,
- stateful — no PVC, no session persistence, no disk writes server-side,
- a retrieval surface — no index/ask tools; the FAISS stack stays CLI-only,
- a transcription service — no whisper in the container; caption-less videos
  return a clean error pointing at the CLI.

The CLI remains the **primary product**. The server wraps the same
in-memory extraction path one function deep (fetch chain →
`extract_from_cache` → `render_note`) and adds no pipeline logic; note
quality tracks `main` automatically. The server face rides the packaging
extras split (ADR-worthy dependency surgery in its own right): base deps
cover fetch+extract; `server`, `cli`, `whisper`, `retrieval`, and `full`
extras keep each face's install honest.

Transport is **streamable-HTTP** (the MCP spec's production remote
transport since the 2025-03-26 revision; SSE-only deprecated). Progress
notifications at each pipeline stage keep long calls alive under clients
with per-request timeouts (pi resets its 60s default on each notification),
so callers need zero client config.

## Consequences

- **The negative list is shorter but still enforced.** The escapes from
  ADR-0004 are enumerated above; anything outside them still needs a fresh
  decision.
- **Deployment ownership lands in the homelab repo.** The k8s manifest
  lives in `dzungtr/homelab` under the harness6 ArgoCD app, mirroring the
  graphiti-mcp precedent; the image is built and published manually
  (podman) to `ghcr.io/dzungtr/yt-ingest:<tag>`, and the manifest pins a
  tag — cluster state always traces to an immutable artifact.
- **LLM wiring stays provider-abstract (ADR-0008).** The manifest defaults
  `OPENAI_BASE_URL` to the in-cluster aperture model gateway; switching
  endpoints is env config, never code.
- **Long calls are a feature, not a bug.** A full extraction takes minutes;
  the progress-notification contract is load-bearing for every client.
- **The trust perimeter is the tailnet ACL.** No app auth on the ingress,
  matching both live MCP servers; tags `tag:k8s,tag:family,tag:mcp`.
