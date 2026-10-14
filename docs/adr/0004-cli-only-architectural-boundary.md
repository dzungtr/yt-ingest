# ADR 0004: CLI-only architectural boundary

## Status

**Superseded** by [ADR-0011](0011-mcp-server-face.md) (MCP server face).
The CLI remains the primary product; the supersession adds a bounded
server face, it does not dissolve the boundary discipline.

## Context

The tool's user is one person, running it on their own workstation, against
their own list of YouTube URLs. Every other "use case" — exposing it to other
people via HTTP, running it as a background service, shipping it as a Docker
image, deploying it as a cloud function, wiring up multi-user auth — expands
the surface area into things the project owner has no current need for, and
each would pull in a fresh set of operational obligations (TLS, auth, session
lifecycle, container image maintenance, CI for a service runtime).

A CLI tool that stays a CLI tool is easy to throw away and rebuild. A CLI tool
that quietly became a web service has taken on responsibilities its owner
did not consent to.

## Decision

`yt-ingest` is a **CLI-first tool**. Within this ADR's original frame it
was explicitly not a web UI, not a REST API, not a multi-user service, and
not a cloud deployable. ADR-0011 supersedes the container/deployment
entries of this negative list (single-tool MCP server, stateless container,
tailnet perimeter); the remaining entries — no web UI, no multi-user
product, no REST outside MCP, no retrieval or transcription surface in the
container — still stand and require fresh decisions to escape.

Three invocation modes were supported within this boundary: batch,
single-URL, and agent (see ADR-0007). ADR-0011 adds a fourth,
server-shaped face while leaving those three untouched.

## Consequences

- **Scope discipline.** Proposals that require a server, a reverse proxy, or
  shared state between users are out of scope by default and have to be made
  explicitly.
- **The project stays easy to throw away.** A single `pip install -e .` gets a
  working tool. Nothing has to be orchestrated.
- **Sharing with others requires a fresh decision.** If the owner later wants
  to expose the pipeline to other users, it is a new design — not a silent
  extension of this one.
- **Agent integration is supported via the `--agent` flag** rather than by
  exposing an HTTP endpoint. The calling agent shells out to the CLI.
