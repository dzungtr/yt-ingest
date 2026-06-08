# ADR 0004: CLI-only architectural boundary

## Status

**Accepted**

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

`yt-ingest` is a **CLI-only tool**. It is explicitly not:

- a web UI or web application,
- a REST or gRPC API,
- a Docker image or Compose stack,
- a multi-user service,
- a cloud deployable (no Terraform, no Helm, no Kubernetes manifests).

The boundary is stated as a negative list so that proposals that look like
"small extensions" (e.g. "just add a FastAPI wrapper", "just ship a Docker
image so I can run it on the server") are clearly out of scope and need to be
defended as scope changes rather than slipped in.

Three invocation modes are supported within this boundary: batch, single-URL,
and agent (see ADR-0007). All state lives on the local filesystem.

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
