# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Development pattern: brain/worker

All non-trivial implementation work in this repo follows the brain/worker
pattern documented in full at
`docs/agent-patterns/brain-worker-pattern.md`. Read that file before
planning multi-step or multi-task implementation work. Summary:

- **Building the app itself, at runtime, as a product feature** (e.g. an
  in-app agent that plans with a frontier model and fans work out to cheap
  specialist agents): use the Claude Managed Agents (CMA) `multiagent`
  coordinator pattern — see section 1 of the reference doc.
- **Building this codebase, inside a Claude Code session** (the normal case
  — implementing a feature, fixing a bug, running a multi-task plan): use
  the worker/brain dev-loop via the `Agent` tool — see section 2 of the
  reference doc. Concretely, for each task:
  1. Spawn a **worker** (`model: "sonnet"`, `subagent_type:
     "general-purpose"`) with the full build spec, the exact acceptance bar,
     and enough context to be self-contained. It reports back what it did
     with evidence (command output), not just a claim.
  2. Spawn a separate **brain** (`model: "fable"`, a read-only reviewer) with
     the *same* acceptance bar and task description, instructed to verify
     independently (re-run tests, re-read the diff) rather than trust the
     worker's report. Never show it the worker's self-report.
  3. **Gate** on the brain's verdict: PASS moves to the next task; FAIL
     re-spawns a worker with the brain's specific, evidenced feedback and
     repeats until PASS.
  - Skip this loop for small, low-risk, single-file edits — it's for
    genuine implementation tasks, not every keystroke.

## Project status

This repository is newly initialized; no application code exists yet.
