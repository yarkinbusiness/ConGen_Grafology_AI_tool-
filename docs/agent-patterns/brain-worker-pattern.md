# Fable 5 (brain) / Sonnet 5 (worker) pattern — two ways to build it

There are two different things that fit "frontier model coordinates, cheap
model executes," depending on what you're building. Pick the one that
matches your situation — they're not the same mechanism.

---

## 1. Inside a running app: Claude Managed Agents (CMA) multiagent roster

Use this when the brain/worker split needs to run **as part of your
product**, at request time, potentially long-running or scheduled.

Source pattern: Anthropic's `claude-cookbooks` repo, `managed_agents/`
directory, specifically `CMA_plan_big_execute_small.ipynb` ("Plan big,
execute small"). If the `claude-cookbook-patterns` skill is installed, the
condensed reference is at
`~/.claude/skills/claude-cookbook-patterns/references/managed-agents.md`.

**The mechanism**: a coordinator agent gets a `multiagent` field:
```json
{
  "multiagent": {
    "type": "coordinator",
    "agents": [
      {"type": "agent", "id": "agent_worker1_id", "version": 3},
      {"type": "agent", "id": "agent_worker2_id", "version": 1}
    ]
  }
}
```
That field alone is what makes it a coordinator — the server auto-grants it
`create_agent` / `send_to_agent` / `wait_for_agents` / `list_agents`, and
auto-grants each worker `submit_result` / `send_to_parent`. You never define
those tools yourself.

**Two sharp edges, straight from the cookbook notebook:**
- The roster is **snapshotted** at agent create/update time. Bump a worker's
  version later and the coordinator won't see it until you re-provision the
  coordinator too.
- The coordinator **cannot see** its workers' prompts, names, or
  descriptions — only whatever its own system prompt says about them. Keep
  that description in sync by hand; nothing on the server enforces it.

**Working example** (a small-talk coaching app: a frontier coordinator
grades a conversation transcript by fanning out to 4 cheap specialist
graders, then synthesizes one report).

`config.py` — the model tiering:
```python
# coach_coordinator is the "brain" — frontier model, does the actual
# synthesis judgment. The 4 graders are cheap/fast workers scoped to one
# narrow grading dimension each.
COORDINATOR_MODEL = os.environ.get("SMALLTALK_COORDINATOR_MODEL", "claude-fable-5")
WORKER_MODEL = os.environ.get("SMALLTALK_WORKER_MODEL", "claude-sonnet-5")
```

`agents_setup.py` — provisioning the roster (idempotent: creates on first
run, bumps the version on later runs if the prompt text changed):
```python
def _ensure_agent(client, state, key, *, name, model, system, multiagent=None):
    """Create the agent if unknown, or update (new version) if its prompt
    changed since last run. Returns the state entry {id, version, hash}."""
    entry = state.get(key)
    prompt_hash = _hash(system + (str(multiagent) if multiagent else ""))

    if entry is None:
        kwargs = {"name": name, "model": model, "system": system}
        if multiagent:
            kwargs["multiagent"] = multiagent
        agent = client.beta.agents.create(**kwargs)
        entry = {"id": agent.id, "version": agent.version, "hash": prompt_hash, "model": model}
    elif entry.get("hash") != prompt_hash:
        kwargs = {"system": system}
        if multiagent:
            kwargs["multiagent"] = multiagent
        agent = client.beta.agents.update(entry["id"], **kwargs)
        entry = {"id": agent.id, "version": agent.version, "hash": prompt_hash, "model": model}
    state[key] = entry
    return entry


def provision(client, state):
    worker_entries = []
    for key, name, system in worker_specs:  # 4 specialist grader prompts
        entry = _ensure_agent(client, state, key, name=name, model=WORKER_MODEL, system=system)
        worker_entries.append(entry)

    roster = [{"type": "agent", "id": e["id"], "version": e["version"]} for e in worker_entries]
    _ensure_agent(
        client, state, "coach_coordinator",
        name=COORDINATOR_NAME, model=COORDINATOR_MODEL, system=COORDINATOR_SYSTEM,
        multiagent={"type": "coordinator", "agents": roster},
    )
```

The coordinator's system prompt is what tells it what its workers do (since
it can't see their prompts directly):
```
Your job: create_agent for each of the four workers with the full transcript
and a one-line task description of their dimension, wait_for_agents, then
synthesize — do not just concatenate their output. Output must be valid JSON...
```

Then at session time you run the coordinator as a real CMA session
(`client.beta.sessions.create(agent={"type": "agent", "id": ..., "version": ...}, environment_id=...)`,
stream events until `session.status_idle`) — the workers execute inside that
session automatically once the coordinator calls `create_agent`.

**When this is the right tool**: the brain/worker split needs to happen
*inside the product*, potentially unattended/scheduled, and you want
Anthropic's infrastructure (sandboxing, session persistence, versioning) to
own the execution. Costs a session/sandbox provision per run — don't reach
for it for something that needs to feel instant to an end user.

---

## 2. Inside a Claude Code session: worker/brain dev-loop via the `Agent` tool

Use this when the brain/worker split is about **how work gets done in an
agentic coding session itself** — e.g. "have a strong model plan and review,
have a cheaper model implement" — not something the shipped app does at
runtime. **This is the default mode for building this project.**

**The mechanism**: Claude Code's `Agent` tool takes a `model` parameter
(`sonnet` / `opus` / `haiku` / `fable`) and a `subagent_type`. There's no
special API for this — it's just: spawn one subagent per role, in sequence,
and don't let the next step start until the previous one is graded.

The loop, concretely, run once per task:

1. **Worker** — spawn `Agent` with `model: "sonnet"`, `subagent_type:
   "general-purpose"` (needs Read/Write/Edit/Bash). Prompt = the task's full
   build spec + the exact acceptance bar it'll be graded against + enough
   project context to be self-contained (a fresh agent has no memory of
   anything said earlier in the conversation). Ask it to report back exactly
   what it did and how, with evidence (command output), not just a claim.

2. **Brain** — spawn a *separate* `Agent` with `model: "fable"` (a
   read-only-tooled subagent type like `code-reviewer`, or `general-purpose`
   restricted by instruction). Give it the **same acceptance bar**, the task
   description, and explicit instructions to verify independently — re-run
   the tests itself, re-read the diff itself — rather than trust the
   worker's self-report. Do **not** show it the worker's report; that's what
   keeps it an independent check instead of a rubber stamp.

3. **Gate**: if the brain says PASS, mark the task done and move to the next
   task's worker. If FAIL, spawn a new worker with the brain's specific,
   evidenced feedback and re-review. Repeat until PASS.

This is the same shape as CMA's "Gate" pattern (human-in-the-loop escalation)
and "Verify with outcome grader" pattern conceptually, just running via
Claude Code's own subagent tool instead of the Managed Agents API — no
sessions, no sandboxes, no versioned agents. It's the right choice when the
loop is for *building software in an agentic coding tool*, and the CMA
approach (section 1) is the right choice when the loop needs to be part of
the *shipped product*.

**Concretely, what a worker-spawn prompt looks like** (trimmed example):
```
You are a worker agent implementing one task in <project>, at <path>.
Read <specific files> before changing anything.

## Your task
Build: <precise spec — files, behavior, what NOT to touch>
Accept: <the exact bar you'll be graded against, verbatim from the plan>
Verifiable now: <yes/no and why, if something is environmentally blocked>

## What to report back
Exactly what you changed and why, command output as evidence, explicit
confirmation of what you did NOT touch, and any issues you noticed but
were out of scope to fix.
```

**Concretely, what a brain-review prompt looks like** (trimmed example):
```
You are "the brain" — an independent reviewer grading a worker's completed
task in <project>, at <path>. Do NOT take their word for anything — verify
by reading the actual diff and running tests/checks yourself.

## The task
<same build spec>

## Acceptance bar — check every one of these yourself
<same bar, itemized>

## What to output
A clear verdict: PASS or FAIL. If FAIL, exact specific problems with
evidence. Keep it tight — this is a grading pass, not an essay.
```

That's really the whole trick: same acceptance bar given to both, worker
never sees the grading criteria phrased as "you will be checked on X" being
optional, and the brain is instructed to distrust and re-verify rather than
summarize the worker's claims.
