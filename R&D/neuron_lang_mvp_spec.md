# NeuronLang — MVP Spec (2‑page)

*Status: MVP — implementable prototype for prototyping safe, introspective AIs.*

## 1. Overview

NeuronLang is a minimal domain-specific language + runtime that provides: deterministic replay for audited decisions, auditable and provenance-rich memory, strictly gated self-modification, safe multi-agent knowledge merging, and human-in-the-loop safety gates. The MVP focuses on primitives and runtime guarantees that make safety the path of least resistance.

## 2. Core runtime guarantees

1. **Deterministic replay** — any run marked `auditable` can be reproduced exactly given saved seed(s), inputs, and environment snapshot.
2. **Provenance & signatures** — every world-facing change (memory export, network send, self-mod commit) includes an immutable provenance record signed by a keypair.
3. **Capability-based access control** — every operation that affects external state or shared agents requires an explicit capability token.
4. **Three-phase self-mod lifecycle** — `propose -> canary -> commit` (with automated tests + optional human signatures).
5. **Append-only audit logs** — tamper-evident append-only logs (WAL) persisted outside of volatile memory.

## 3. High-level primitives

- `memory` — read/write with metadata, versioning, and CRDT merge support.
- `selfmod` — propose/canary/commit/rollback lifecycle for code or weight changes.
- `introspect` — generate structured decision traces and thought-inspection objects.
- `capabilities` — request, verify, and enforce rights for sensitive operations.
- `resource` — query and enforce compute/time budgets and QoS.
- `comms` — labeled channels (public, audited, private) with rate limits and provenance.
- `lifecycle` — hooks for checkpointing, shutdown, rehearsal (sleep/consolidation).

## 4. Minimal semantics (what each primitive must guarantee)

- **memory.put(key, value, meta)**
  - Atomic; appends a new version with timestamp, source, confidence, importance.
  - Must be signed if source is external or shared.
- **memory.merge(a,b,rule)**
  - Deterministic merge functions (CRDTs preferred). Support `most_recent`, `highest_confidence`, and custom merge callbacks.
- **selfmod.propose(patch, tests)**
  - Produces an immutable proposal object. Proposal includes patch diff, tests, provenance, and resource estimate.
- **selfmod.run\_canary(proposal, env)**
  - Runs proposal in an isolated sandbox with identical seeds and controlled inputs; results are recorded and compared.
- **selfmod.commit(proposal, signatures)**
  - Only allowed if policy engine approves (tests pass, signatures present, budget available). Commits produce new signed version and audit entry.
- **introspect.why(action\_id)**
  - Returns structured trace: inputs, sub-decisions, confidences, heuristics used, relevant memory references.

## 5. Runtime components (implementation hints)

- **Language front-end**: small DSL parser or JSON DSL for the first prototype.
- **Memory Store**: relational DB (SQLite/Postgres) with append-only changelog table and indexing for fast reads.
- **Provenance & Crypto**: Ed25519 signatures for provenance; keys stored in an HSM or secure key store for production (file-based dev keys OK for MVP).
- **Sandboxing**: Docker containers or OS-level processes with strict network and filesystem policies for canary runs.
- **Policy Engine**: rules expressed as JSON/YAML policies; evaluate requirements like tests, required signatories, and resource caps.
- **Replay Engine**: record RNG seeds, input streams, environment variables, and deterministic sources. Provide a `replay(run_id)` command.

## 6. Example APIs (mini-syntax)

- Memory write:

```neuron
memory.put("alice:todo:pay_rent", "2025-08-01", meta={source:"user", importance:0.95, confidence:0.9})
```

- Propose & commit:

```neuron
proposal = selfmod.propose(patch="tune_layer(7, lr=1e-5)", tests=["safety_check","perf_test"])
selfmod.run_canary(proposal, env="sandbox")
selfmod.commit(proposal, require_signatures=["ops_lead"])
```

- Introspect:

```neuron
trace = introspect.why("send-email-123")
print(trace.steps, trace.confidences)
```

## 7. Test suite (priority list)

1. **Rollback test** — baseline -> commit -> behavior changes -> rollback -> behavior equals baseline.
2. **Merge determinism test** — create conflicting updates; assert deterministic merges for each merge rule.
3. **Capability enforcement test** — unauthorized op must be rejected and logged.
4. **Reproducibility test** — auditable run replay yields identical outputs.
5. **Sandbox isolation test** — canary runs cannot access host network/files.

## 8. Evaluation metrics

- **Safety pass rate** (unit tests passing for safety-critical flows).
- **Determinism score** (fraction of audited runs that replay exactly).
- **Merge correctness** (test suite coverage for CRDT/merge rules).
- **Explainability score** (human-rated usefulness of `introspect.why` outputs).
- **Red-team resilience** (fraction of red-team attack vectors mitigated).

## 9. Minimal deployment checklist

- Signed baseline artifacts and keys in secure storage.
- Immutable backup of audit logs.
- Human approver workflow for `selfmod.commit` requiring at least one human signature for public or network-facing changes.
- Canary environments with no external network by default.

## 10. Next steps (first 7 days)

1. Prototype memory store + provenance layer + `memory.put/get/merge` (Day 1–2).
2. Implement simple `introspect.why` that logs structured traces for a toy decision (Day 2–3).
3. Build `selfmod.propose` + local sandbox runner + `selfmod.commit` policy check (Day 4–6).
4. Run core tests (rollback, merge determinism, capability enforcement) and run a red-team query against policies (Day 6–7).

---

*End of MVP Spec — drop into **`docs/spec.md`** in the repo.*

