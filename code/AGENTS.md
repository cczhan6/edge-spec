# Repository instructions

## Research scope

- This repository models decode-stage execution only.
- Do not restore prefill computation, prompt transmission, TTFT, or prefill metrics.
- The prompt remains available only as semantic context.
- Do not modify the proposed method during baseline reconstruction.

## Baseline authority

- `docs/baseline_contract.md` is the semantic authority.
- Official SpecEdge repository behavior takes precedence for
  Server-only-Tree and SpecEdge-Tree.
- Linear variants must retain the corresponding deployment and
  scheduling semantics while replacing tree candidates with linear drafts.
- DiP-SD is a paper-based reimplementation.

## Development rules

- Work milestone by milestone.
- Run relevant tests after every meaningful change.
- Never continue to the next milestone while current milestone tests fail.
- Do not silently weaken a baseline to make tests pass.
- Do not use future acceptance outcomes in scheduling.
- Preserve greedy output equivalence with Target-only.
- Keep legacy implementations until replacement tests pass.
- Commit each completed milestone separately.

## Required validation

Run:

```bash
pytest -q