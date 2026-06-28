# Real-Model Greedy Equivalence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every baseline consume one shared, on-demand exact-prefix target-next-token semantic so real-model outputs remain identical to Target-only across verification modes and batch shapes.

**Architecture:** `HuggingFaceModelRunner.target_next_token(prefix)` performs one target forward for an exact prefix and memoizes only that exact tuple-to-token result. Target-only extends its prefix one token at a time; linear verification queries until rejection or the post-acceptance bonus; tree verification walks only the selected path. Batch APIs preserve request order and simulator batch boundaries while delegating semantic decisions to the same exact-prefix primitive, leaving simulator scheduling and analytical latency unchanged.

**Tech Stack:** Python 3, PyTorch, Hugging Face Transformers, unittest/pytest.

---

### Task 1: Add exact-prefix semantic regressions

**Files:**
- Modify: `tests/test_dssd_oracle.py`

- [ ] **Step 1: Add failing exact-prefix cache and linear verification tests**

Add tests that call the same prefix twice and assert one target-model forward, then call a longer prefix and assert the forward input contains exactly that prefix. Add rejection and full-acceptance cases asserting queried prefixes stop at correction or include exactly one bonus query.

- [ ] **Step 2: Add a failing tree path test**

Build a tree with selected and unselected branches, verify it, and assert target forwards occur only for the root prefix, each selected path prefix, and the leaf bonus prefix.

- [ ] **Step 3: Add a failing all-method equivalence test**

Run `target_only`, `server_only_linear`, `server_only_tree`, `specedge_linear`, `specedge_tree`, and `dip_sd` with the contextual Hugging Face test double, mixed output lengths `[3, 5]`, and multi-request batch settings. Assert every request token list exactly equals Target-only.

- [ ] **Step 4: Verify RED**

Run: `pytest -q tests/test_dssd_oracle.py`

Expected: new tests fail because target-only uses continuation/KV generation, linear verification uses batch logits, and tree verification evaluates packed-tree logits.

### Task 2: Implement shared on-demand target semantics

**Files:**
- Modify: `src/model_runner.py`

- [ ] **Step 1: Add the shared protocol method and exact-prefix cache**

Add `target_next_token(prefix_ids)` to `ModelRunner`, implement it in `FakeModelRunner`, and initialize `HuggingFaceModelRunner._target_next_token_cache: dict[tuple[int, ...], int]`.

- [ ] **Step 2: Implement exact-prefix target lookup**

For an uncached non-empty prefix, run one target forward with `input_ids` equal to that prefix and `use_cache=False`, select greedy argmax over the shared vocabulary, cache only `tuple(prefix_ids) -> token_id`, and return it.

- [ ] **Step 3: Route linear and Target-only semantics through the lookup**

Linear verification must query each accepted prefix in order, return the first mismatch as a correction, stop on EOS, and query a bonus only after every draft token is accepted. Target-only must repeatedly query the current full prefix and append one token at a time.

- [ ] **Step 4: Route tree and batch semantics through the lookup**

Use `verify_candidate_tree(..., self.target_next_token, ...)` for each tree and invoke per-request linear/tree verification from batch APIs while preserving input order. Do not change `Simulator`, drafter generation, optimizer behavior, proposed methods, logical batching, or latency calculations.

- [ ] **Step 5: Update obsolete implementation-specific assertions**

Replace tests that require target KV continuation, equal-length batch forwards, or packed-tree masks with assertions for exact-prefix, on-demand behavior. Keep drafter KV-cache tests unchanged.

- [ ] **Step 6: Verify GREEN**

Run: `pytest -q tests/test_dssd_oracle.py`

Expected: all related model-runner tests pass.

### Task 3: Verify with one minimal real-model probe and commit

**Files:**
- Modify only if tests require: `src/model_runner.py`, `tests/test_dssd_oracle.py`

- [ ] **Step 1: Run the related model-runner tests fresh**

Run: `pytest -q tests/test_dssd_oracle.py`

- [ ] **Step 2: Run one minimal real-model smoke invocation**

Use cached Qwen target/drafter models, two requests, eight output tokens, both GPUs, and all six canonical baseline methods through `scripts/run_real_model_smoke.sh`. Do not run preflight.

- [ ] **Step 3: Check scope and whitespace**

Run `git diff --check`, inspect `git diff --name-only`, and confirm no changes to simulator scheduling/latency, drafter, optimizer, or proposed code.

- [ ] **Step 4: Commit once after all verification passes**

Stage only the implementation, regressions, and this plan; commit with `fix: unify target greedy semantics`.
