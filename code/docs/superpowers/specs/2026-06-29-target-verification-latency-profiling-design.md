# Target Verification Latency Profiling Design

## Scope

Add a standalone CUDA profiling script under `scripts/`. It measures physical
target-model decode and verification latency without changing baseline
scheduling, simulator behavior, analytical latency formulas, or semantic model
verification. The profiler loads no drafter model and writes one CSV row per
requested parameter combination.

The initial execution is limited to this smoke matrix:

- `batch_size`: 1, 2
- `context_length`: 128
- `gamma`: 1, 4
- `tree_nodes`: 8
- `warmup`: 10
- `repeat`: 30

The CLI defaults expose the full requested matrix for a later run, but the full
matrix is not executed as part of this change.

## Measurement Semantics

`context_length` is the number of tokens already represented by
`past_key_values` immediately before the timed forward. Prefix token creation,
prefix KV construction, model loading, tokenizer loading, and initial CUDA
runtime initialization are outside CUDA event timing.

All timed forwards use:

- `model.eval()`;
- `torch.inference_mode()`;
- `use_cache=True`;
- a CUDA start event and end event;
- explicit CUDA synchronization before reading elapsed time;
- an attention mask covering both the cached prefix and timed input;
- position IDs beginning at `context_length`;
- `cache_position` beginning at `context_length` when the model forward
  signature supports it.

Each sample synchronizes the target device immediately before recording its
start event. Rows record `latency_scope=cuda_device_elapsed`.

Each warmup and measured repeat starts from the same immutable prefix KV state.
The profiler never feeds a timed forward's returned cache into a later repeat.
The prefix cache is cloned into an isolated legacy or native cache when
supported. The profiler does not assume every cache implementation can be
converted to a legacy tuple. It fingerprints the canonical prefix cache and
verifies it is not modified in place. When the model/cache combination cannot
provide an isolated cache object, the profiler reconstructs the prefix cache
before each sample outside the timed region.

### Target decode

After constructing `[batch_size, context_length]` prefix KV state outside the
timer, measure one cached-token forward with timed input shape
`[batch_size, 1]`.

### Linear verification

After constructing the same prefix KV state outside the timer, measure one
forward with timed input shape `[batch_size, gamma]`. This represents verifying
all draft tokens in one target forward. Returned KV state and token decisions
are not part of subsequent repeats.

### Tree verification

The repository has no executable target tree kernel with tree attention masks
and tree-aware KV positions. Therefore this profiler does not claim a real tree
kernel measurement. It measures one cached-token target forward per non-empty
tree batch, independent of `tree_nodes`, matching the simulator's current
fixed-forward latency approximation. Each tree row records its requested node
count and sets `tree_mode=fixed_forward_approx`. For each
`(batch_size, context_length)` pair this forward is measured exactly once; all
requested `tree_nodes` rows reuse the same samples and statistics.

## Model Loading

The script reads the existing config and reuses the target-loading behavior of
`HuggingFaceModelRunner` with `drafter_models` replaced by an empty mapping.
This preserves current Hugging Face cache, revision, dtype, and error-handling
behavior while preventing unrelated drafter models from loading. It then uses
the runner's target model and tokenizer metadata only; tokenization is not used
to construct profiling inputs.

Synthetic token IDs are deterministic and constrained to the target model's
usable vocabulary. Padding is unnecessary because every row has a uniform
sequence length.

## CLI and Matrix

The standalone script accepts comma-separated values for batch sizes, context
lengths, gammas, and tree node counts. Defaults are:

- `batch_size`: 1, 2, 4, 8, 16
- `context_length`: 128, 512, 1024, 2048
- `gamma`: 1, 2, 4, 8
- `tree_nodes`: 8, 16, 32, 64
- `warmup`: 10
- `repeat`: 30

It also accepts config, output, target model, target device, revision, cache,
local-only, dtype, and attention-implementation overrides where supported by
the current target loader.

## CSV Contract

Every successful or failed row contains at least:

- `method`
- `batch_size`
- `context_length`
- `gamma`
- `tree_nodes`
- `mean_ms`
- `p50_ms`
- `p95_ms`
- `std_ms`
- `gpu_name`
- `model_name`
- `tree_mode`
- `status`
- `error`
- `dtype`
- `attention_implementation`
- `use_cache`
- `torch_version`
- `transformers_version`
- `cuda_version`
- `model_revision`
- `past_length`
- `timed_input_length`
- `warmup`
- `repeat`
- `latency_scope`
- `device`
- `peak_memory_mb`

Irrelevant dimensions are empty: target decode has empty `gamma` and
`tree_nodes`; linear verification has empty `tree_nodes`; tree verification has
empty `gamma`. `use_cache` is `true` for every measured row.

Statistics use the 30 measured samples only. The standard deviation is the
population standard deviation. Percentiles use deterministic linear
interpolation.

## Failure Handling

CUDA OOM during cache preparation, warmup, or measured repeats produces a
failed CSV row for that exact combination. Its timing statistics are empty,
`status=oom`, and `error` contains a compact error message. The profiler drops
references, runs garbage collection, empties the CUDA allocator cache, and
continues with the remaining combinations. Non-OOM failures stop execution so
configuration and implementation errors are not silently converted into data.

The CSV is rewritten after each row so completed results survive a later fatal
failure.

If initial prefix KV construction fails with OOM, every decode, linear, and
tree row sharing that `(batch_size, context_length)` is written as OOM without
attempting its timed forward. Later combinations continue normally.

## Tests and Verification

Tests are written before implementation and cover:

- full default and explicit smoke matrix expansion;
- shape and metadata selection for decode, linear, and tree rows;
- correct attention-mask and position-ID construction;
- repeat isolation from returned KV state;
- statistics and CSV schema;
- OOM row recording and continuation;
- explicit `fixed_forward_approx` tree labeling;
- target-only runner configuration with no drafter models.

Targeted unit tests run locally. The only real-model execution is the specified
smoke matrix. Full preflight, the full profiling matrix, and unrelated simulator
tests are outside this task.
