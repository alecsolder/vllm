# Gemma 4 grammar jump decoding

This branch is a research prototype for accelerating constrained Gemma 4 tool
calls. When the grammar has exactly one legal continuation, the decoder can
copy that deterministic token run into the speculative draft buffer instead of
asking the MTP drafter to predict it token by token.

The prototype is intentionally presented as a readable demonstration, not as a
production-ready vLLM change.

## How it works

1. Chat-completion tool definitions are converted to a Gemma 4-native
   llguidance grammar (`call:NAME{key:value}` rather than JSON syntax).
2. The scheduler, which owns the grammar matchers, asks for upcoming forced
   tokens and predicts where the current speculative draft will leave each
   matcher.
3. If at least `K` forced tokens remain, their token IDs are sent to the worker
   and written directly into the `K` MTP draft slots.
4. When every request in the batch is fully forced, the rejection sampler can
   accept those drafts without its normal verification kernel. Mixed batches
   retain the ordinary MTP path.

The matcher remains scheduler-owned, so tensor-parallel workers receive only a
picklable `{request_id: [token_ids]}` map. A batch-fraction gate avoids replacing
useful MTP work when only a small part of a large batch can jump.

Useful entry points:

- `vllm/v1/core/sched/scheduler.py`: forced-run discovery and matcher updates.
- `vllm/v1/worker/gpu_model_runner.py`: draft-buffer quick step.
- `vllm/v1/sample/rejection_sampler.py`: fully forced batch fast path.
- `vllm/v1/structured_output/gemma_chat_grammar.py`: Gemma 4 tool grammar.
- `vllm/entrypoints/openai/chat_completion/serving.py`: request-time grammar
  injection.

## Prototype results

The evaluation suggests that jump decoding preserves quality, but that its
speed benefit depends on workloads containing long deterministic grammar runs.

| Evaluation | MTP baseline | Jump decoding | Interpretation |
| --- | ---: | ---: | --- |
| BFCL, Gemma 4 A4B, 3-group macro average (`n=4,441` total) | 68.73 | 68.92 | No measurable quality change |
| BFCL, Gemma 4 31B, count-weighted (`n=4,441`) | 79.15 | 79.06 | No measurable quality change |
| Synthetic grammar-heavy latency, A4B | 1175.0 ms | 1120.3 ms | 4.7% lower latency |
| Synthetic grammar-heavy latency, 31B | 2064.4 ms | 1957.3 ms | 5.2% lower latency |

An experimental **detect-only / unenforced** arm reached 5,309 tok/s versus
4,234 tok/s for MTP at concurrency 32 on the synthetic A4B workload (+25.4%).
That mode deliberately removes masking at non-forced positions and should be
treated as a research comparison, not equivalent constrained decoding.

The key negative result is equally important: a realistic BFCL non-live + live
run observed **zero jump-forward events** across 3,641 requests and 328,878
generated tokens. Consequently, this work does not demonstrate a real-BFCL
speedup; those requests rarely exposed the deterministic runs that activate the
mechanism.

The A4B BFCL row is the unweighted mean of non-live, live, and multi-turn
accuracy. The 31B row is the count-weighted aggregate (`correct / 4,441`), so
the absolute values across those two rows are not directly comparable.

## Sketch of use

Serve a Gemma 4 model with MTP speculation and the llguidance backend, then opt
into the prototype:

```bash
vllm serve <model> \
  --speculative-config \
    '{"method":"mtp","model":"<assistant-model>","num_speculative_tokens":3}' \
  --structured-outputs-config \
    '{"backend":"guidance","enable_jump_decoding":true}' \
  --tool-call-parser gemma4
```

A normal OpenAI-compatible chat request containing `tools` then receives an
automatically generated Gemma 4 grammar. The optional
`VLLM_GEMMA4_JUMP_MIN_BATCH_FRACTION` environment variable controls the
per-step batch gate (default `0.5`).

The batched forced-token discovery path uses
`LLExecutor.compute_ff_tokens_par`, available in recent llguidance builds.

For experiments only, setting `jump_decoding_unenforced` in the structured
outputs configuration enables the detect-only mode. If unconstrained sampling
diverges from the grammar, that request fails open and stops jumping.

## Scope and limitations

- This is a historical research snapshot and is not proposed upstream as-is.
- The grammar generator targets Gemma 4's native tool-call tokenization and a
  useful subset of JSON Schema; it is not a general JSON Schema implementation.
- Benefits require sufficiently long forced-token runs and can disappear on
  realistic traffic or heterogeneous batches.
- The branch still needs model-specific integration testing before deployment.
