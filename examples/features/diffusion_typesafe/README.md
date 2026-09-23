# Structured decisions with Diffusion Gemma

This example implements the TypeSafe Python SDK's `system_one` and `models.list`
HTTP contracts using Diffusion Gemma. It supports Noul, Choice, and zero-based
Score questions, including structured state and rubric descriptions.

Shared instructions, state, and optional images form one prompt. Each question
and sample gets an independent bidirectional canvas containing its question,
options, and fixed answer scaffold. The default single denoising pass scores
every allowed answer token and returns the highest-scoring legal token without a
causal commit pass. Prompt prefill is additional work. Questions are never
concatenated into one model context.

Question IDs are routing keys and do not enter the model inputs or random seeds.
The complete question must fit within the configured canvas; oversized questions
are rejected without truncation. The default capacity is 256 tokens. Larger
questions or reasoning spans require a larger served capacity.

The worker-owned constraint design builds on
[alecsolder/vllm#8](https://github.com/alecsolder/vllm/pull/8). The read-only
inference and candidate-score interface also draw on
[vllm-project/vllm#57250](https://github.com/vllm-project/vllm/pull/57250).
XGrammar validates the canonical answer grammar for JSON templates. Chat
templates validate token positions and legal answer codes directly. The
worker enforces fixed tokens and the finite single-token answer set; this is not
a general arbitrary-JSON structured-output backend.

## Run

Install this checkout using the vLLM development instructions, and install the
public SDK for the client examples:

```bash
uv pip install typesafe-sdk
```

Provide a compatible Diffusion Gemma checkpoint, including its tokenizer, chat
template, and generation configuration. Model weights are external dependencies.
The adapter and model canvas capacities must agree.

```bash
VLLM_BATCH_INVARIANT=1 .venv/bin/python -m vllm.entrypoints.cli.main serve /path/to/checkpoint \
    --served-model-name diffusion-gemma \
    --host 127.0.0.1 --port 8000 \
    --diffusion-config '{"canvas_length":256}' \
    --max-logprobs 100 --max-model-len 8192 --max-num-seqs 32 \
    --attention-backend TRITON_ATTN --enforce-eager \
    --enable-prefix-caching --async-scheduling --enable-prompt-tokens-details

.venv/bin/python examples/features/diffusion_typesafe/server.py \
    --tokenizer /path/to/checkpoint --model diffusion-gemma \
    --upstream http://127.0.0.1:8000 --port 8011 --canvas 256
```

Batch-invariant mode may reduce throughput and does not establish universal
bitwise equivalence across hardware or shapes. Recheck probability stability
when changing the serving configuration.

To serve the SDK endpoint inside the vLLM API process, replace
`.venv/bin/python -m vllm.entrypoints.cli.main` with
`.venv/bin/python -m examples.features.diffusion_typesafe.inprocess` in the model
command, omit the standalone adapter, and use the model's port for SDK calls.
This serves the native OpenAI routes and `/v1/systemone`; `/v1/models` includes
both response formats. An optional [H200 runtime example](runtime/README.md)
also supplies text and vision optimization settings.

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(
    base_url="http://127.0.0.1:8011",
    api_key="local",
    model="diffusion-gemma",
    timeout=120,
) as client:
    response = client.system_one(
        state={"ticket": "I was charged twice. Please fix this today."},
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "team": Choice(criteria={"billing": None, "technical": None}),
            "urgency": Score(criteria=["can wait", "this week", "today"]),
        },
        extra_body={"samples": 1, "seed": 0},
    )
    print(response.answers)
```

The adapter accepts at most 128 questions, 32 draws per question, 100 Choice
categories, and 26 Score levels, subject to canvas and server admission limits.
Choices use `A`–`Z` for up to 26 options and uniform two-letter codes for larger
sets. Every code must occupy one distinct token at the same answer position in
the complete scaffold, as checked against the actual tokenizer. A singleton
category is valid. A 100-choice question usually needs more than 256 tokens.

For JSON canvases, full padding is the default. `--trim-canvas-padding` sends each
complete scaffold without padding, and `--minimum-canvas-padding 4` retains up to four
trailing padding tokens. `--full-canvas-type choice` retains full padding for
Choice questions. In-process equivalents are `VLLM_TYPESAFE_TRIM_CANVAS_PADDING`,
`VLLM_TYPESAFE_MINIMUM_CANVAS_PADDING`, and `VLLM_TYPESAFE_FULL_CANVAS_TYPES`.
Padding changes model inputs and can change predictions. Chat canvases round
their physical length to a 16-token boundary independently of these options.

## Request controls

Pass these fields directly to `POST /v1/systemone` or through the SDK's
`extra_body`. They apply to that request only.

| Field | Default | Meaning |
| --- | --- | --- |
| `samples` | `1` | 1–32 independent initial-noise draws per question. |
| `steps` | `1` | 1–32 constrained answer passes per draw, after reasoning. |
| `seed` | `0` | Seed for each question and draw's noise. |
| `canvas_format` | `"json"` | `"json"`, `"chat"`, or `"chat_single_user"`. |
| `reason_mode` | `false` | Generate reasoning inside each question's canvas. |
| `reason_tokens` | `32` | 1–512 reserved reasoning token positions. |
| `reason_steps` | `8` | 1–64 reasoning passes before answer passes. |
| `reason_schedule_steps` | `null` | Annealing schedule length; null uses `reason_steps`. An explicit value must be at least `reason_steps`. |
| `reason_entropy_bound` | `null` | Nonnegative reasoning acceptance budget; null uses the checkpoint default. |
| `reason_answer_noise` | `"candidates"` | Renewed answer-input noise from `"candidates"` or `"vocab"`. |
| `reason_initial_answer_noise` | `null` | Initial answer-input noise; null inherits `reason_answer_noise`. |
| `self_conditioning` | `true` | Soft model feedback after the first pass. |
| `self_conditioning_scope` | `"answer"` | `"answer"` feeds back active reasoning or answer rows; `"canvas"` feeds back all positions during intermediate passes. |
| `adaptive_samples` | `false` | Expand uncertain questions from `samples` to `max_samples`. |
| `max_samples` | `4` | Maximum 1–32 draws for adaptive sampling; at least `samples`. |
| `adaptive_threshold` | `0.1` | Expand when the initial mean distribution's normalized entropy exceeds this value in [0, 1]. |
| `return_diagnostics` | `false` | Include settings, draw counts, probability standard errors, and argmax agreement. |
| `system_prompt` | `null` | Optional replacement system instructions; null retains the default. |
| `reason_ensemble` | `null` | List of reasoning budgets, described below. |

With reasoning disabled, fixed scaffold tokens remain unchanged and each answer
pass selects the highest-scoring legal token. Soft self-conditioning and the
selected hard answer can feed the next pass. Self-conditioning controls have no
effect with a single answer pass and no reasoning.

Reason mode reserves a mutable span in the same canvas as the question and
options. After the requested reasoning passes, its tokens are frozen and the
answer passes score legal answer codes. `reason_tokens` counts physical token
positions, and `reason_steps` is a pass budget rather than a semantic completion
test. Returned `reasoning` contains one completed string per draw. JSON reasoning
cannot break the surrounding structure; chat reasoning permits plain text while
role markers and the answer scaffold stay fixed.

`chat_single_user` continues the shared state's user turn with each independent
question. `chat` starts a separate user turn for each question. Selecting either
format does not itself enable reasoning.

To average different reasoning budgets in one request:

```json
{
  "canvas_format": "chat_single_user",
  "reason_mode": true,
  "reason_ensemble": [
    {"tokens": 256, "steps": 32},
    {"tokens": 512, "steps": 64}
  ],
  "samples": 1,
  "steps": 1,
  "seed": 0
}
```

Add these fields to a request with `state` and `questions`. Each budget overrides
`reason_tokens` and `reason_steps`; `samples` applies to every budget, with at
most 32 total draws per question. Each canvas must fit independently. Ensembles
require `chat_single_user`, reason mode, and fixed sampling. Probabilities are
averaged equally; reasoning is returned in budget order, then sample order.
Omitting the field or sending null retains the single-budget path.

Adaptive sampling takes one initial round and at most one additional round for
questions above the entropy threshold. Expanded questions use the same draw
indices as fixed sampling at `max_samples`. All executed draws contribute to the
mean. Entropy allocates work; it does not establish correctness.

## Images

Images belong to the shared prompt. Enable vision with
`--limit-mm-per-prompt '{"image":4,"video":0}'` and omit
`--language-model-only`. Pass inline base64 image data through `extra_body`:

```python
extra_body = {"images": [{"url": "data:image/png;base64,..."}]}
```

Only PNG, JPEG, and WebP data URLs are accepted. The adapter validates the actual
format and does not fetch remote URLs or read server-local paths. Limits are
four images, 8 MiB per image, 16 MiB total, and 16 million pixels per image.
Every question receives the same ordered images. Vision encoding and prompt
prefill are additional work beyond answer denoising.

## Native grouped reads

The adapter submits a `/v1/chat/completions` parent for each sampling round, with
one child per question, sample, and reasoning budget in that round. An adaptive
round includes only the selected questions and their additional samples. Each
child's `diffusion_reads` descriptor supplies a complete `seed_canvas`, one
`answer_positions` entry,
the complete `logprob_token_ids` candidate set, and optional `steps`,
`self_conditioning`, and `self_conditioning_scope` controls.

The parent uses `ignore_eos=true`, unfiltered temperature-one sampling,
`logprobs=true`, `return_tokens_as_token_ids=true`, and a `max_tokens` budget
large enough for every canvas. `answer_only_logprobs=true` returns candidate
scores for the answer row. Output `choices[i].index` identifies its descriptor.
The complete emitted canvas still counts toward usage when only answer rows are
returned. Each child receives a budget equal to its own physical canvas length.

Scalar `vllm_xargs` controls include `diffusion_read_only`,
`diffusion_seed_canvas`, `diffusion_answer_positions`, `diffusion_read_steps`,
`diffusion_read_self_conditioning`, and
`diffusion_read_self_conditioning_scope`. The native request schema additionally
documents experimental reasoning feedback, stopping, and answer-pass averaging
controls. Sampling filters, penalties, and stop strings are unsupported for
constrained reads.

Grouped admission is atomic and requires async scheduling. Children share the
prompt but keep independent canvases and attention sequences. Prefix caching can
reuse prompt blocks; logical usage and shared-prompt hashes do not measure
executed model forwards or prove cache reuse. Use scheduler/profiler evidence
when measuring execution cost.

## Scores and inspection

Raw logprobs are gathered before sampling constraints and temperature annealing.
The adapter normalizes over all allowed candidates. Noul returns the normalized
yes score, Choice returns the full distribution and its argmax, and Score returns
the probability-weighted mean of zero-based rubric indices. Missing or nonfinite
candidate scores fail the request.

`confidence` is `1 - H(p) / log(N)`, or one for a singleton. This example defines
that convention explicitly. Denoiser scores are not calibrated probabilities of
correctness. Samples average final distributions rather than voting on answers.

Add `--trace .local/diffusion-example/decisions.jsonl` to the standalone adapter,
or set
`VLLM_TYPESAFE_TRACE` in the in-process server, to record requests and canvases.
Traces include caller state, question text, and model outputs. Image records
contain metadata and hashes rather than pixels.

```bash
.venv/bin/python examples/features/diffusion_typesafe/tui.py \
    .local/diffusion-example/decisions.jsonl --follow
.venv/bin/python examples/features/diffusion_typesafe/evaluate.py \
    --base-url http://127.0.0.1:8011 --model diffusion-gemma \
    --output .local/diffusion-example/evaluation.json
.venv/bin/python examples/features/diffusion_typesafe/image_showcase.py \
    --base-url http://127.0.0.1:8011 \
    --trace .local/diffusion-example/decisions.jsonl \
    --output-dir .local/diffusion-example/image-showcase
```

The TUI displays recorded token positions and candidate scores, without inventing
intermediate denoising frames. The evaluator and showcases use synthetic examples
for protocol, quality, and request-isolation smoke checks. They do not establish
general model accuracy. Re-evaluate model quality and performance with the chosen
checkpoint, hardware, and request controls before drawing comparisons.
