# H200 BF16 runtime example

This optional launcher combines the in-process structured-decision endpoint with
TP1 BF16 settings, Triton attention, atomic image inputs, fused vision
RoPE/RMSNorm, and shape-specific attention/MoE launch settings. The model weights
and installed vLLM environment are external dependencies.

From the checkout root, choose an available GPU and port:

```bash
.venv/bin/python examples/features/diffusion_typesafe/runtime/launch.py \
    --checkpoint /path/to/checkpoint --gpu 0 --port 8000 \
    --output-dir .local/diffusion-example/runtime --no-trace
```

`--dry-run` prints the resolved configuration without loading CUDA or starting a
service. The default bind address is `127.0.0.1`. Canvas capacity defaults to 256
and can be set from 1 to 1024 with `--canvas-length`. The default total context
limit is 50,000 tokens; this capacity does not establish long-context accuracy.
Use `--max-model-len 8192 --enforce-eager` for a smaller configuration.

`--text-only` disables multimodal inputs. For batches of 32 full 1024-token text
canvases, `--optimize-text-32 --text-only --canvas-length 1024` selects larger
attention/MoE tiles, CUDA graphs, and fused text normalization. These settings
are specific to the supported H200 shapes and require performance and quality
validation with the checkpoint and workload in use.

The launcher uses this checkout in spawned workers and places compiler caches
under the output directory. The attention hook checks the kernel's source hash
and applies its override only to the supported SM90 BF16 geometry. Other shapes
retain their normal launch settings. The example enables
`VLLM_BATCH_INVARIANT_FAST_LINEAR`, which retains regular GEMM dispatch while
using the other batch-invariant controls; it does not promise bitwise invariance.
Experimental model controls use the `VLLM_DIFFUSION_` environment prefix.

Logs go to the terminal. `launch.json` records the resolved configuration locally.
Unless `--no-trace` is set, `decisions.jsonl` records caller state, questions,
canvases, and responses under the output directory. `--trace PATH` selects a
different location. No recorded outputs or model weights are bundled here.

See the [API guide](../README.md) for request-time controls, image inputs,
inspection tools, and synthetic evaluations.
