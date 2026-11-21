# Local Code Model Memory Planner

A model file fitting on disk does not mean an inference workload fits in memory. Plan four buckets separately, reserve headroom, then correct the estimate with a measured run.

This is a cross-model worksheet—not a model recommendation, pooled-device guide, or replacement for a model-specific failure report.

## Conservative planning model

For an inference process:

```text
required_memory
  = loaded_weights
  + KV_cache_per_sequence × concurrent_sequences
  + runtime_workspace_and_graphs
  + host_or_device_buffers
  + safety_headroom
```

A first weight estimate is `parameter_count × effective_bytes_per_parameter`, but quantized files also carry metadata/scales and runtimes may transform or duplicate buffers. Treat file size or estimator output as a **lower-bound input**, not measured peak.

A simplified dense-transformer KV estimate (when the architecture exposes these values) is:

```text
KV bytes per sequence
≈ 2 × layers × KV_heads × head_dim × context_tokens × KV_element_bytes
```

The factor 2 is key and value. Architectures, cache quantization, sliding windows, prefix caching, parallelism, and backend allocation change reality; verify the runtime’s architecture/configuration and measure.

## Fill the worksheet

Use [`assets/measurement-sheet.csv`](assets/measurement-sheet.csv). Required inputs:

- immutable model/quant identifier, parameter count, file bytes;
- runtime version/commit and launch flags;
- architecture layers, KV heads, head dimension, cache dtype if known;
- target context, batch/prefill size, concurrent sequences;
- GPU VRAM, host/unified RAM, offload split;
- measured idle, post-load, prefill peak, and steady decode peak.

Unknown values remain `unknown`; do not fill them with zero.

## Two planning examples (illustrative, not model claims)

### Example A — single-user workstation

Assume a hypothetical 7B model with a 4.5 GiB quantized artifact, one sequence, 8k target context, and 12 GiB VRAM. Start with artifact size as a lower bound, add the architecture-derived KV estimate, then reserve runtime/buffer overhead plus at least a workload-tested headroom margin. If the estimate reaches 10.8 GiB before prefill, classify it **borderline**, not “fits,” and measure with the exact runtime/model.

### Example B — shared GPU service

The same loaded weights serve four concurrent 8k sequences. KV and request-dependent buffers may scale with concurrency even when weights do not. Multiply the per-sequence term, add scheduler/workspace behavior, and test worst-case simultaneous prefill. A configuration that fits one chat can OOM under four prefills; “model fits” is the wrong conclusion.

Numbers in these examples demonstrate arithmetic only. They are not benchmark results or promises about a real model/runtime.

## Measurement protocol

1. Record free/used host and device memory before process start.
2. Load the exact model; record post-load steady state.
3. Run a short warm-up and discard its timing—but keep memory observations.
4. Prefill the target context/batch/concurrency; record peak allocated/reserved and system-level peak.
5. Decode a fixed token count; record steady and peak.
6. Repeat after changing one variable: context, concurrency, cache dtype, quant, or offload.
7. Compare estimated versus measured buckets and write a correction factor for this tuple only.
8. Deliberately test the rollback: reduce context/concurrency or offload safely after an OOM; verify process/service recovers.

Framework allocator metrics can omit driver/runtime allocations; pair them with device/system observations. Unified memory and swapping may avoid an immediate allocation failure while making the workload unusably slow—record latency separately.

## Runtime evidence and limits

- [llama.cpp](https://github.com/ggml-org/llama.cpp): GGUF quantization/offload/context/runtime options; actual memory depends on model and flags.
- [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html): explicit memory utilization, cache dtype, parallelism, and batching controls for the reviewed docs version.
- [SGLang](https://github.com/sgl-project/sglang): serving/runtime and parallel-execution project; measure version-specific pool behavior.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): NVIDIA engine-building/quantization path; include engine/workspace/cache.
- [ExLlamaV2](https://github.com/turboderp-org/exllamav2): EXL2 GPU inference path; architecture/cache/context support varies.
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes): low-bit components integrated by other frameworks; it does not represent total process memory.
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ): GPTQ toolkit; the upstream repository is archived/maintenance-sensitive, so check current runtime compatibility before choosing it.
- [Accelerate model-size estimator docs](https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator) and [official repository](https://github.com/huggingface/accelerate): planning estimates and implementation context; runtime KV/workspace/concurrency still need measurement.

The linked sources support project controls or estimator behavior. The combined formula, headroom policy, and examples are planner guidance and must be calibrated locally.

## Decision rule

Classify a target as:

- **comfortable:** measured worst-case peak leaves the project’s predeclared headroom and meets latency;
- **borderline:** only estimate exists, margin is small, swapping/offload is untested, or concurrent prefill was not measured;
- **does not fit:** repeatable OOM or unacceptable memory-pressure behavior at the required workload.

Do not lower context/concurrency after failure and still report that the original target fits.

## Maintenance

Create a new row when model artifact, runtime, flags, hardware, context, cache dtype, batch, or concurrency changes. Preserve measured corrections; never replace a failed tuple with a smaller successful one.

Sources: [`sources/sources.json`](sources/sources.json).

## License

Original planner and worksheet are available under CC BY 4.0. Linked projects retain their licenses.
