# Qwen3.8-27B Port Plan

Status: proposed (research complete, no code changes yet)
Date: 2026-08-27
Target model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)

This document records the codebase analysis and the staged plan for running
Qwen3.8-27B through the TurboFieldfare stack (`.gturbo` format, repack
installer, Metal runtime, CLI, server, Mac app).

## 1. Target model architecture

From the model card and `config.json` (`model_type: qwen3_5`,
architectures `Qwen3_5ForConditionalGeneration`):

- Dense (not MoE): 27B parameters, 64 layers, hidden 5120, FFN intermediate
  17408.
- Hybrid layer layout: 16 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated
  Attention -> FFN)); `full_attention_interval: 4`.
  - Gated DeltaNet (linear attention): 48 V heads / 16 QK heads, head dim
    128, conv kernel dim 4, state in FP32 (`mamba_ssm_dtype`).
  - Gated attention: GQA 24 Q heads / 4 KV heads, head dim 256, partial
    rotary factor 0.25 (64 rotary dims), rope theta 10M, swish output gate
    (`attn_output_gate: true`).
- Vocab 248,320 (padded), `tie_word_embeddings: false` (separate lm_head).
- One MTP (multi-token prediction) layer; usable later for speculative
  decoding. `mlx-community` publishes `Qwen3.8-27B-MTP-4bit` draft weights.
- Native vision-language model: vision tower depth 27, hidden 1152, patch
  16, spatial merge 2, interleaved mrope sections [11, 11, 10].
- Thinking mode on by default (`<think>...</think>`), ChatML template,
  `reasoning_effort` levels, `preserve_thinking`.
- Recommended sampling:
  - Thinking: temperature 1.0, top_p 0.95, top_k 20.
  - Instruct: temperature 0.7, top_p 0.80, top_k 20, presence_penalty 1.5.
- Context: 262,144 natively; up to 1M via YaRN (out of scope initially).
- Source checkpoint: `mlx-community/Qwen3.8-27B-4bit` exists and matches the
  MLX 4-bit affine quant format the current installer consumes.

## 2. Current Gemma 4 coupling (what has to change)

### Format (`.gturbo` v1)

- `Sources/TurboFieldfareFormat/GTurboFormatV1.swift` — magic, dtypes,
  alignment: model-agnostic, reusable as-is.
- Wire layout: `model_weights.bin` (resident index + mmap payload),
  `packed_experts/` (per-layer streamed expert files), `manifest.json`.
- `ManifestArch` (`Sources/TurboFieldfare/Infrastructure/ModelIO/
  ManifestReader.swift`) is a Gemma-MoE field set: slidingWindow,
  finalLogitSoftcap, numExperts, topKExperts, attentionKEqV, dual
  headDim/KVHeads, fullAttentionLayerMask.
- The runtime validates the manifest field-by-field against the
  compile-time baseline `ArchConfig.gemma4_26B_A4B`
  (`Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift`);
  `Model.load(expecting:)` defaults to it
  (`Sources/TurboFieldfare/Runtime/Inference/Model.swift`).

### Runtime

- `Model.swift` hardcodes Gemma tensor names (router, shared expert,
  sandwich FFN norms, layer_scalar, per-head q/k norms) and
  `validateRuntimeSchema` requires the full MoE sandwich layout.
- `RealForwardRunner.swift` (~1900 lines) implements the Gemma pipeline:
  sliding-window + full attention (every 6th layer), QK-norm, MoE with
  expert streaming (`PreadExpertStreamer`).
- No linear-attention / recurrent-state support exists anywhere.
- Reusable kernels: `DequantInt4GEMV`, `DequantInt8GEMV`, `RMSNorm`,
  `RoPE`, `Attention` (SWA/full), sampler, LM head int4 chain.

### Repack installer

- Pinned source: `Sources/TurboFieldfareRepack/Core/Remote/
  SupportedModelSource.swift` = `mlx-community/gemma-4-26b-a4b-it-4bit` +
  pinned revision; `Core/Verification/SourceFingerprint.swift` pins the
  index.json SHA-256 per repo.
- `Core/Format/ArchInfo.swift` parses the HF `config.json` but expects
  Gemma fields (moe_intermediate_size, sliding_window, softcap...); it
  would fail on the qwen3_5 config.

### Tokenization / chat

- `GFTokenizer` (`Sources/TurboFieldfare/Tokenization/Tokenizer.swift`)
  uses swift-transformers (Hub/Tokenizers), which is generic and supports
  Qwen BPE, but is pinned to `google/gemma-4-26B-A4B-it`, owns the Gemma
  chat framing (`chatTemplateIdentity`), and has a Gemma-specific tool
  parser (`GemmaToolCallParser.swift`) and SentencePiece byte-fallback
  decoder (`GemmaDecoding.swift`) that do not apply to Qwen's byte-level
  BPE.

## 3. Key gap

Gemma 4 is MoE with sliding/full attention; Qwen3.8-27B is dense hybrid
with Gated DeltaNet. 48 of 64 layers do not use a KV cache at all — they
carry a fixed-size recurrent state (short causal conv1d + chunked delta
rule). Building Metal kernels for the delta rule (chunked prefill scan and
recursive decode step) is the largest new component. Reference
implementations exist in mlx-lm / mlx-swift-lm (Qwen3-Next / Qwen3.5
Gated DeltaNet) and should be ported rather than invented.

## 4. Staged plan

### Phase 0 — Scope

- Text first; vision in a later phase (the Qwen tower with interleaved
  mrope is a project of its own).
- No MTP initially; speculative decoding as an optional later phase using
  `mlx-community/Qwen3.8-27B-MTP-4bit`.
- Format strategy: add an `archID` discriminator (e.g. `"qwen3_5"`) and
  make MoE fields optional, rather than inventing fake MoE values.

### Phase 1 — Format and manifest

- Add `archID` to the manifest; generalize `ManifestArch` / `ArchConfig`
  into per-family fields.
- New arch fields for linear attention: `linearNumKeyHeads` (16),
  `linearNumValueHeads` (48), `linearKeyHeadDim` (128),
  `linearValueHeadDim` (128), `linearConvKernelDim` (4),
  `fullAttentionInterval` (4).
- Model-free format tests in `Tests/TurboFieldfareFormat*`.

### Phase 2 — Repack installer

- New pinned source: `mlx-community/Qwen3.8-27B-4bit` + pinned revision +
  new `SourceFingerprint` entry.
- Extend `ArchInfo.load` for the qwen3_5 `text_config` schema.
- Tensor mapping Qwen (`model.layers.N.{linear_attn,self_attn,mlp}.*`)
  into the gturbo layout. Dense model: everything resident; make the
  packed-experts layout tolerate zero routed experts (or define a dense
  variant).

### Phase 3 — Runtime loader

- `ArchConfig.qwen3_8_27B` baseline + new runtime schema validation for
  Gated DeltaNet tensors (in_proj, conv1d, A_log / dt_bias, gate norm) and
  a separate lm_head (embeddings are not tied).

### Phase 4 — Kernels (largest chunk)

- Gated DeltaNet: causal conv1d, chunked delta rule for prefill, recursive
  step for decode, FP32 state.
- Gated attention: output gate + per-head QK norm + partial RoPE (reuse
  `RoPE.swift` / `Attention.swift` with modifications).
- Reuse unchanged: int4/int8 GEMV dequant, RMSNorm, sampler, embedding
  lookup, LM head.

### Phase 5 — State management

- New `LinearAttentionState` store for the 48 linear layers alongside
  `KVCacheManager` (only the 16 full-attention layers need KV). Recurrent
  state is fixed-size, which simplifies long-context memory budgeting.

### Phase 6 — Tokenizer and chat

- Load the Qwen tokenizer via swift-transformers; drop the Gemma decoder
  path for this arch.
- ChatML template with thinking blocks (`enable_thinking`,
  `preserve_thinking`); Qwen tool-call parser (`<tool_call>` JSON);
  stop tokens `<|im_end|>` / `<|endoftext|>`; Qwen sampling defaults.

### Phase 7 — CLI / server / app integration

- Model selection; stream `reasoning_content` on the OpenAI-compatible
  server; update `AGENTS.md`, `docs/RUNTIME_CONTROLS.md`, and the server
  guide.

### Phase 8 — Vision (later)

- `.vision.gturbo` companion for the Qwen tower (patch 16, merge 2,
  interleaved mrope), replicating the existing vision-pack installer
  pattern and keeping the fail-closed image path semantics.

### Phase 9 — Validation

- Model-free test suites as today; golden-logit comparison against an
  MLX/mlx-lm reference run; benchmark strictly per
  `docs/COMMUNITY_BENCHMARKS.md`.

## 5. Risks

1. Gated DeltaNet kernels are the top technical risk — mitigate by porting
   the mlx-swift-lm reference and validating against golden logits early.
2. Memory: 27B dense at 4-bit is ~15 GB fully resident; without expert
   streaming the minimum RAM rises versus the current MoE model.
3. Vision mrope + thinking-mode plumbing are each meaningful follow-up
   work; both are deliberately deferred to their own phases.
