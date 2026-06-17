---
status: accepted
---

# Flow-matching policies compile without CUDA graphs

The GPU server compiles policies with `torch.compile` in a mode that does **not**
capture CUDA graphs (the image default is `DRTC_COMPILE_MODE=default`). CUDA-graph
modes (`reduce-overhead`, `max-autotune`) crash **flow-matching** policies —
SmolVLA, Pi0, Pi0.5 — at *every* forward with:

```
RuntimeError: Offset increment outside graph capture encountered unexpectedly.
  (torch/_inductor/cudagraph_trees.py … graph.replay())
```

These policies sample noise (`torch.randn`) inside the compiled `sample_actions`
region. CUDA-graph replay advances the Philox RNG offset outside of capture,
which trips this assertion. It is intrinsic to the compiled graph (hence *every*
forward, not data-dependent), so it cannot be worked around with input handling.

## Considered options

- **`reduce-overhead`** (the previous default) — rejected: chosen originally for
  fast cold compile (~60s vs ~20min), but it captures CUDA graphs and so crashes
  every flow-matching policy. The fast-compile win is worthless if the policy
  can't serve.
- **`max-autotune`** — same problem (also captures CUDA graphs).
- **`max-autotune-no-cudagraphs`** — viable and fastest steady-state, but its long
  autotune compile is painful on a cold box. Kept as the recommended *opt-in*
  override for long-lived boxes with a persistent `/root/.cache` volume, where the
  one-time compile is cached.
- **`default`** — chosen: inductor kernel fusion without CUDA graphs. No crash,
  fast-enough cold compile, medium steady-state latency. The safe default across
  the GPU range the image targets (4090 → H100) and across cold/ephemeral pods.
- **Make the RNG graph-safe / register the generator with the graph** — rejected:
  deep torch/lerobot-internal change we don't own, and version-fragile.

## Consequences

- **Deterministic policies (ACT, etc.) also lose CUDA graphs** under the global
  default. Acceptable: the latency delta is small next to a hard crash for the VLA
  families this image is built for. A per-policy-type split (cudagraphs for
  deterministic, off for flow-matching) is possible later but adds branching for
  marginal gain.
- **The knob stays an env var only** (`DRTC_COMPILE_MODE`, read in
  `lerobot_backend.py`); there is no `--compile-mode` CLI flag. On managed
  providers (RunPod, etc.) set it through the provider's environment-variable
  field — it overrides the image `ENV`.
- **This is about correctness, not the policy-warmup / compile-signature work.**
  The earlier "blind warmup poisons the runtime cache" theory does not apply to the
  compiling backends: for generic lerobot, the compile signature is effectively the
  `policy_uri` (per-session metadata is ignored), so the existing
  `(backend, policy_uri)` cache key is already correct. See CONTEXT.md
  ("Compile signature").
