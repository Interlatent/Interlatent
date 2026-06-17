# Stability test plan

What must be exercised before declaring a release of the DRTC serving stack
(`interlatent-serve` / the docker image) stable.

## Why this exists: the compile path has no CI coverage

Every automated test uses the `echo` / `tiny_torch` backends — see
`tests/test_e2e_loopback.py` ("No GPU, no robot — echo and tiny_torch
backends"). Those backends never call `torch.compile`, so the **real-policy /
compile / CUDA-graph / warmup path has zero automated coverage**. CI stays green
even if a real policy is broken. Anything in Tiers 1–2 below therefore requires a
**manual run on a real GPU**; the control plane (Tier "already covered") does not.

Context for the current release: the default `torch.compile` mode was changed
from `reduce-overhead` to `default` to stop a CUDA-graph + Philox-RNG crash on
flow-matching policies — see
[ADR-0004](docs/adr/0004-no-cudagraphs-for-flow-matching.md). Tier 1 validates
that fix and guards against regressing the other policy families.

---

## Tier 1 — Compile-path fix + regression (real GPU required)

- [ ] **SmolVLA** — load → warmup → sustained inference; **no** `RuntimeError:
  Offset increment outside graph capture` at any forward. *(the fix itself)*
- [ ] **Pi0 / Pi0.5** — same flow-matching RNG path; confirm independently, do not
  assume SmolVLA covers them.
- [ ] **ACT** (deterministic) — still loads + serves correct actions under
  `default`. Primary *regression* risk: it previously got CUDA graphs via
  `max-autotune`. Check output sanity + latency.
- [ ] **Diffusion / VQ-BeT / TDMPC** — at least load + serve one action chunk
  under `default`.
- [ ] **Latency at the control rate** — measure SmolVLA forward latency under
  `default`. Reference: ~370 ms eager vs ~100 ms CUDA-graph-compiled; `default`
  sits between. If too slow for the loop, the production answer is
  `DRTC_COMPILE_MODE=max-autotune-no-cudagraphs`.
- [ ] **`max-autotune-no-cudagraphs` override** — confirm it *also* doesn't crash
  and beats `default` on latency (recommended for persistent boxes).
- [ ] **Persistent cache** — first run compiles; container restart reuses the
  inductor cache + HF weights (no recompile, no re-download) and still doesn't
  crash on the cached path. Requires `-v …:/root/.cache`.

## Tier 2 — End-to-end stack with a real policy (once, on GPU)

- [ ] **Full coordinator path** — `pair → gpu add → session start → node drives
  robot → inference → session stop`.
- [ ] **Graceful stop → CloseSession → dataset build/merge/upload** — the
  ADR-0001 critical path. Verify the server's idle-GC does **not** discard the
  episode.
- [ ] **Recording destinations** flush on stop: local `output_dir`, `s3_uri`, and
  the hosted inbox (ADR-0002).
- [ ] **Boot warmup** — `DRTC_WARMUP_POLICY=<smolvla>` under `default` → first
  session is fast and does not crash.
- [ ] **Coordinator-absent resilience** — kill the coordinator mid-session; the
  node keeps driving the robot (control plane vs data plane, ADR-0001).

## Tier 3 — Known latent gaps: decide **block vs. accept-and-document**

These were surfaced in design and deliberately deferred (see the
`deferred-gpu-server-hardening` project memory). None block the compile-mode fix,
but each should be consciously ruled in or out of "stable":

- [ ] **Two nodes → one GPU** — no guard exists. At minimum *characterize* the
  failure (OOM? garbage output? works?) and document it.
- [ ] **Policy switch on a warm box** — `PolicyChangeError` surfaces cleanly via
  the coordinator (`--confirm-policy-change` to override).

## Deployment

- [ ] Rebuilt image boots on the target provider(s) with `--gpus all`; healthcheck
  reaches `(healthy)`; Tailscale path works if used.

