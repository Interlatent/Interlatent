# Interlatent DRTC serving — Context

The control plane and GPU inference server for running robot policies: a
**Coordinator** assigns sessions, a **Node** drives the robot and connects to a
**GPU box**, and the box loads + compiles policies and serves action chunks over
the DRTC gRPC protocol.

## Language

**Policy**:
A trained model (an HF repo id or local checkpoint) that maps observations to
action chunks. Identified by a **policy URI**.
_Avoid_: model (overloaded — used for the recorded-dataset "Model layer" too).

**Compile signature**:
The set of inputs that determine a *distinct compiled graph* for a policy.
**Membership is backend-specific and was verified against the code:**
- **Generic lerobot** (SmolVLA, Pi0, ACT, Diffusion, VQ-BeT — the backends that
  actually `torch.compile` → cudagraphs): every compile-relevant input
  (`image_keys`/`input_features`, `num_inference_steps`, `chunk_size`,
  `action_dim`) derives from the checkpoint config, i.e. from `policy_uri`.
  `session_metadata` is ignored; task seq-len is pinned. So the de-facto
  signature is **`policy_uri`** (+ process-global `DRTC_RTC` / `DRTC_COMPILE_MODE`),
  and the existing `(backend, policy_uri)` cache key is already correct.
  The runtime axis that cudagraphs are actually sensitive to is the **observation
  schema the robot sends** (which camera keys are present + their shapes) vs the
  schema `_warmup` compiled for — not metadata.
- **MolmoAct2** (released checkpoint): does **not** compile (no cudagraphs), but
  *does* read per-session `image_keys`, `norm_tag`, `inference_action_mode`,
  `num_inference_steps` from metadata. For it, those fields are a real per-session
  axis the `(backend, policy_uri)` cache key does **not** capture — a correctness
  (wrong-normalization) bug, not a cudagraph crash.
_Avoid_: assuming a single universal signature tuple across backends.

**Warmup** (disambiguate — two senses):
1. **Pre-warm** — loading + compiling a policy *before* any session, so the first
   session is fast. Triggered at GPU-process boot (`DRTC_WARMUP_POLICY`) or, in
   the closed source, by a backend warmup-target fetch.
2. (loose, avoid) "getting the box into the right state" — too vague; say
   *pre-warm* (sense 1) or *compile-for-signature*.
_Avoid_: using "warmup" to mean correctness. Pre-warm is a latency optimization;
correctness comes from keying the runtime cache on the **compile signature**.

**Onboard policy** (a.k.a. **warm policy**):
The policy a GPU box has already compiled and holds resident. The Coordinator
tracks it per-box (`warm_policy`) and refuses a session that would switch it
without confirmation (`PolicyChangeError`) because a switch recompiles (slow) and
may OOM. _Note_: today this is tracked at **policy URI** granularity, not full
**compile signature** — a known gap.

## Relationships

- A **Policy URI** has many possible **compile signatures** (one per distinct
  `image_keys` / `num_inference_steps` / `inference_action_mode` combination).
- A **GPU box** holds one or more compiled runtimes, one per **compile
  signature**, in a process-wide cache (`policy_runtime._RUNTIME_CACHE`).
- A **Session** (held in `InferenceServicer._sessions`) pins the runtime for its
  compile signature for the session's lifetime — so the live resident set is
  bounded by *concurrent sessions*, not by the cache cap.

## Flagged ambiguities

- "warmup" meant both *pre-warm* (latency) and *correct compilation* (the crash
  fix) — resolved: these are distinct. The crash fix is keying the cache on the
  **compile signature**; pre-warm is a separate latency optimization.
