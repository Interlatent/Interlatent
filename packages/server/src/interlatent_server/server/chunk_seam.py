"""Chunk-boundary seam smoothing, shared across policy backends.

Extracted from ``lerobot_backend`` so backends that cannot do RTC in-painting
still have a seam smoother. Two families need it for opposite reasons:

* **Non-flow lerobot policies** (ACT and other single-shot decoders) have no
  in-painting mechanism at all — RTC needs a flow-matching sampler.
* **World-action models** have one in spirit (ground-truth frames are injected
  into the KV cache after each chunk) but expose nothing equivalent to
  lerobot's ``prev_chunk_left_over``, so from this side they are also
  single-shot.

The blend operates in **robot-action space**, on the already-postprocessed
chunk, which is what makes it backend-agnostic: it needs no knowledge of
normalization, of the model's latent action representation, or of how the
chunk was produced — only that consecutive chunks are indexed in the same
absolute action-step space.

Nothing here holds state. The caller owns the previous chunk and its start
step, so a backend that does not smooth seams simply never calls in.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def crossfade_chunk(
    chunk: np.ndarray,
    prev: Optional[np.ndarray],
    prev_start: int,
    next_action_step: int,
    inference_delay: int,
    *,
    ramp_steps: int = 0,
) -> np.ndarray:
    """Blend ``chunk`` into ``prev``'s overlapping tail. Always safe to call.

    Returns ``chunk`` unchanged when there is no previous chunk, no overlap,
    or a shape mismatch.

    Geometry, all in absolute action-step space:

    * previous chunk covers ``[ps, ps + P)`` where ``ps = prev_start``
    * new chunk covers ``[ns, ns + N)`` where ``ns = next_action_step``
    * overlap is ``[ns, min(ns + N, ps + P))`` — the new chunk anchors at the
      client's cursor, so ``ns >= ps`` and the overlap starts at ``ns``

    Weighting: the weight on the PREVIOUS plan holds at 1.0 through the
    latency region (the first ``inference_delay`` steps, which the client will
    mostly drop as already-executed), then cosine-ramps to 0.0 over
    ``ramp_steps``. So the seam is C0-continuous with what the robot is
    actually executing and hands off smoothly to the fresh prediction,
    instead of stepping.

    ``ramp_steps=0`` derives the ramp from ``inference_delay``.
    """
    if prev is None or prev.shape[0] == 0:
        return chunk
    if prev.shape[1:] != chunk.shape[1:]:
        return chunk  # action_dim changed — don't blend

    # Where the new chunk's first step sits inside the previous chunk. A
    # negative offset means the new chunk starts BEFORE the previous one,
    # which is what a fresh session on a cached backend looks like (the new
    # cursor restarts near 0 while prev_start still holds the old session's
    # high-water step). offset >= len(prev) means the robot ran past the whole
    # previous chunk. Neither has a usable seam, and both would index out of
    # bounds.
    offset = int(next_action_step) - int(prev_start)
    if offset < 0 or offset >= prev.shape[0]:
        return chunk

    m = min(chunk.shape[0], prev.shape[0] - offset)
    if m <= 0:
        return chunk

    anchor = max(0, min(int(inference_delay), m - 1))
    ramp = ramp_steps or max(2, int(inference_delay))
    ramp = max(1, min(ramp, m - anchor))

    out = chunk.copy()
    for o in range(m):
        p = offset + o  # index into prev, guaranteed in range by the guards
        if o <= anchor:
            w = 1.0
        elif o >= anchor + ramp:
            w = 0.0
        else:
            phase = (o - anchor) / ramp
            w = 0.5 * (1.0 + float(np.cos(np.pi * phase)))
        if w <= 0.0:
            continue
        out[o] = w * prev[p] + (1.0 - w) * chunk[o]
    return out
