"""Action-schedule reconstruction from client-supplied spans.

The client tells the server, in each Observation, which action steps
it already has scheduled (as a list of Spans). The server uses those
spans to look up the matching raw actions from the ChunkBuffer and
reconstruct enough trailing context for RTC in-painting.

We deliberately do NOT trust the client to send the raw actions
themselves — round-trip bandwidth, and the server already has the
ground-truth raw actions in its buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .chunk_buffer import ChunkBuffer


@dataclass
class InpaintingContext:
    """What policy_runtime.forward() needs to do RTC in-painting.

    `prior_actions` is the contiguous run of raw actions immediately
    preceding `next_action_step`. May be None if the buffer doesn't
    cover the requested range — in that case the server runs a
    cold-start forward (no in-painting) and accepts a one-time
    discontinuity, which is rare in practice.
    """

    prior_actions: Optional[np.ndarray]
    next_action_step: int


def reconstruct(
    buf: ChunkBuffer,
    session_id: str,
    next_action_step: int,
    spans: Iterable[tuple[int, int]],
    context_steps: int,
) -> InpaintingContext:
    """Pull up to `context_steps` of contiguous actions ending at
    next_action_step-1 out of the chunk buffer.

    Spans are an upper bound on what the client has — we only need
    the tail. Anything further back is irrelevant for in-painting.
    """
    if context_steps <= 0 or next_action_step <= 0:
        return InpaintingContext(prior_actions=None, next_action_step=next_action_step)

    # Restrict to steps the client claims to already have.
    client_steps: set[int] = set()
    for lo, hi in spans:
        if hi < lo:
            continue
        client_steps.update(range(lo, hi + 1))

    end = next_action_step - 1
    start = max(0, end - context_steps + 1)
    requested = list(range(start, end + 1))

    if not all(s in client_steps for s in requested):
        # Client doesn't claim coverage; don't try to in-paint.
        return InpaintingContext(prior_actions=None, next_action_step=next_action_step)

    prior = buf.lookup_steps(session_id, start, end)
    return InpaintingContext(prior_actions=prior, next_action_step=next_action_step)
