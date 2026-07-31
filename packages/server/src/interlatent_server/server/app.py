"""Local-only entry point for the DRTC gRPC server.

The production launcher lives in :mod:`interlatent_server.serve_gpu`
(persistent GPU box, optional ``--warmup-policy``, persists
torch.compile artifacts across restarts). This file is intentionally
minimal — it exists so tests and SDK developers can run a real gRPC
server on localhost without any cloud dependency.

Run:
    python -m interlatent_server.server.app
"""

from __future__ import annotations

import asyncio
import logging
import sys

from .transport import serve_local


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve_local())


if __name__ == "__main__":
    main()
    sys.exit(0)
