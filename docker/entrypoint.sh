#!/usr/bin/env bash
# entrypoint.sh — launch the Interlatent policy server.
#
# Two modes, picked from the environment (no flags needed on providers
# that only let you set env vars):
#
#   INTERLATENT_API_KEY set (registered with a coordinator, the normal case)
#     -> `interlatent-serve`: register this box with your coordinator using
#        that key, then serve with owner-checked RPC auth ON.
#        Required: INTERLATENT_ADVERTISE_ADDRESS (host[:port] your robot
#        nodes can reach — the coordinator hands it to them verbatim).
#        Optional: INTERLATENT_COORDINATOR, INTERLATENT_BOX_NAME,
#        INTERLATENT_BOX_ID, DRTC_WARMUP_POLICY, INTERLATENT_INSECURE=1.
#
#   otherwise (no coordinator)
#     -> bare `python -m interlatent_server.serve_gpu`: a local,
#        unregistered server. Optional: DRTC_WARMUP_POLICY.
#
# Extra args are forwarded verbatim, so CLI flags still work:
#   docker run ... interlatent-server:latest --port 50052
set -euo pipefail

PORT="${INTERLATENT_PORT:-50051}"

if [ -n "${INTERLATENT_API_KEY:-}" ]; then
    ARGS=(--port "$PORT")
    if [ -n "${INTERLATENT_BOX_NAME:-}" ]; then
        ARGS+=(--name "$INTERLATENT_BOX_NAME")
    fi
    if [ "${INTERLATENT_INSECURE:-}" = "1" ]; then
        ARGS+=(--insecure)
    fi
    # --api-key/--coordinator/--advertise-address/--box-id/--warmup-policy
    # all default from their env vars inside the CLI.
    exec interlatent-serve "${ARGS[@]}" "$@"
else
    echo "[entrypoint] INTERLATENT_API_KEY not set — serving locally," \
         "unregistered (set it to register this box with your coordinator)" >&2
    exec python -m interlatent_server.serve_gpu --port "$PORT" "$@"
fi
