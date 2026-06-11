#!/usr/bin/env bash
# entrypoint.sh — wraps `interlatent-serve` (the DRTC inference server).
#
# All knobs are env vars so the same image works across providers
# (RunPod, Lambda, Vast, Prime Intellect, bare metal) without
# rebuilding. Any extra args passed via `docker run ... <args>` are
# forwarded to interlatent-serve verbatim, so CLI flags still work too.
#
# Env (DRTC):
#   DRTC_PORT           Port to listen on (default 50051).
#   DRTC_HOST           Bind host (default 0.0.0.0).
#   DRTC_WARMUP_POLICY  Policy to pre-warm at startup (HF repo id or
#                       checkpoint path). Self-describing policies
#                       (SmolVLA, Pi0, ACT, ...) work here; MolmoAct2's
#                       released checkpoints need per-session image_keys
#                       and load on first OpenSession instead.
#   HF_TOKEN            Hugging Face token for private policy repos.
#                       Forwarded as HUGGING_FACE_HUB_TOKEN.
#
# Env (teleop — optional; only used if INTERLATENT_TELEOP_SECRET is set):
#   INTERLATENT_TELEOP_SECRET   HMAC secret used to verify browser/node
#                               join tokens for DAgger takeover. Mint
#                               session tokens with the same secret.
#                               When unset, the relay is skipped
#                               entirely and no port is opened.
#   INTERLATENT_TELEOP_WS_PORT  Port for the WS relay (default 50052).
#                               Exposed via `tailscale serve` when
#                               Tailscale is in use, so the dashboard
#                               can reach it through the same tailnet
#                               as the gRPC port.
#
# Env (Tailscale — optional; only used if TS_AUTHKEY is set):
#   TS_AUTHKEY          Reusable / ephemeral Tailscale auth key. When
#                       present, the entrypoint starts tailscaled in
#                       userspace-networking mode, joins the tailnet,
#                       and exposes DRTC_PORT to tailnet peers via
#                       `tailscale serve`. The container does NOT need
#                       --cap-add NET_ADMIN or /dev/net/tun.
#   TS_HOSTNAME         Tailnet hostname (default: interlatent-drtc).
#   TS_EXTRA_ARGS       Extra flags forwarded to `tailscale up` (e.g.
#                       "--advertise-tags=tag:gpu --ssh").
#   TS_STATE_DIR        Where to persist tailscaled state (default
#                       /var/lib/tailscale). Mount a volume here to
#                       survive container restarts without re-auth.
set -euo pipefail

PORT="${DRTC_PORT:-50051}"
HOST="${DRTC_HOST:-0.0.0.0}"
TELEOP_PORT="${INTERLATENT_TELEOP_WS_PORT:-50052}"
TS_STATE_DIR="${TS_STATE_DIR:-/var/lib/tailscale}"
TS_HOSTNAME="${TS_HOSTNAME:-interlatent-drtc}"
# Tag advertised on `tailscale up`. The ACL grants ``funnel`` to this
# tag (see Tailscale admin → Access Controls → nodeAttrs), so every new
# box auto-gets Funnel permission without per-node config. Override
# via TS_TAGS=tag:foo when joining a tailnet that uses a different tag.
TS_TAGS="${TS_TAGS:-tag:gpu-drtc}"

# HF auth: accept either canonical name. Some providers inject one
# but not the other — pick whichever was set.
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

# ---------- Tailscale (optional) ----------------------------------------
#
# If TS_AUTHKEY is provided we bring the container onto the tailnet so
# the Pi-side `interlatent-node` daemon can reach DRTC_PORT directly
# over Tailscale's MagicDNS / 100.x address. Userspace networking
# keeps the container unprivileged; `tailscale serve --tcp` is what
# actually makes the in-container port reachable from peers.
start_tailscale() {
  mkdir -p "${TS_STATE_DIR}" /tmp/tailscale

  echo "[entrypoint] starting tailscaled (userspace networking)..."
  tailscaled \
      --tun=userspace-networking \
      --socks5-server=localhost:1055 \
      --state="${TS_STATE_DIR}/tailscaled.state" \
      --socket=/tmp/tailscale/tailscaled.sock \
      >/var/log/tailscaled.log 2>&1 &

  # Wait for the control socket to appear. `tailscale` blocks on it
  # otherwise, which makes startup races painful to diagnose.
  for i in $(seq 1 30); do
      [ -S /tmp/tailscale/tailscaled.sock ] && break
      sleep 0.5
  done
  if [ ! -S /tmp/tailscale/tailscaled.sock ]; then
      echo "[entrypoint] ERROR: tailscaled socket never appeared; see /var/log/tailscaled.log" >&2
      tail -n 50 /var/log/tailscaled.log >&2 || true
      exit 1
  fi

  echo "[entrypoint] joining tailnet as '${TS_HOSTNAME}' with tags '${TS_TAGS}'..."
  # shellcheck disable=SC2086  # we want word-splitting on TS_EXTRA_ARGS
  tailscale --socket=/tmp/tailscale/tailscaled.sock up \
      --authkey="${TS_AUTHKEY}" \
      --hostname="${TS_HOSTNAME}" \
      --advertise-tags="${TS_TAGS}" \
      --accept-routes \
      ${TS_EXTRA_ARGS:-}

  # Expose DRTC_PORT to tailnet peers. Without this, userspace
  # networking blocks inbound TCP — outbound (SOCKS5) would still
  # work but the Pi could not connect in. `--bg` returns immediately
  # and persists the config in tailscaled state.
  echo "[entrypoint] exposing DRTC port ${PORT} to tailnet via tailscale serve..."
  tailscale --socket=/tmp/tailscale/tailscaled.sock serve \
      --bg --tcp "${PORT}" "tcp://127.0.0.1:${PORT}"

  # Teleop port: expose :${TELEOP_PORT} to tailnet peers as raw TCP.
  # That single listener serves BOTH teleop clients:
  #
  #   - Browser: the dashboard connects to the public teleop proxy
  #     (teleop.interlatent.com, see teleop-proxy/), which is
  #     itself a tailnet member and forwards to this box at
  #     ws://il-drtc-<short>.<tailnet>:${TELEOP_PORT}. The browser only
  #     ever sees the proxy's *.interlatent.com cert, so HTTPS
  #     mixed-content / SNI-filtering networks are not a problem.
  #
  #   - Node: already on the tailnet, connects directly over plain
  #     ws:// (tailnet encrypts the underlay).
  #
  # We intentionally do NOT set up Tailscale Funnel anymore. The proxy
  # superseded it — Funnel required a per-tailnet ACL grant, only
  # listened on :443, and broke on SNI-filtered networks. Reaching the
  # box over the tailnet via the proxy needs nothing beyond this serve.
  if [ -n "${INTERLATENT_TELEOP_SECRET:-}" ]; then
    echo "[entrypoint] exposing teleop port ${TELEOP_PORT} to tailnet via tailscale serve..."
    tailscale --socket=/tmp/tailscale/tailscaled.sock serve \
        --bg --tcp "${TELEOP_PORT}" "tcp://127.0.0.1:${TELEOP_PORT}"
    echo "[entrypoint] browser teleop is served via the teleop proxy"
    echo "[entrypoint]   (teleop.interlatent.com -> ${TS_HOSTNAME}:${TELEOP_PORT} over the tailnet)"
  fi

  # Friendly log line so the operator can find the connect URL.
  TS_IP="$(tailscale --socket=/tmp/tailscale/tailscaled.sock ip -4 2>/dev/null | head -n1 || true)"
  if [ -n "${TS_IP}" ]; then
      echo "[entrypoint] reachable on tailnet at  ${TS_IP}:${PORT}"
      echo "[entrypoint] on the Pi:  export INTERLATENT_DRTC_URL=${TS_IP}:${PORT}"
  fi
}

if [ -n "${TS_AUTHKEY:-}" ]; then
  start_tailscale
else
  echo "[entrypoint] TS_AUTHKEY not set — skipping Tailscale. The"
  echo "[entrypoint] DRTC port will only be reachable via docker -p mapping."
fi

# ---------- GPU sanity check --------------------------------------------
#
# Confirm a GPU is actually visible before we burn time loading a
# policy. `nvidia-smi` is shipped by the NVIDIA container runtime
# when --gpus all is passed; if it's missing we tell the operator
# rather than failing 90s later inside torch.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[entrypoint] WARNING: nvidia-smi not found. Did you start the container with --gpus all?" >&2
elif ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[entrypoint] WARNING: nvidia-smi present but no GPU detected." >&2
else
  echo "[entrypoint] GPU(s):"
  nvidia-smi -L | sed 's/^/[entrypoint]   /'
fi

# Authoritative GPU check: torch.cuda.is_available(). nvidia-smi can
# show a GPU while torch silently can't use it — happens when the host
# driver is too old for the image's CUDA runtime, or when CUDA libs in
# the image are mismatched. Failing here is much better than spending
# 60s loading a policy onto CPU and then silently serving SmolVLA at
# ~300ms/step. The backend marks the box `error` and the operator can
# spin a fresh one (often a different host = working drivers).
#
# Skip with DRTC_SKIP_CUDA_CHECK=1 only for CPU-only smoke tests.
if [ "${DRTC_SKIP_CUDA_CHECK:-0}" != "1" ]; then
  if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[entrypoint] FATAL: torch.cuda.is_available() == False." >&2
    echo "[entrypoint]   nvidia-smi can see a GPU but torch can't use it." >&2
    echo "[entrypoint]   Usually means host driver is too old for this image's CUDA runtime," >&2
    echo "[entrypoint]   or RunPod's GPU runtime injection failed on this host." >&2
    echo "[entrypoint]   torch.cuda.init() details:" >&2
    python -c "import torch; torch.cuda.init()" 2>&1 | sed 's/^/[entrypoint]     /' >&2 || true
    echo "[entrypoint]   nvidia-smi header:" >&2
    nvidia-smi 2>&1 | head -4 | sed 's/^/[entrypoint]     /' >&2 || true
    echo "[entrypoint]   Stop this box and spin up a fresh one — RunPod will usually" >&2
    echo "[entrypoint]   land you on a different host with working drivers." >&2
    exit 1
  fi
  echo "[entrypoint] torch.cuda.is_available() == True — GPU usable."
fi

# ---------- launch interlatent-serve -------------------------------------
ARGS=(--host "${HOST}" --port "${PORT}" --teleop-port "${TELEOP_PORT}")
if [ -n "${DRTC_WARMUP_POLICY:-}" ]; then
  ARGS+=(--policy "${DRTC_WARMUP_POLICY}")
fi

if [ -n "${INTERLATENT_TELEOP_SECRET:-}" ]; then
  echo "[entrypoint] teleop WS relay enabled on port ${TELEOP_PORT}"
else
  echo "[entrypoint] INTERLATENT_TELEOP_SECRET not set — teleop WS relay disabled"
fi

# Any extra positional args from `docker run` get appended, so users
# can still pass flags directly (e.g. for future interlatent-serve options).
exec python -m interlatent_server.server.app "${ARGS[@]}" "$@"
