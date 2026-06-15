"""Local coordinator — the offline control plane the node polls.

The coordinator is the self-hosted replacement for Interlatent Cloud's
session assignment. It speaks the exact ``/api/v1/nodes/*`` HTTP API the
:mod:`interlatent.node.daemon` already long-polls (pair, heartbeat, poll),
plus an ``/admin/*`` surface the :mod:`interlatent.cli` thin client uses to
register GPU boxes and start/stop inference sessions.

It is *not* in the inference data path — the DRTC link is direct
node↔GPU. See docs/adr/0001-offline-coordinator-control-plane.md.
"""
