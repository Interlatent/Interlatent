"""DRTC wire format. Source of truth: messages.proto.

Regenerate stubs with proto/gen_proto.sh.
"""

from . import messages_pb2, messages_pb2_grpc  # noqa: F401
from .timestamps import ControlClock  # noqa: F401
