"""Pluggable publish destinations ("sinks") for the DRTC episode recorder.

The recorder builds a complete LeRobot v3 dataset on local SSD at
``CloseSession`` (see :mod:`.recorder`). A :class:`DatasetSink` decides
where that finished dataset goes:

- :class:`LocalDirSink` — **merge-on-stop** into one flat canonical
  LeRobot dataset on the GPU server's filesystem, via lerobot's
  ``aggregate_datasets``.
- :class:`S3Sink` — the same merge-on-stop, run against a local mirror of
  the canonical dataset, then re-uploaded to an S3-compatible bucket
  (boto3 + endpoint-url).

Those two are the whole list. The hosted upload inbox that used to sit
behind a third sink is gone with the control plane that served it
(ADR 0039), and with it the last reason publishing ever needed an
account: neither sink here authenticates against anything of ours.

The destination + credentials arrive per-session in ``OpenSession``
metadata (configured on the coordinator) or from ``interlatent-serve``
CLI flags; selection happens in ``transport._maybe_build_recorder``. See
docs/adr/0002-recording-destination-via-session-metadata.md.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from .recorder import RecorderConfig

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Sink protocol
# ----------------------------------------------------------------------


class DatasetSink(Protocol):
    """Where a finished LeRobot dataset is published at session close."""

    def normalize_for_merge(self) -> bool:
        """True iff the recorder must emit a stable, mergeable schema.

        Local/S3 sinks accumulate sessions with ``aggregate_datasets``,
        which rejects mismatched ``fps``/``features``. When True the
        recorder pins the dataset fps to the declared rate and forces the
        ``control_source`` column so successive sessions stay mergeable.
        """
        ...

    async def publish(
        self,
        *,
        episode_id: str,
        dataset_root: Path,
        config: "RecorderConfig",
        recorder: Any = None,
    ) -> None:
        ...


# ----------------------------------------------------------------------
# Local-directory / S3 merge-on-stop helpers
# ----------------------------------------------------------------------

# Per-destination locks so two sessions closing into the same canonical
# dataset can't race the read-modify-swap merge. All sinks run on the
# server's single event loop, so a module-level dict of asyncio.Locks is
# safe.
_DEST_LOCKS: dict[str, asyncio.Lock] = {}


def _dest_lock(dest: Path) -> asyncio.Lock:
    key = str(Path(dest).resolve())
    lock = _DEST_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _DEST_LOCKS[key] = lock
    return lock


def merge_local_dataset(dest: Path, new_root: Path, episode_id: str) -> None:
    """Accumulate the freshly built dataset at ``new_root`` into ``dest``.

    Blocking; call from an executor. ``dest`` is a single flat canonical
    LeRobot dataset (NOT keyed by env_slug — see ADR-0002). If ``dest``
    has no dataset yet, the new one is moved into place. Otherwise the two
    are merged with ``aggregate_datasets`` and atomically swapped in.

    Never drops data: if the merge raises (e.g. ``validate_all_metadata``
    rejects a schema/fps/robot_type mismatch because the destination was
    pointed at two incompatible robots), the episode is written to a
    sibling ``<dest>__<episode_id>/`` and a loud warning is logged.
    """
    dest = Path(dest)
    new_root = Path(new_root)
    if not (dest / "meta" / "info.json").exists():
        # No canonical dataset yet — move the new one into place.
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(new_root), str(dest))
        return

    try:
        _aggregate_swap(dest, new_root)
    except Exception:
        sibling = dest.parent / f"{dest.name}__{episode_id}"
        log.exception(
            "Dataset merge into %s failed; writing this episode to %s instead. "
            "The destination likely holds mixed (incompatible) collections — "
            "point one destination at one robot/policy.",
            dest, sibling,
        )
        if sibling.exists():
            shutil.rmtree(sibling, ignore_errors=True)
        shutil.move(str(new_root), str(sibling))


def _aggregate_swap(dest: Path, new_root: Path) -> None:
    """``aggregate_datasets([dest, new_root]) -> tmp`` then atomic swap.

    ``aggr_root`` must NOT pre-exist (aggregate creates it). The temp
    aggregate is built inside ``dest.parent`` so the final rename is on
    the same filesystem (atomic).
    """
    from lerobot.datasets.aggregate import aggregate_datasets

    parent = dest.parent
    tmp_aggr = parent / f".aggr_{uuid.uuid4().hex}"  # must not exist
    aggregate_datasets(
        repo_ids=["interlatent/_existing", "interlatent/_incoming"],
        aggr_repo_id=f"interlatent/{dest.name}",
        roots=[dest, new_root],
        aggr_root=tmp_aggr,
    )
    bak = parent / f".bak_{uuid.uuid4().hex}"
    shutil.move(str(dest), str(bak))
    try:
        shutil.move(str(tmp_aggr), str(dest))
    except Exception:
        # Restore the original on a failed swap so we never lose the
        # accumulated dataset.
        shutil.move(str(bak), str(dest))
        shutil.rmtree(tmp_aggr, ignore_errors=True)
        raise
    shutil.rmtree(bak, ignore_errors=True)


# ----------------------------------------------------------------------
# Sinks
# ----------------------------------------------------------------------


class LocalDirSink:
    """Merge each session into one flat canonical dataset under ``dest``."""

    def __init__(self, dest: Path | str) -> None:
        self.dest = Path(dest).expanduser()

    def normalize_for_merge(self) -> bool:
        return True

    async def publish(
        self,
        *,
        episode_id: str,
        dataset_root: Path,
        config: "RecorderConfig",
        recorder: Any = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        async with _dest_lock(self.dest):
            await loop.run_in_executor(
                None, merge_local_dataset, self.dest, Path(dataset_root), episode_id
            )
        log.info("Published episode %s to local dataset %s", episode_id, self.dest)


class S3Sink:
    """Merge into a local mirror, then re-upload to an S3-compatible bucket.

    Keeps a local mirror of the canonical dataset at
    ``~/.interlatent/s3-cache/<bucket>/<prefix>/`` so the common
    same-box case never re-downloads. boto3 is imported lazily.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
        mirror_base: Optional[Path] = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        base = Path(mirror_base) if mirror_base else Path.home() / ".interlatent" / "s3-cache"
        self._mirror = base / bucket / (self.prefix or "_root")

    def normalize_for_merge(self) -> bool:
        return True

    @classmethod
    def from_uri(cls, uri: str, **kw: Any) -> "S3Sink":
        """Build from ``s3://bucket/prefix``."""
        rest = uri[len("s3://"):] if uri.startswith("s3://") else uri
        bucket, _, prefix = rest.partition("/")
        return cls(bucket=bucket, prefix=prefix, **kw)

    def _client(self):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for S3 dataset upload. "
                "Install with: pip install 'interlatent-server[s3]'"
            ) from exc
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )

    def _key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    async def publish(
        self,
        *,
        episode_id: str,
        dataset_root: Path,
        config: "RecorderConfig",
        recorder: Any = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        async with _dest_lock(self._mirror):
            await loop.run_in_executor(
                None, self._publish_sync, episode_id, Path(dataset_root)
            )
        log.info(
            "Published episode %s to s3://%s/%s", episode_id, self.bucket, self.prefix
        )

    def _publish_sync(self, episode_id: str, dataset_root: Path) -> None:
        client = self._client()
        # 1. Warm the mirror from S3 if cold (first run on this box).
        if not (self._mirror / "meta" / "info.json").exists():
            self._download_prefix(client)
        # 2. Merge the new episode into the mirror (same flow as local).
        merge_local_dataset(self._mirror, dataset_root, episode_id)
        # 3. Re-upload the merged mirror under the prefix.
        self._upload_dir(client)

    def _download_prefix(self, client) -> None:
        self._mirror.mkdir(parents=True, exist_ok=True)
        paginator = client.get_paginator("list_objects_v2")
        list_prefix = f"{self.prefix}/" if self.prefix else ""
        found = False
        for page in paginator.paginate(Bucket=self.bucket, Prefix=list_prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                rel = key[len(list_prefix):] if list_prefix else key
                if not rel or key.endswith("/"):
                    continue
                target = self._mirror / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(self.bucket, key, str(target))
                found = True
        if found:
            log.info("Warmed S3 mirror from s3://%s/%s", self.bucket, self.prefix)

    def _upload_dir(self, client) -> None:
        for f in sorted(self._mirror.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(self._mirror).as_posix()
            client.upload_file(str(f), self.bucket, self._key(rel))


# ----------------------------------------------------------------------
# Selection from OpenSession metadata / serve flags
# ----------------------------------------------------------------------


def sink_from_metadata(metadata: dict[str, str]) -> Optional[DatasetSink]:
    """Build a Local/S3 sink from ``OpenSession`` metadata, or None.

    Recognized keys (set by the coordinator, see ADR-0002):
    ``output_dir`` | ``s3_uri`` (+ ``s3_endpoint_url``, ``s3_access_key``,
    ``s3_secret_key``, ``s3_region``). Returns None when neither is present
    so the caller can fall back to the serve-level default — the second and
    last step of ADR-0002's precedence now that the inbox tier is gone.
    """
    output_dir = (metadata.get("output_dir") or "").strip()
    s3_uri = (metadata.get("s3_uri") or "").strip()
    if output_dir:
        return LocalDirSink(output_dir)
    if s3_uri:
        return S3Sink.from_uri(
            s3_uri,
            endpoint_url=(metadata.get("s3_endpoint_url") or None),
            access_key=(metadata.get("s3_access_key") or None),
            secret_key=(metadata.get("s3_secret_key") or None),
            region=(metadata.get("s3_region") or None),
        )
    return None


__all__ = [
    "DatasetSink",
    "LocalDirSink",
    "S3Sink",
    "merge_local_dataset",
    "sink_from_metadata",
]
