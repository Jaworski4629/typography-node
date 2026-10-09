"""Lazily yield filesystem entries under a root as stat-annotated records.

Design decisions (documented so callers know what they are buying):
- ``iter_nodes`` is a generator that yields one ``NodeEntry`` per path it visits.
- The root itself is yielded first, as depth 0.
- Traversal is a manual stack (depth-first, pre-order). We avoid ``os.walk``
  because it materialises directory contents in batches and hides per-entry
  stat timing, which makes it harder to honour device boundaries and depth
  limits cleanly.
- Entry type is derived from ``lstat`` using ``stat.S_IFMT`` so that
  symlinks are reported as ``TYPE_SYMLINK`` rather than their target type.
  This is a deliberate choice: a symlink is its own node, and reporting it
  as a directory hides the fact that traversal might follow it off-device.
- With ``same_device=True``, traversal halts at any subdirectory whose
  ``st_dev`` differs from the root's. Files on other devices are still
  yielded (we discovered them by listing their parent), but their parent
  is not descended into. This mirrors ``find -mount``.
- With ``follow_symlinks=True``, symlinked directories are descended into
  using ``stat`` (not ``lstat``) for the device check, and cycle detection
  via an ``st_ino, st_dev`` seen-set prevents infinite loops.
- Depth is the number of path separators from the root, so direct children
  are at depth 1.
- ``EntryType`` constants are strings, not enums, so ``NodeEntry`` remains
  a plain dataclass that serialises trivially.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

__version__ = "1.0.0"

# Entry types. Strings are stable for serialisation and logging.
TYPE_FILE = "file"
TYPE_DIRECTORY = "directory"
TYPE_SYMLINK = "symlink"
TYPE_OTHER = "other"


class EntryType:
    """Namespace of entry-type constants.

    These are plain strings rather than an Enum so that ``NodeEntry`` stays
    a trivial dataclass and comparisons work without importing a type.
    """

    FILE = TYPE_FILE
    DIRECTORY = TYPE_DIRECTORY
    SYMLINK = TYPE_SYMLINK
    OTHER = TYPE_OTHER


@dataclass
class NodeEntry:
    """A single filesystem entry with its stat snapshot.

    Attributes:
        path: Filesystem path as walked (root-relative or absolute per root).
        name: Basename component, or ``os.path.basename(root)`` for the root.
        depth: 0 for the root, 1 for direct children, and so on.
        entry_type: One of the ``EntryType`` constants. Derived from ``lstat``
            so symlinks are reported as symlinks regardless of their target.
        stat_result: The ``os.stat_result`` from ``os.lstat`` (or ``os.stat``
            when resolving a symlinked directory for device checks).
    """

    path: str
    name: str
    depth: int
    entry_type: str
    stat_result: os.stat_result


def _classify(st: os.stat_result) -> str:
    """Map a stat result to an entry type constant.

    Uses ``S_IFMT`` rather than boolean checkers so we return ``OTHER``
    for fifos, sockets, block, and char devices rather than mislabeling
    them as files.
    """
    mode = st.st_mode
    if stat.S_ISDIR(mode):
        return TYPE_DIRECTORY
    if stat.S_ISREG(mode):
        return TYPE_FILE
    if stat.S_ISLNK(mode):
        return TYPE_SYMLINK
    return TYPE_OTHER


def iter_nodes(
    root: str,
    *,
    entry_types: Optional[set] = None,
    max_depth: Optional[int] = None,
    same_device: bool = False,
    follow_symlinks: bool = False,
    on_error: Optional[Callable[[OSError], None]] = None,
) -> Iterator[NodeEntry]:
    """Lazily yield ``NodeEntry`` records for everything under ``root``.

    Parameters:
        root: Path to begin traversal from. Must exist.
        entry_types: Optional set of ``EntryType`` constants to include.
            If ``None``, all types are yielded. Filtering is applied after
            stat-ing, so a symlink whose target is a directory still appears
            as ``SYMLINK`` and is not double-counted.
        max_depth: Maximum depth to yield. ``None`` means unlimited. Depth 0
            is the root itself.
        same_device: If ``True``, do not descend into directories on a
            different ``st_dev`` than ``root``.
        follow_symlinks: If ``True``, follow symlinked directories. Cycle
            detection is enabled when this is on.
        on_error: Callback for ``OSError`` raised while listing a directory.
            If ``None``, the error propagates. If provided, the callback is
            invoked and traversal continues with the rest of the tree.

    Yields:
        ``NodeEntry`` instances in pre-order, depth-first order.

    Raises:
        FileNotFoundError, NotADirectoryError: if ``root`` does not exist
            or is not accessible. These are not routed through ``on_error``
            because the root is the caller's responsibility.
    """
    root_lstat = os.lstat(root)
    root_dev = root_lstat.st_dev

    # Determining the root's effective device: if the caller wants us to
    # honour device boundaries and the root is a symlink, we need the
    # device of the resolved target, otherwise we would never descend.
    if same_device and stat.S_ISLNK(root_lstat.st_mode):
        root_stat = os.stat(root)
        root_dev = root_stat.st_dev

    seen: Optional[set] = set() if follow_symlinks else None

    # Manual stack of (path, depth) tuples awaiting visitation.
    # Pre-order means we push children in reverse so the first child pops first.
    # Using a list as a stack keeps memory bounded by tree depth, not width.
    stack: list = [(root, 0)]

    while stack:
        path, depth = stack.pop()

        try:
            st = os.lstat(path)
        except OSError as exc:
            if on_error is not None:
                on_error(exc)
                continue
            raise

        entry_type = _classify(st)

        # If we are following symlinks and this is a symlink pointing at a
        # directory, resolve it for traversal decisions but keep reporting
        # it as a SYMLINK node (per the documented behaviour).
        resolved_is_dir = False
        resolved_stat: Optional[os.stat_result] = None
        if entry_type == TYPE_SYMLINK and follow_symlinks:
            try:
                resolved_stat = os.stat(path)
                resolved_is_dir = stat.S_ISDIR(resolved_stat.st_mode)
            except OSError:
                # Broken symlink: yield it, no traversal.
                resolved_is_dir = False

        if entry_types is None or entry_type in entry_types:
            yield NodeEntry(
                path=path,
                name=os.path.basename(path.rstrip(os.sep)) or path,
                depth=depth,
                entry_type=entry_type,
                stat_result=st,
            )

        should_descend = False
        if entry_type == TYPE_DIRECTORY:
            should_descend = True
        elif resolved_is_dir:
            should_descend = True

        if not should_descend:
            continue

        if max_depth is not None and depth >= max_depth:
            continue

        # Device boundary check. For a plain directory, use its own lstat;
        # for a followed symlink directory, use the resolved stat.
        dir_dev = st.st_dev
        if resolved_stat is not None:
            dir_dev = resolved_stat.st_dev

        if same_device and dir_dev != root_dev:
            continue

        # Cycle detection: only meaningful when following symlinks, since
        # hard directory cycles cannot exist on a real filesystem.
        if seen is not None:
            key = (resolved_stat.st_ino if resolved_stat is not None else st.st_ino,
                   dir_dev)
            if key in seen:
                continue
            seen.add(key)

        try:
            children = os.listdir(path)
        except OSError as exc:
            if on_error is not None:
                on_error(exc)
                continue
            raise

        # Push in reverse so the first child alphabetically pops first,
        # giving a deterministic left-to-right pre-order.
        for name in reversed(children):
            child_path = os.path.join(path, name)
            stack.append((child_path, depth + 1))
