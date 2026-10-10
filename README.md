# filesystem_node_iterator

Lazily yields every filesystem entry under a root path as a `NodeEntry` record with its `lstat` snapshot, optional filtering by entry type and depth, and an optional same-device boundary to stop cross-filesystem traversal.

## Usage

```python
from filesystem_node_iterator import iter_nodes, NodeEntry
from filesystem_node_iterator.core import EntryType

for entry in iter_nodes("/", entry_types={EntryType.FILE}, max_depth=3):
    print(entry.path, entry.depth, entry.entry_type)
```

The exported names are `iter_nodes` and `NodeEntry` from the top-level package, and the `EntryType` constants (`FILE`, `DIRECTORY`, `SYMLINK`, `OTHER`) from `filesystem_node_iterator.core`.

## Why

Walking a filesystem with `os.walk` gives you batches of names per directory and hides the per-entry `stat` timing. When you need to (a) filter by entry type as classified by `lstat` (so symlinks stay symlinks), (b) cap depth, and (c) refuse to cross mount points — all while holding a real `stat_result` per entry — a dedicated generator is simpler than post-processing `walk` output.

The trade-off: this library does directory listing itself with `os.listdir`, so it does not offer `os.walk`'s `topdown`/`bottomup` toggle or on-the-fly pruning via mutating `dirnames`. Pruning is done by passing `max_depth` or by filtering `entry_types`.

## Edge cases you will hit

- **Symlinks to directories** are reported as `EntryType.SYMLINK` and are *not* descended into unless you pass `follow_symlinks=True`. This is deliberate: descending silently would yield duplicate paths and could cross devices without warning.
- **`follow_symlinks=True`** enables cycle detection via `(st_ino, st_dev)`. A self-referential symlink cycle terminates rather than looping forever.
- **`same_device=True`** compares each directory's `st_dev` against the root's. Directories on a different device are yielded (we discovered them by listing their parent) but are not descended into, mirroring `find -mount`.
- **Unreadable directories** raise `OSError` unless you pass an `on_error` callback, which receives the exception and lets traversal continue.
