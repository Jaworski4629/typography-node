"""Public API for filesystem_node_iterator."""

from .core import NodeEntry, iter_nodes, __version__ as _version

__version__ = _version

__all__ = ["NodeEntry", "iter_nodes", "__version__"]
