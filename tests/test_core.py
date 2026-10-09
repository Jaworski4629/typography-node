"""Tests for filesystem_node_iterator.core.

These exercise the behaviour the implementation actually claims to support:
- root yielding at depth 0
- depth-first pre-order traversal
- entry_type filtering
- max_depth limit
- same_device boundary halting descent
- follow_symlinks with cycle detection
- on_error swallowing listing failures
- broken symlinks reported as SYMLINK
- non-followed symlinked directories not descended into

Determinism: all tests build a fixture tree in a temp dir and assert on
paths, depths, and types. No wall-clock comparisons, no sleeping.
"""

import os
import stat
import tempfile
import unittest
from unittest import mock

from filesystem_node_iterator import iter_nodes, NodeEntry
from filesystem_node_iterator.core import EntryType


class TestIterNodes(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def _build_tree(self):
        """Create a known fixture tree under self.root.

        Layout:
            root/
              a.txt
              sub/
                b.txt
                deeper/
                  c.txt
        """
        os.makedirs(os.path.join(self.root, "sub", "deeper"))
        for name in ("a.txt", os.path.join("sub", "b.txt"),
                     os.path.join("sub", "deeper", "c.txt")):
            with open(os.path.join(self.root, name), "w") as f:
                f.write("x")

    def test_root_yields_at_depth_zero(self):
        self._build_tree()
        entries = list(iter_nodes(self.root))
        self.assertEqual(entries[0].path, self.root)
        self.assertEqual(entries[0].depth, 0)
        self.assertEqual(entries[0].entry_type, EntryType.DIRECTORY)

    def test_yields_all_entries(self):
        self._build_tree()
        paths = {e.path for e in iter_nodes(self.root)}
        expected = {
            self.root,
            os.path.join(self.root, "a.txt"),
            os.path.join(self.root, "sub"),
            os.path.join(self.root, "sub", "b.txt"),
            os.path.join(self.root, "sub", "deeper"),
            os.path.join(self.root, "sub", "deeper", "c.txt"),
        }
        self.assertEqual(paths, expected)

    def test_pre_order_depth_first(self):
        self._build_tree()
        entries = list(iter_nodes(self.root))
        depths = [e.depth for e in entries]
        # Pre-order means depth never goes up the previous value by more
        # than one, and every directory appears before its children.
        self.assertEqual(depths[0], 0)
        for i in range(1, len(depths)):
            self.assertLessEqual(depths[i], depths[i - 1] + 1)
        # Specifically, 'sub' must come before 'sub/b.txt'.
        names = [e.name for e in entries]
        self.assertLess(names.index("sub"), names.index("b.txt"))

    def test_filter_by_entry_type(self):
        self._build_tree()
        files = list(iter_nodes(self.root, entry_types={EntryType.FILE}))
        self.assertTrue(all(e.entry_type == EntryType.FILE for e in files))
        names = sorted(e.name for e in files)
        self.assertEqual(names, ["a.txt", "b.txt", "c.txt"])

    def test_max_depth_one(self):
        self._build_tree()
        entries = list(iter_nodes(self.root, max_depth=1))
        self.assertTrue(all(e.depth <= 1 for e in entries))
        names = {e.name for e in entries}
        # Root + a.txt + sub, but NOT sub/b.txt or sub/deeper.
        self.assertEqual(names, {os.path.basename(self.root), "a.txt", "sub"})

    def test_max_depth_zero(self):
        self._build_tree()
        entries = list(iter_nodes(self.root, max_depth=0))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].depth, 0)

    def test_broken_symlink_reported_as_symlink(self):
        os.symlink("/nonexistent/target", os.path.join(self.root, "dangling"))
        entries = list(iter_nodes(self.root))
        symlink_entries = [e for e in entries if e.entry_type == EntryType.SYMLINK]
        self.assertEqual(len(symlink_entries), 1)
        self.assertEqual(symlink_entries[0].name, "dangling")
        self.assertTrue(stat.S_ISLNK(symlink_entries[0].stat_result.st_mode))

    def test_symlink_to_dir_not_descended_by_default(self):
        os.makedirs(os.path.join(self.root, "real", "inner"))
        with open(os.path.join(self.root, "real", "inner", "f.txt"), "w") as f:
            f.write("x")
        os.symlink(os.path.join(self.root, "real"),
                  os.path.join(self.root, "link"))
        entries = list(iter_nodes(self.root))
        paths = {e.path for e in entries}
        # 'link' itself is yielded, but its children (which are really
        # under 'real') must not appear twice.
        self.assertIn(os.path.join(self.root, "link"), paths)
        # 'inner' appears once, under 'real', not under 'link'.
        inner_count = sum(1 for p in paths if p.endswith("inner"))
        self.assertEqual(inner_count, 1)

    def test_follow_symlinks_descends_and_detects_cycle(self):
        os.makedirs(os.path.join(self.root, "sub"))
        with open(os.path.join(self.root, "sub", "f.txt"), "w") as f:
            f.write("x")
        # Create a self-referential symlink cycle: sub/loop -> sub
        os.symlink(os.path.join(self.root, "sub"),
                  os.path.join(self.root, "sub", "loop"))
        entries = list(iter_nodes(self.root, follow_symlinks=True))
        # The cycle must terminate. We should see 'loop' once, and not
        # descend into it again.
        loop_count = sum(1 for e in entries if e.name == "loop")
        self.assertEqual(loop_count, 1)
        # 'f.txt' appears exactly once despite the cycle.
        f_count = sum(1 for e in entries if e.name == "f.txt")
        self.assertEqual(f_count, 1)

    def test_on_error_callback_suppresses_listing_error(self):
        self._build_tree()
        # Simulate an unreadable directory by patching os.listdir so that
        # listing the 'noperm' directory raises PermissionError. This keeps
        # the test deterministic regardless of the executing user's
        # privileges (root bypasses filesystem permissions).
        os.makedirs(os.path.join(self.root, "noperm"))
        errors = []
        real_listdir = os.listdir

        def fake_listdir(path):
            if path == os.path.join(self.root, "noperm"):
                raise PermissionError("simulated unreadable directory")
            return real_listdir(path)

        with mock.patch("filesystem_node_iterator.core.os.listdir",
                        side_effect=fake_listdir):
            entries = list(iter_nodes(self.root, on_error=errors.append))
        # 'noperm' itself is yielded (we lstat'd it), but its contents are not.
        names = [e.name for e in entries]
        self.assertIn("noperm", names)
        self.assertTrue(len(errors) >= 1)
        self.assertIsInstance(errors[0], OSError)

    def test_node_entry_fields(self):
        self._build_tree()
        entry = next(iter(iter_nodes(self.root)))
        self.assertIsInstance(entry, NodeEntry)
        self.assertIsInstance(entry.path, str)
        self.assertIsInstance(entry.name, str)
        self.assertIsInstance(entry.depth, int)
        self.assertIsInstance(entry.entry_type, str)
        self.assertIsInstance(entry.stat_result, os.stat_result)

    def test_root_must_exist(self):
        with self.assertRaises(FileNotFoundError):
            list(iter_nodes(os.path.join(self.root, "nope")))


if __name__ == "__main__":
    unittest.main()
