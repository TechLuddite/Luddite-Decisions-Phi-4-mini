"""Repository checks that need no weights and no third-party package.

    python3 -m unittest discover -s tests -v

They cover what a pull request can break without anyone running the model:
a JSON file that no longer parses, a Python file that no longer compiles, and
a weights.sha256 that no longer agrees with the files it describes.
"""

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"^([0-9a-f]{64})  (\S+)$")


def tracked(suffix):
    return sorted(p for p in ROOT.rglob(f"*{suffix}") if ".git" not in p.relative_to(ROOT).parts)


def checksums():
    return dict(reversed(LINE.match(line).groups()) for line in (ROOT / "weights.sha256").read_text().splitlines())


class RepositoryChecks(unittest.TestCase):
    def test_json_files_parse(self):
        files = tracked(".json")
        self.assertTrue(files)
        for path in files:
            with self.subTest(path=path.name):
                json.loads(path.read_text(encoding="utf-8"))

    def test_python_files_compile(self):
        files = tracked(".py")
        self.assertTrue(files)
        for path in files:
            with self.subTest(path=str(path.relative_to(ROOT))):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_checksum_file_is_well_formed(self):
        lines = (ROOT / "weights.sha256").read_text().splitlines()
        self.assertTrue(lines)
        for line in lines:
            with self.subTest(line=line):
                self.assertRegex(line, LINE)
        names = [LINE.match(line).group(2) for line in lines]
        self.assertEqual(len(names), len(set(names)), "a file is listed twice")

    def test_tracked_index_matches_its_checksum(self):
        # The index is the one file that is both in the repository and in weights.sha256.
        index = ROOT / "model.safetensors.index.json"
        self.assertEqual(hashlib.sha256(index.read_bytes()).hexdigest(), checksums()[index.name])

    def test_every_shard_in_the_index_has_a_checksum(self):
        index = json.loads((ROOT / "model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
        self.assertTrue(shards)
        self.assertLessEqual(shards, set(checksums()))


if __name__ == "__main__":
    unittest.main()
