"""Download this model's weights from its GitHub release and check them against weights.sha256.

    python download_weights.py

Standard library only. Files already present with the right checksum are skipped.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/TechLuddite/Luddite-Decisions-Phi-4-mini/releases/download/v0.1/"
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    for line in (HERE / "weights.sha256").read_text().splitlines():
        digest, name = line.split()
        dest = HERE / name
        if dest.exists() and sha256(dest) == digest:
            print(f"{name}: present")
            continue
        print(f"{name}: downloading")
        urllib.request.urlretrieve(URL + name, dest)
        if sha256(dest) != digest:
            dest.unlink()
            print(f"{name}: checksum mismatch, removed", file=sys.stderr)
            return 1
    print("weights ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
