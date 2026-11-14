"""
Pure-Python Git bootstrap (uses *dulwich*, Apache-2.0) for machines without the git executable.

Creates the repository (branch ``main``) and a first commit that honours ``.gitignore``.
When you later install Git you can push this repository as usual (scripts/publish_github.*).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    try:
        from dulwich import porcelain
        from dulwich.ignore import IgnoreFilterManager
        from dulwich.repo import Repo
    except ImportError:
        print("dulwich is missing: pip install dulwich  (or install git: winget install --id Git.Git)")
        return 1

    if (ROOT / ".git").exists():
        repo = Repo(str(ROOT))
        print("Repository already exists.")
    else:
        repo = Repo.init(str(ROOT))
        repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/main")
        print("Initialised empty repository on branch main.")

    ignore = IgnoreFilterManager.from_repo(repo)
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = Path(dirpath).relative_to(ROOT)
        dirnames[:] = [d for d in dirnames if d != ".git" and not ignore.is_ignored(str(rel_dir / d).replace("\\", "/") + "/")]
        for name in filenames:
            rel = name if str(rel_dir) == "." else (rel_dir / name).as_posix()
            if not ignore.is_ignored(rel):
                files.append(rel)

    porcelain.add(repo, paths=[str(ROOT / f) for f in files])
    identity = b"AstroGuard-X <astroguard-x@users.noreply.github.com>"
    try:
        sha = porcelain.commit(repo, message=b"Initial commit: AstroGuard-X", author=identity, committer=identity)
        print(f"Committed {len(files)} files as {sha.decode()[:10]}.")
    except Exception as exc:  # nothing to commit
        print(f"No commit created: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
