#!/usr/bin/env python3
"""Download, verify and extract the D3P-JP dataset from its Hugging Face repository.

    python scripts/download_data.py                 # core archive: everything the default configuration needs
    python scripts/download_data.py --panels        # + inputs of the sensitivity experiments (13 GB)
    python scripts/download_data.py --dest /big/disk/d3p-jp      # then: export JPDH_DATA_ROOT=/big/disk/d3p-jp

The archives are listed, with their sizes and SHA-256 checksums, in the
repository's ``MANIFEST.json``.  Downloads resume where they stopped; an
archive whose checksum does not match is refused; an archive already installed
(same checksum) is skipped.  Only the Python standard library is used, so this
runs before the model's environment exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

#: the dataset repository; MANIFEST.json and the archives live under this URL
DATASET_URL = "https://huggingface.co/datasets/tristhas/D3P-JP/resolve/main"
REPO = Path(__file__).resolve().parents[1]
CHUNK = 1 << 20
INSTALLED = ".installed.json"
HEADERS = {"User-Agent": "jpdh-download"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1000 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path, size: int) -> None:
    """Resumable download of ``url`` to ``dst`` (expected ``size`` bytes)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = dst.stat().st_size if dst.exists() else 0
    if have == size:
        print(f"  {dst.name}: already downloaded ({human(have)})")
        return
    if have > size:
        dst.unlink()
        have = 0
    headers = dict(HEADERS)
    if have:
        headers["Range"] = f"bytes={have}-"
        print(f"  {dst.name}: resuming at {human(have)}")
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
    except urllib.error.HTTPError as e:
        if e.code == 416:                      # already complete
            return
        raise
    mode = "ab" if have and resp.status == 206 else "wb"
    done = have if mode == "ab" else 0
    with open(dst, mode) as f:
        while chunk := resp.read(CHUNK):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {dst.name}: {human(done)} / {human(size)} ({100 * done / size:5.1f}%)",
                  end="", flush=True)
    print()


def extract(tar_path: Path, dest: Path) -> None:
    print(f"  extracting {tar_path.name} -> {dest}")
    with tarfile.open(tar_path, "r") as tar:
        try:
            tar.extractall(dest, filter="data")   # rejects absolute paths and '..'
        except TypeError:                          # Python < 3.12
            for m in tar.getmembers():
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    raise RuntimeError(f"unsafe path in archive: {m.name}")
            tar.extractall(dest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=os.environ.get("JPDH_DATA_ROOT", str(REPO / "data")),
                   help="dataset directory (default: $JPDH_DATA_ROOT or <repository>/data)")
    p.add_argument("--panels", action="store_true",
                   help="also fetch the inputs of the sensitivity experiments (13 GB)")
    args = p.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest / "_downloads"
    req = urllib.request.Request(f"{DATASET_URL}/MANIFEST.json", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        manifest = json.load(r)
    archives = manifest["archives"]
    wanted = ["core"] + (["panels"] if args.panels else [])
    state_file = dest / INSTALLED
    installed = json.loads(state_file.read_text()) if state_file.exists() else {}
    todo = [w for w in wanted if installed.get(w) != archives[w]["sha256"]]
    total = sum(archives[w]["bytes"] for w in todo)
    print(f"{manifest['title']}, version {manifest['version']} ({manifest['snapshot']})")
    print(f"{len(todo)} archive(s) to fetch, {human(total)} -> {dest}\n")

    for name in todo:
        a = archives[name]
        tar_path = staging / a["file"]
        download(f"{DATASET_URL}/{a['file']}", tar_path, a["bytes"])
        got = sha256(tar_path)
        if got != a["sha256"]:
            print(f"error: checksum mismatch for {a['file']} (got {got}); delete it and retry",
                  file=sys.stderr)
            return 4
        print(f"  {a['file']}: sha256 ok")
        extract(tar_path, dest)
        tar_path.unlink()
        installed[name] = a["sha256"]
        state_file.write_text(json.dumps(installed, indent=1))

    if staging.exists() and not any(staging.iterdir()):
        staging.rmdir()
    print(f"\ndone: {', '.join(wanted)} in {dest}")
    if dest != (REPO / "data").resolve():
        print(f"set   export JPDH_DATA_ROOT={dest}")
    print("check python scripts/verify_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
