"""Download EHM correspondence data from a fixed upstream revision."""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fetch(item, destination):
    target = destination / item["path"]
    with urllib.request.urlopen(item["url"], timeout=45) as response:
        data = response.read()
    digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if digest != item["git_blob_sha1"]:
        raise ValueError(f"Upstream checksum mismatch: {item['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != data:
        raise FileExistsError(f"Refusing to replace a different file: {target}")
    target.write_bytes(data)
    return item["path"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "assets/support_manifest.json").read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(lambda item: fetch(item, args.assets), manifest):
            print(f"Verified {name}")
