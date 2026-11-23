"""Download the hand-labeled Sen1Floods11 subset (S1, labels, JRC water, splits).

Source: public GCS bucket gs://sen1floods11 (Bonafilia et al., 2020), fetched over HTTPS.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://storage.googleapis.com"
LIST_URL = BUCKET + "/storage/v1/b/sen1floods11/o?prefix={prefix}&fields=items(name,size),nextPageToken&maxResults=1000"
PREFIXES = [
    "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "v1.1/data/flood_events/HandLabeled/LabelHand/",
    "v1.1/data/flood_events/HandLabeled/JRCWaterHand/",
    "v1.1/splits/flood_handlabeled/",
    "v1.1/catalog/sen1floods11_hand_labeled_source/",
]


def list_objects(prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        url = LIST_URL.format(prefix=prefix) + (f"&pageToken={token}" if token else "")
        with urllib.request.urlopen(url) as resp:
            page = json.load(resp)
        items += page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            return items


def fetch(obj: dict, root: Path) -> None:
    dest = root / obj["name"].removeprefix("v1.1/")
    if dest.exists() and dest.stat().st_size == int(obj["size"]):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(f"{BUCKET}/sen1floods11/{obj['name']}", tmp)
    tmp.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    objects = [o for p in PREFIXES for o in list_objects(p) if not o["name"].endswith("/")]
    print(f"{len(objects)} objects, {sum(int(o['size']) for o in objects) / 1e9:.2f} GB")
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda o: fetch(o, args.root), objects))
    print("done")


if __name__ == "__main__":
    main()
