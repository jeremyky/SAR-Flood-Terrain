"""How do GEOID-Flood test events overlap with its train and validation splits? (PREREGISTRATIONS.md #11; descriptive)

GEOID-Flood splits by area of interest (AoI), so a test activation can have other AoIs in train or validation, and
AoIs of one activation can cover the same footprint. Reports, for the test activations: how many have AoIs in train,
in validation, in either; and how many test tiles have exactly the footprint (bounds and CRS) of a train or a
validation tile (any activation: tiles lie on a common grid, so a later flood in the same place can share
footprints). Writes the activations to exclude for "unseen by train and validation" as a list file.

    python scripts/split_overlap_audit.py --test $S/data/geoid_test --trainval $S/data/geoid_trainval \
        --write-list $S/data/geoid_trainval/splits/train_valid.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import rasterio


def footprints(root: Path, chips: list[str] | None = None) -> dict[str, tuple]:
    s1 = root / "data/flood_events/HandLabeled/S1Hand"
    names = chips if chips is not None else [p.name.removesuffix("_S1Hand.tif") for p in sorted(s1.glob("*_S1Hand.tif"))]
    out = {}
    for c in names:
        with rasterio.open(s1 / f"{c}_S1Hand.tif") as src:
            out[c] = (str(src.crs), *(round(v, 1) for v in src.bounds))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--trainval", type=Path, required=True)
    p.add_argument("--write-list", type=Path, help="write train + valid chip ids here (input to --unseen-only)")
    args = p.parse_args()
    act = lambda c: c.split("_")[0]  # noqa: E731
    lists = {s: (args.trainval / "splits" / f"{s}.txt").read_text().split() for s in ("train", "valid")}
    acts = {s: {act(c) for c in v} for s, v in lists.items()}
    test = footprints(args.test)
    tacts = {act(c) for c in test}
    print(f"test: {len(test)} tiles, {len(tacts)} activations")
    print(f"  activations with AoIs in train: {len(tacts & acts['train'])}; in validation: {len(tacts & acts['valid'])}; "
          f"in validation but not train: {sorted((tacts & acts['valid']) - acts['train'])}")
    print(f"  unseen by train: {len(tacts - acts['train'])}; unseen by train and validation: "
          f"{len(tacts - acts['train'] - acts['valid'])}")
    for s in ("train", "valid"):
        fp = footprints(args.trainval, lists[s])  # all tiles: other activations can cover the same place
        same = set(fp.values())
        dup = [c for c, b in test.items() if b in same]
        print(f"  test tiles with exactly the footprint of a {s} tile: {len(dup)} "
              f"(activations: {len({act(c) for c in dup})}; in activations unseen by {s}: "
              f"{sum(act(c) not in acts[s] for c in dup)})")
    if args.write_list:
        args.write_list.write_text("\n".join(lists["train"] + lists["valid"]) + "\n")
        print(f"wrote {args.write_list}")


if __name__ == "__main__":
    main()
