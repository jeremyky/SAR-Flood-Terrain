"""Write training lists that keep only a random subset of events, for a learning curve over event count.

    python scripts/make_event_subsets.py --root /scratch/$USER/floodseg/data/geoid_trainval --sizes 10 20 40 80 --draws 3

Writes <root>/splits/train_e<size>_r<draw>.txt. Each draw uses its own random event subset, so
seed-to-seed spread in the learning curve includes which events were picked.

With --tiles T, every subset is cut to exactly T tiles spread as evenly as possible over its events
(train_t<T>_e<size>_r<draw>.txt), so the number of events varies while the amount of data does not.
Event subsets are redrawn until they hold at least T tiles.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--first-draw", type=int, default=0, help="start numbering draws here (to add draws later)")
    parser.add_argument("--tiles", type=int, default=None, help="fixed tile budget per subset (data-volume control)")
    args = parser.parse_args()

    by_event: dict[str, list[str]] = defaultdict(list)
    for chip in (args.root / "splits" / "train.txt").read_text().split():
        by_event[chip.rsplit("_", 1)[0]].append(chip)
    events = sorted(by_event)
    print(f"{len(events)} events, {sum(map(len, by_event.values()))} tiles in train.txt")
    for size in args.sizes:
        for draw in range(args.first_draw, args.first_draw + args.draws):
            rng = random.Random(1000 * size + draw + (7919 * args.tiles if args.tiles else 0))
            picked = rng.sample(events, size)
            if args.tiles:
                while sum(len(by_event[e]) for e in picked) < args.tiles:
                    picked = rng.sample(events, size)
                chips = even_cut(sorted(picked), by_event, args.tiles, rng)
                name = f"train_t{args.tiles}_e{size}_r{draw}.txt"
            else:
                chips = [c for e in sorted(picked) for c in by_event[e]]
                name = f"train_e{size}_r{draw}.txt"
            (args.root / "splits" / name).write_text("\n".join(chips) + "\n")
            n_ev = len({c.rsplit("_", 1)[0] for c in chips})
            print(f"{name}: {len(chips)} tiles from {n_ev} events")


def even_cut(events: list[str], by_event: dict[str, list[str]], budget: int, rng: random.Random) -> list[str]:
    """Take `budget` tiles round-robin over events (shuffled within each event), so small events give all they have."""
    pools = {e: rng.sample(by_event[e], len(by_event[e])) for e in events}
    chips: list[str] = []
    while len(chips) < budget:
        for e in events:
            if pools[e] and len(chips) < budget:
                chips.append(pools[e].pop())
    return chips

if __name__ == "__main__":
    main()
