"""Sen1Floods11 hand-labeled chips with optional aligned terrain channels.

Layout expected under `root` (see scripts/download_sen1floods11.py and
scripts/build_terrain.py):

    data/flood_events/HandLabeled/{S1Hand,LabelHand,JRCWaterHand}/<chip>_<kind>.tif
    splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv
    terrain/<chip>_{DEM,HAND,Slope}.tif

Labels: 1 flood/water, 0 dry, -1 no data (ignored in loss and metrics).
Optional HandLabeled/S1PreHand/<chip>_S1Hand.tif: the pre-event S1 image on the same grid (a link to another
root's S1Hand directory, see slurm/prepare_geoid_cd.sbatch), for change-detection inputs vv_pre, vh_pre.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

HAND_DIR = Path("data/flood_events/HandLabeled")
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}
EVENTS = (
    "Bolivia", "Ghana", "India", "Mekong", "Nigeria", "Pakistan",
    "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA",
)
CHANNELS = ("vv", "vh", "dem", "hand", "slope", "jrc", "vv_pre", "vh_pre")
TERRAIN_SUFFIX = {"dem": "DEM", "slope": "Slope"}
IGNORE = -1


@dataclass(frozen=True)
class Chip:
    chip_id: str

    @property
    def event(self) -> str:
        return self.chip_id.rsplit("_", 1)[0]


def official_split(root: Path, split: str) -> list[Chip]:
    with open(root / "splits/flood_handlabeled" / SPLIT_FILES[split], newline="") as f:
        return [Chip(row[0].removesuffix("_S1Hand.tif")) for row in csv.reader(f) if row]


def all_chips(root: Path) -> list[Chip]:
    return [Chip(p.name.removesuffix("_S1Hand.tif")) for p in sorted((root / HAND_DIR / "S1Hand").glob("*_S1Hand.tif"))]


def leave_one_event_out(root: Path, held_out: str, val_fraction: float = 0.15, seed: int = 0) -> dict[str, list[Chip]]:
    """Test on one event; train/val on a chip-level random split of the others."""
    chips = all_chips(root)
    test = [c for c in chips if c.event == held_out]
    rest = [c for c in chips if c.event != held_out]
    random.Random(seed).shuffle(rest)
    n_val = round(len(rest) * val_fraction)
    return {"train": rest[n_val:], "valid": rest[:n_val], "test": test}


def all_events(root: Path, val_fraction: float = 0.15, seed: int = 0) -> dict[str, list[Chip]]:
    """Train on every event (for testing on an external dataset); validation is a random chip subset."""
    chips = all_chips(root)
    random.Random(seed).shuffle(chips)
    n_val = round(len(chips) * val_fraction)
    return {"train": chips[n_val:], "valid": chips[:n_val]}


def listed_split(root: Path, train_list: str = "train") -> dict[str, list[Chip]]:
    """Chip ids from <root>/splits/<name>.txt, one per line (written by scripts/prepare_geoid.py).

    `train_list` selects an alternative training list, e.g. an event subset from scripts/make_event_subsets.py.
    """
    read = lambda name: [Chip(line.strip()) for line in (root / "splits" / f"{name}.txt").read_text().splitlines() if line.strip()]
    return {"train": read(train_list), "valid": read("valid")}


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def normalize(channel: str, arr: np.ndarray) -> np.ndarray:
    """Fixed, data-independent scaling so every split and event is treated identically."""
    if channel in ("vv", "vh", "vv_pre", "vh_pre"):  # backscatter in dB
        return (np.clip(arr, -30.0, 5.0) + 12.5) / 7.5
    if channel == "dem":  # relative relief within the chip; absolute elevation does not transfer across events
        return np.clip((arr - np.nanmedian(arr)) / 50.0, -3.0, 3.0)
    if channel == "hand":  # metres above nearest drainage, heavy-tailed
        return np.log1p(np.clip(arr, 0.0, None)) / 2.0
    if channel == "slope":  # degrees
        return np.clip(arr, 0.0, 45.0) / 15.0
    if channel == "jrc":  # JRC permanent/seasonal water occurrence mask
        return arr
    raise KeyError(channel)


HAND_MISSING = -1.0  # normalized HAND is log1p(m)/2 >= 0


class Sen1Floods11(Dataset):
    def __init__(
        self,
        root: str | Path,
        chips: list[Chip],
        channels: tuple[str, ...] = ("vv", "vh"),
        crop: int | None = None,
        augment: bool = False,
        hand_dropout: float = 0.0,
    ) -> None:
        self.root = Path(root)
        self.chips = chips
        self.channels = tuple(channels)
        unknown = set(self.channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown channels {unknown}; choose from {CHANNELS}")
        self.crop = crop
        self.augment = augment
        # modality dropout (training only): with this probability per chip the HAND channel is replaced by
        # HAND_MISSING, a value outside the normalized HAND range, so the model cannot lean on HAND alone
        self.hand_dropout = hand_dropout
        if hand_dropout and "hand" not in self.channels:
            raise ValueError("hand_dropout needs a hand channel")

    def __len__(self) -> int:
        return len(self.chips)

    def _load(self, chip: Chip) -> dict[str, np.ndarray]:
        cid, hand_dir, terrain = chip.chip_id, self.root / HAND_DIR, self.root / "terrain"
        s1 = _read(hand_dir / "S1Hand" / f"{cid}_S1Hand.tif")
        raw = {
            "vv": s1[0],
            "vh": s1[1],
            "label": _read(hand_dir / "LabelHand" / f"{cid}_LabelHand.tif")[0],
        }
        # HAND is always loaded when available: the monotonicity loss and plausibility metrics need it
        # even when it is not a model input.
        hand_path = terrain / f"{cid}_HAND.tif"
        if hand_path.exists():
            raw["hand_raw"] = _read(hand_path)[0]
        if {"vv_pre", "vh_pre"} & set(self.channels):
            pre = _read(hand_dir / "S1PreHand" / f"{cid}_S1Hand.tif")
            raw["vv_pre"], raw["vh_pre"] = pre[0], pre[1]
        for ch in self.channels:
            if ch == "jrc":
                raw["jrc"] = _read(hand_dir / "JRCWaterHand" / f"{cid}_JRCWaterHand.tif")[0]
            elif ch in ("dem", "slope"):
                raw[ch] = _read(terrain / f"{cid}_{TERRAIN_SUFFIX[ch]}.tif")[0]
            elif ch == "hand":
                raw["hand"] = raw["hand_raw"]
        return raw

    def __getitem__(self, idx: int) -> dict:
        chip = self.chips[idx]
        raw = self._load(chip)
        x = np.stack([normalize(ch, raw[ch]) for ch in self.channels])
        y = raw["label"]
        missing = ~np.isfinite(x).all(axis=0)
        x = np.nan_to_num(x, nan=0.0)
        if self.hand_dropout and random.random() < self.hand_dropout:
            x[self.channels.index("hand")] = HAND_MISSING
        y = np.where(missing, IGNORE, y)  # pixels without full input are ignored, not guessed
        hand = raw.get("hand_raw", np.full_like(y, np.nan))

        arrays = [x, y[None], hand[None]]
        if self.crop:
            h, w = y.shape
            i, j = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
            arrays = [a[:, i : i + self.crop, j : j + self.crop] for a in arrays]
        if self.augment:
            if random.random() < 0.5:
                arrays = [a[:, :, ::-1] for a in arrays]
            if random.random() < 0.5:
                arrays = [a[:, ::-1, :] for a in arrays]
            k = random.randint(0, 3)
            arrays = [np.rot90(a, k, axes=(1, 2)) for a in arrays]
        x, y, hand = (torch.from_numpy(np.ascontiguousarray(a)) for a in arrays)
        return {"x": x, "y": y.long(), "hand": hand, "chip": chip.chip_id, "event": chip.event}
