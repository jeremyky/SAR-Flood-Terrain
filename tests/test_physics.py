import pytest
import numpy as np
import torch

from floodseg.losses import bce_dice, hand_ceiling, hand_monotonicity
from floodseg.metrics import Tracker, confusion, plausibility, scores
from floodseg.model import UNet


def logits_from_probs(p):
    p = torch.tensor(p, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))[None, None]


def test_monotonicity_penalizes_higher_pixel_being_wetter():
    hand = torch.tensor([[0.0, 5.0]])[None, None]
    bad = hand_monotonicity(logits_from_probs([[0.1, 0.9]]), hand, margin=0.0, min_delta=1.0)
    good = hand_monotonicity(logits_from_probs([[0.9, 0.1]]), hand, margin=0.0, min_delta=1.0)
    assert bad.item() > 0.7
    assert good.item() == 0.0


def test_monotonicity_ignores_small_deltas_and_nan():
    logits = logits_from_probs([[0.1, 0.9, 0.9]])
    hand = torch.tensor([[0.0, 0.5, float("nan")]])[None, None]
    assert hand_monotonicity(logits, hand, margin=0.0, min_delta=1.0).item() == 0.0


def test_monotonicity_has_gradient():
    logits = logits_from_probs([[0.1, 0.9]]).requires_grad_()
    hand_monotonicity(logits, torch.tensor([[0.0, 5.0]])[None, None], margin=0.0).backward()
    g = logits.grad[0, 0, 0]
    assert g[1] > 0 and g[0] < 0  # push high-HAND pixel down, low-HAND pixel up


def test_hand_ceiling_only_counts_high_pixels():
    logits = logits_from_probs([[0.9, 0.2]])
    hand = torch.tensor([[1.0, 30.0]])[None, None]
    assert abs(hand_ceiling(logits, hand, threshold=15.0).item() - 0.2) < 1e-3


def test_bce_dice_ignores_nodata():
    y = torch.tensor([[1, -1]])[None, None]
    a = bce_dice(logits_from_probs([[0.9, 0.9]]), y)
    b = bce_dice(logits_from_probs([[0.9, 0.1]]), y)
    assert torch.isclose(a, b)


def test_confusion_and_scores():
    pred = np.array([[1, 1, 0, 0]])
    y = np.array([[1, 0, 1, -1]])
    cm = confusion(pred, y)
    assert cm.tolist() == [1, 1, 1, 0]
    assert scores(cm)["iou"] == 1 / 3


def test_plausibility_counts_violations():
    hand = np.array([[0.0, 5.0, 20.0]])
    out = plausibility(np.array([[0, 1, 1]]), hand, hand_threshold=15.0, stride=1)
    assert out["high_hand_flood_fraction"] == 0.5
    assert out["monotonicity_violation_rate"] == 0.5  # (0,5) violates; (5,20) both flooded


def test_tracker_summary_runs():
    t = Tracker()
    t.add("Ghana_1", "Ghana", np.ones((4, 4), bool), np.ones((4, 4), int), np.zeros((4, 4)))
    t.add("Ghana_2", "Ghana", np.zeros((4, 4), bool), np.zeros((4, 4), int), np.zeros((4, 4)))
    s = t.summary()
    assert s["total"]["iou"] == 1.0
    assert s["dry_chip_false_positive_rate"] == 0.0


def test_unet_shape_on_odd_size():
    out = UNet(3, base=8, depth=3)(torch.zeros(2, 3, 100, 100))
    assert out.shape == (2, 1, 100, 100)


def test_monotonicity_stride_compares_distant_pixels():
    hand = torch.tensor([[0.0, 0.0, 0.0, 5.0]])[None, None]  # 30 m cells: only pixels 3 apart differ
    logits = logits_from_probs([[0.1, 0.5, 0.5, 0.9]])
    assert hand_monotonicity(logits, hand, margin=0.0, min_delta=1.0, stride=1).item() > 0  # pair (2,3)
    assert hand_monotonicity(logits, hand, margin=0.0, min_delta=1.0, stride=3).item() == pytest.approx(0.8, abs=1e-5)


def test_hand_dropout_replaces_only_hand_channel():
    from floodseg.data import HAND_MISSING, Chip, Sen1Floods11

    class Fake(Sen1Floods11):
        def _load(self, chip):
            r = np.random.default_rng(0)
            return {"vv": r.normal(-12, 3, (8, 8)).astype(np.float32), "vh": r.normal(-18, 3, (8, 8)).astype(np.float32),
                    "hand": np.full((8, 8), 3.0, np.float32), "hand_raw": np.full((8, 8), 3.0, np.float32),
                    "label": np.ones((8, 8), np.float32)}

    keep, drop = (Fake(".", [Chip("Spain_1_S1Hand")], channels=("vv", "vh", "hand"), hand_dropout=p)[0] for p in (0.0, 1.0))
    xk, xd = (np.asarray(s["x"]) for s in (keep, drop))
    assert np.allclose(xk[:2], xd[:2])
    assert (xk[2] >= 0).all() and (xd[2] == HAND_MISSING).all()
    with pytest.raises(ValueError):
        Fake(".", [Chip("Spain_1_S1Hand")], channels=("vv", "vh"), hand_dropout=0.5)


def test_hand_perturbations_change_only_hand_and_keep_scored_pixels():
    from floodseg.data import Chip, Sen1Floods11

    class Fake(Sen1Floods11):
        def _load(self, chip):
            r = np.random.default_rng(len(chip.chip_id))
            h = r.uniform(0, 30, (16, 16)).astype(np.float32)
            h[0, 0] = np.nan
            return {"vv": r.normal(-12, 3, (16, 16)).astype(np.float32), "vh": r.normal(-18, 3, (16, 16)).astype(np.float32),
                    "hand": h, "hand_raw": h, "label": np.ones((16, 16), np.float32)}

        def _load_hand(self, chip):
            return self._load(chip)["hand_raw"]

    chips = [Chip("Spain_1_S1Hand"), Chip("Ghana_2_S1Hand"), Chip("India_33_S1Hand")]
    base = Fake(".", chips, channels=("vv", "vh", "hand"))[0]
    for p in ("shift:3", "blur:2", "swap"):
        s = Fake(".", chips, channels=("vv", "vh", "hand"), hand_perturb=p)[0]
        assert np.allclose(np.asarray(s["x"])[:2], np.asarray(base["x"])[:2])          # SAR untouched
        assert not np.allclose(np.asarray(s["x"])[2], np.asarray(base["x"])[2])        # HAND changed
        assert np.array_equal(np.asarray(s["y"]), np.asarray(base["y"]))              # same scored pixels
        assert np.isfinite(np.asarray(s["x"])).all()
    d = Fake(".", chips, channels=("vv", "vh", "hand"), hand_perturb="swap")
    assert all(chips[d._donor[i]].event != chips[i].event for i in range(len(chips)))
    with pytest.raises(ValueError):
        Fake(".", chips, channels=("vv", "vh"), hand_perturb="swap")
