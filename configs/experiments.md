# Experiment matrix

Every variant is `configs/base.yaml` plus overrides. Run each for several seeds
(`seed=0..4`) and on both the official split and leave-one-event-out
(`split.mode=loeo split.held_out=<Event>`).

| name | overrides |
|---|---|
| sar_only | (none) |
| sar_hand | `data.channels=[vv,vh,hand]` |
| sar_terrain | `data.channels=[vv,vh,dem,hand,slope]` |
| sar_mono | `loss.hand_monotonicity.weight=1.0` |
| sar_ceil | `loss.hand_ceiling.weight=1.0` |
| sar_hand_mono | `data.channels=[vv,vh,hand] loss.hand_monotonicity.weight=1.0` |

The HAND post-mask baseline needs no training: every run also reports
`hand_mask` metrics next to `raw`.
