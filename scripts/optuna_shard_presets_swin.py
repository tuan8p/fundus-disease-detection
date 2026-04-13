"""
optuna_shard_presets_swin.py
-----------------------------
Các case HPO cố định cho **SwinV2-Base-384**.

PHÂN CHIA: 9 shard × 3 case = 27 case — mỗi máy chạy đúng 1 shard.
  Shard 0-1 : LR=5e-5 / 3-8e-5 (baseline, T_0 / wd biến thể)
  Shard 2   : Warmup dài wu=3, LR trung bình
  Shard 3-4 : LR nhỏ (1-2e-5), backbone_lr thấp
  Shard 5-6 : LR cao (1e-4), wd / blr biến thể
  Shard 7   : AdamW beta variations (beta1 và beta2)
  Shard 8   : BACKBONE_LR_SCALE biên (0.02 ↔ 0.20)

Ép cứng bởi run_optuna_shard_swin.py:
  BATCH_SIZE=8  |  FREEZE_STRATEGY=none  |  AUGMENT_VERSION=v1
"""
from __future__ import annotations

SHARD_PRESETS: list[list[dict]] = [

    # ── Shard 0: LR=5e-5, T_0 ngắn-trung, wd=1e-4 ──────────────────────────
    [
        {
            "_label": "sw0_lr5e5_t6",
            "LR": 5e-5, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 6, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw0_lr5e5_t10",
            "LR": 5e-5, "BACKBONE_LR_SCALE": 0.05,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 10, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw0_lr5e5_wd2e4_t8",
            "LR": 5e-5, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 2e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
    ],

    # ── Shard 1: LR=3e-5 / 8e-5 / 6e-5 (biến thể LR trung bình) ─────────────
    [
        {
            "_label": "sw1_lr3e5_t8",
            "LR": 3e-5, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 2e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw1_lr8e5_t6_wd2e4",
            "LR": 8e-5, "BACKBONE_LR_SCALE": 0.08,
            "WEIGHT_DECAY": 2e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 6, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw1_lr6e5_t8",
            "LR": 6e-5, "BACKBONE_LR_SCALE": 0.08,
            "WEIGHT_DECAY": 2e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
    ],

    # ── Shard 2: Warmup dài wu=3, LR trung bình ──────────────────────────────
    [
        {
            "_label": "sw2_wu3_lr5e5_wd3e4",
            "LR": 5e-5, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 3e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 3, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw2_wu3_lr4e5",
            "LR": 4e-5, "BACKBONE_LR_SCALE": 0.05,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 3, "T_0": 9, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw2_wu3_wd5e4",
            "LR": 3e-5, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 5e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 3, "T_0": 9, "HEAD_DROPOUT": 0.0,
        },
    ],

    # ── Shard 3: LR nhỏ (1e-5 / 5e-6), backbone_lr rất thấp ─────────────────
    [
        {
            "_label": "sw3_lr1e5_blr001",
            "LR": 1e-5, "BACKBONE_LR_SCALE": 0.01,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 3, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw3_lr1e5_wd5e4",
            "LR": 1e-5, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 5e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw3_lr5e6_blr002",
            "LR": 5e-6, "BACKBONE_LR_SCALE": 0.02,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 3, "T_0": 9, "HEAD_DROPOUT": 0.0,
        },
    ],

    # ── Shard 4: LR nhỏ-trung (2e-5), backbone_lr thấp-cao ───────────────────
    [
        {
            "_label": "sw4_lr2e5_blr005",
            "LR": 2e-5, "BACKBONE_LR_SCALE": 0.05,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 3, "T_0": 9, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw4_lr2e5_blr020",
            "LR": 2e-5, "BACKBONE_LR_SCALE": 0.20,
            "WEIGHT_DECAY": 5e-5, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 10, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw4_lr5e6_wd5e5",
            "LR": 5e-6, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 5e-5, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 3, "T_0": 9, "HEAD_DROPOUT": 0.0,
        },
    ],

    # ── Shard 5: LR cao (1e-4), T_0 và wd biến thể ───────────────────────────
    [
        {
            "_label": "sw5_lr1e4_t8",
            "LR": 1e-4, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw5_lr1e4_blr015",
            "LR": 1e-4, "BACKBONE_LR_SCALE": 0.15,
            "WEIGHT_DECAY": 5e-5, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw5_lr1e4_wd3e4",
            "LR": 1e-4, "BACKBONE_LR_SCALE": 0.08,
            "WEIGHT_DECAY": 3e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
    ],

    # ── Shard 6: LR cao + wd cao / blr thấp ──────────────────────────────────
    [
        {
            "_label": "sw6_lr1e4_wd5e4",
            "LR": 1e-4, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 5e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw6_lr7e5_t10",
            "LR": 7e-5, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 10, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw6_lr1e4_blr001_wd1e4",
            "LR": 1e-4, "BACKBONE_LR_SCALE": 0.01,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
    ],

    # ── Shard 7: AdamW beta variations ───────────────────────────────────────
    [
        {
            "_label": "sw7_b1_091_lr5e5",
            "LR": 5e-5, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.91, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw7_b2_9995_lr5e5",
            "LR": 5e-5, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.9995,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw7_b1_092_b2_9999",
            "LR": 6e-5, "BACKBONE_LR_SCALE": 0.1,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.92, "ADAMW_BETA2": 0.9999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 9, "HEAD_DROPOUT": 0.0,
        },
    ],

    # ── Shard 8: BACKBONE_LR_SCALE biên (0.02 / 0.15 / 0.05) ────────────────
    [
        {
            "_label": "sw8_blr002_lr5e5",
            "LR": 5e-5, "BACKBONE_LR_SCALE": 0.02,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw8_blr015_lr5e5",
            "LR": 5e-5, "BACKBONE_LR_SCALE": 0.15,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 8, "HEAD_DROPOUT": 0.0,
        },
        {
            "_label": "sw8_blr005_lr8e5",
            "LR": 8e-5, "BACKBONE_LR_SCALE": 0.05,
            "WEIGHT_DECAY": 1e-4, "ADAMW_BETA1": 0.9, "ADAMW_BETA2": 0.999,
            "EPOCHS": 15, "WARMUP_EPOCHS": 2, "T_0": 7, "HEAD_DROPOUT": 0.0,
        },
    ],
]

NUM_SHARDS = len(SHARD_PRESETS)  # = 9

# ── Sanity checks ──────────────────────────────────────────────────────────────
_all_labels = [c["_label"] for shard in SHARD_PRESETS for c in shard]
_total = len(_all_labels)
assert _total == 27,        f"Kỳ vọng 27 case, có {_total}"
assert len(_all_labels) == len(set(_all_labels)), "duplicate _label in swin presets"
assert NUM_SHARDS == 9,     f"Kỳ vọng 9 shard, có {NUM_SHARDS}"
for _i, _s in enumerate(SHARD_PRESETS):
    assert len(_s) == 3,    f"Shard {_i} phải có 3 case, có {len(_s)}"
    for _c in _s:
        _ep, _wu, _t0 = _c["EPOCHS"], _c["WARMUP_EPOCHS"], _c["T_0"]
        assert _t0 <= _ep - _wu, (
            f"{_c['_label']}: T_0={_t0} > EPOCHS-WARMUP={_ep - _wu}"
        )
