#!/usr/bin/env python3
"""
run_optuna_shard_swin.py
------------------------
Chạy các case HPO cố định cho SwinV2-Base-384 theo shard (0..6).
27 case (7×4 - 1): shard 0-5 có 4 case, shard 6 có 3 case.

Cố định trong runner:
  MODEL_TYPE      = swinv2_base_384
  IMAGE_SIZE      = 384
  BATCH_SIZE      = 8
  FREEZE_STRATEGY = none
  HEAD_TYPE       = ordinal

W&B logging:
  - Mỗi case  : 1 run riêng (train/loss, train/qwk, val/loss, val/qwk, lr/... theo epoch)
                group = "swin_shard{N}"  →  so sánh ngang hàng trong cùng shard
  - Cuối shard: log bảng summary (wandb.Table) + bar chart QWK vào run tổng hợp
  - Bật W&B   : set biến môi trường WANDB_API_KEY trước khi chạy, hoặc dùng --wandb-key
  - Tắt W&B   : không set WANDB_API_KEY (hoặc --wandb-key trống) → tự động disabled

Cách dùng:
  python scripts/run_optuna_shard_swin.py --shard 0 --data-dir ... --output-base ...
  python scripts/run_optuna_shard_swin.py --shard 0 ... --wandb-key <your_key>
  python scripts/run_optuna_shard_swin.py --list
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from optuna_shard_presets_swin import NUM_SHARDS, SHARD_PRESETS


# ── W&B helpers ───────────────────────────────────────────────────────────────

def _setup_wandb_key(key: str) -> bool:
    """
    Set WANDB_API_KEY vào env nếu được cung cấp qua --wandb-key.
    Trả về True nếu W&B sẽ chạy, False nếu disabled.
    """
    if key:
        os.environ["WANDB_API_KEY"] = key
    # Nếu không có key → disabled hoàn toàn (train.py/setup_wandb sẽ bỏ qua)
    has_key = bool(os.environ.get("WANDB_API_KEY", "").strip())
    if not has_key:
        os.environ["WANDB_MODE"] = "disabled"
        print("[W&B] WANDB_API_KEY không được set → W&B disabled.")
    else:
        # Đảm bảo không bị ép disabled từ môi trường cũ
        os.environ.pop("WANDB_MODE", None)
        print("[W&B] API key đã set → W&B enabled (project: optuna-fundus-shard-swin).")
    return has_key


def _log_shard_summary_to_wandb(
    shard_idx: int,
    results: list[dict],
    best_label: str,
    best_qwk: float,
    wandb_project: str,
) -> None:
    """
    Sau khi tất cả case trong shard chạy xong, tạo 1 W&B run tổng hợp để log:
      - wandb.Table  : mỗi hàng = 1 case, cột = label / QWK / LR / WD / warmup / T_0
      - Bar chart    : QWK từng case (wandb.plot.bar)
      - Summary metrics: best_qwk, best_label, n_cases, n_failed
    Run này dùng group = "swin_shard{N}", job_type = "shard_summary".
    """
    if os.environ.get("WANDB_MODE") == "disabled":
        return
    if not os.environ.get("WANDB_API_KEY", "").strip():
        return
    try:
        import wandb

        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        run = wandb.init(
            project=wandb_project,
            name=f"shard{shard_idx}_summary",
            group=f"swin_shard{shard_idx}",
            job_type="shard_summary",
            reinit=True,
            config={
                "shard": shard_idx,
                "model": "swinv2_base_384",
                "n_cases": len(results),
            },
        )

        # ── Table: 1 hàng / case ──────────────────────────────────────────
        columns = [
            "case_label", "max_val_qwk",
            "LR", "backbone_lr_scale", "weight_decay",
            "adamw_beta1", "adamw_beta2",
            "warmup_epochs", "T_0", "epochs",
            "status",
        ]
        table = wandb.Table(columns=columns)
        bar_data = wandb.Table(columns=["case", "val_qwk"])

        for r in results:
            snap = r.get("cfg_snapshot", {})
            status = "ERROR" if r["error"] else "ok"
            qwk_val = r["max_val_qwk"] if r["max_val_qwk"] >= 0 else 0.0
            table.add_data(
                r["label"],
                round(r["max_val_qwk"], 5),
                snap.get("LR"),
                snap.get("BACKBONE_LR_SCALE"),
                snap.get("WEIGHT_DECAY"),
                snap.get("ADAMW_BETA1"),
                snap.get("ADAMW_BETA2"),
                snap.get("WARMUP_EPOCHS"),
                snap.get("T_0"),
                snap.get("EPOCHS"),
                status,
            )
            bar_data.add_data(r["label"], round(qwk_val, 5))

        n_failed = sum(1 for r in results if r["error"])

        wandb.log({
            "shard/best_val_qwk":  round(best_qwk, 5),
            "shard/best_label":    best_label,
            "shard/n_cases":       len(results),
            "shard/n_failed":      n_failed,
            "shard/case_table":    table,
            "shard/qwk_bar":       wandb.plot.bar(
                bar_data, "case", "val_qwk",
                title=f"Shard {shard_idx} — Val QWK per case",
            ),
        })

        wandb.finish()
        print(f"[W&B] Shard {shard_idx} summary đã log: {getattr(run, 'url', '')}")
    except Exception as e:
        print(f"[W&B] log_shard_summary thất bại (không ảnh hưởng kết quả): {e}")


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_base_cfg(data_dir: str, repo_root: str, seed: int, shard_idx: int = 0) -> dict:
    return {
        # ── Model ──────────────────────────────────────────────────────────
        "MODEL_TYPE":             "swinv2_base_384",
        "HEAD_TYPE":              "ordinal",
        "IMAGE_SIZE":             384,
        # ── Data ───────────────────────────────────────────────────────────
        "PREPROCESSING_STRATEGY": "none",
        # BUG FIX: key phải là AUGMENT_VERSION để khớp với train.py
        # (cũ dùng AUG_VERSION → train.py.get("AUG_VERSION") luôn trả "v1" vì key sai)
        "AUGMENT_VERSION":        "v1",
        "BATCH_SIZE":             4,   # ← chỉnh ở đây hoặc override từ notebook
        "GRAD_ACCUM_STEPS":       2,   # effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS
        "NUM_WORKERS":            4,
        "TRAIN_RATIO":            0.7,
        "VAL_RATIO":              0.2,
        # ── Training ───────────────────────────────────────────────────────
        "FREEZE_STRATEGY":        "none",
        "USE_AMP":                True,
        "EPOCHS":                 15,
        # ── Optimizer defaults (preset override) ───────────────────────────
        "LR":                     5e-5,
        "BACKBONE_LR_SCALE":      0.1,
        "WEIGHT_DECAY":           1e-4,
        "ADAMW_BETA1":            0.9,
        "ADAMW_BETA2":            0.999,
        "ADAMW_EPS":              1e-8,
        # ── Scheduler ──────────────────────────────────────────────────────
        "WARMUP_EPOCHS":          2,
        "T_0":                    8,
        # ── Head ───────────────────────────────────────────────────────────
        "HEAD_HIDDEN_DIM":        None,
        "HEAD_DROPOUT":           0.0,
        "HEAD_DROPOUT_IN":        None,
        "HEAD_DROPOUT_OUT":       None,
        # ── Infra ──────────────────────────────────────────────────────────
        "SEED":                   seed,
        "DATA_DIR":               data_dir,
        "REPO_ROOT":              repo_root,
        "WANDB_PROJECT":          "optuna-fundus-shard-swin-v2",
        # W&B group: gom tất cả run cùng shard vào 1 group trên dashboard
        "WANDB_GROUP":            f"swin_shard{shard_idx}",
        "WANDB_JOB_TYPE":         "hpo_case",
        "OUTPUT_DIR":             "",
        "WANDB_RUN_NAME":         "",
    }


def merge_case(base: dict, case: dict) -> tuple[dict, str]:
    label = case.get("_label", "case")
    cfg = copy.deepcopy(base)
    for k, v in case.items():
        if k.startswith("_"):
            continue
        cfg[k] = v

    # Ép cứng — preset không được override
    cfg["MODEL_TYPE"]      = "swinv2_base_384"
    cfg["IMAGE_SIZE"]      = 384
    # BATCH_SIZE KHÔNG ép cứng — để notebook override được
    cfg["FREEZE_STRATEGY"] = "none"
    cfg["HEAD_TYPE"]       = "ordinal"

    # T_0 không được vượt quá (EPOCHS - WARMUP_EPOCHS)
    warmup = cfg.get("WARMUP_EPOCHS", 2)
    t0     = cfg.get("T_0", 8)
    cfg["T_0"] = min(int(t0), max(cfg.get("EPOCHS", 15) - warmup, 1))

    return cfg, label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chạy preset HPO SwinV2 theo shard (27 case, 15 epoch)"
    )
    p.add_argument("--shard",       type=int, default=-1,
                   help=f"Shard 0..{NUM_SHARDS - 1}")
    p.add_argument("--data-dir",    type=str, default="")
    p.add_argument("--output-base", type=str, default="")
    p.add_argument("--repo-root",   type=str, default="")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--wandb-key",   type=str, default="",
                   help="W&B API key (nếu không dùng env WANDB_API_KEY)")
    p.add_argument("--batch-size",  type=int, default=0,
                   help="Override BATCH_SIZE (0 = dùng default trong build_base_cfg)")
    p.add_argument("--grad-accum",  type=int, default=0,
                   help="Override GRAD_ACCUM_STEPS (0 = dùng default)")
    p.add_argument("--list",        action="store_true",
                   help="In danh sách shard + label rồi thoát")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        total = sum(len(s) for s in SHARD_PRESETS)
        print(f"Tổng {NUM_SHARDS} shard × ~4 case = {total} case (SwinV2-Base-384, 15 epoch):\n")
        for i, shard in enumerate(SHARD_PRESETS):
            print(f"  shard {i}: {len(shard)} case")
            for c in shard:
                lr  = c.get("LR", 0)
                blr = c.get("BACKBONE_LR_SCALE", "?")
                wd  = c.get("WEIGHT_DECAY", 0)
                wu  = c.get("WARMUP_EPOCHS", 2)
                t0  = c.get("T_0", "?")
                print(
                    f"    - {c['_label']:<38}"
                    f"  lr={lr:.1e}  blr={blr}  wd={wd:.0e}"
                    f"  wu={wu}  T0={t0}"
                )
        return

    if args.shard < 0 or args.shard >= NUM_SHARDS:
        raise SystemExit(f"--shard phải trong [0, {NUM_SHARDS - 1}] hoặc dùng --list")
    if not args.data_dir or not args.output_base:
        raise SystemExit("Cần --data-dir và --output-base (hoặc dùng --list)")

    root = args.repo_root or _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    # ── W&B: setup key trước khi import train (train.py gọi setup_wandb) ──────
    wandb_enabled = _setup_wandb_key(args.wandb_key)

    from src.train import run_single_training

    base  = build_base_cfg(args.data_dir, root, args.seed + args.shard, shard_idx=args.shard)

    # ── Override BATCH_SIZE / GRAD_ACCUM_STEPS từ CLI (notebook) ─────────────
    if args.batch_size > 0:
        base["BATCH_SIZE"] = args.batch_size
        print(f"[Config] BATCH_SIZE overridden → {args.batch_size}")
    if args.grad_accum > 0:
        base["GRAD_ACCUM_STEPS"] = args.grad_accum
        print(f"[Config] GRAD_ACCUM_STEPS overridden → {args.grad_accum}")
    print(f"[Config] Effective batch = {base['BATCH_SIZE']} × {base['GRAD_ACCUM_STEPS']} = "
          f"{base['BATCH_SIZE'] * base['GRAD_ACCUM_STEPS']}")

    cases = SHARD_PRESETS[args.shard]

    shard_dir = os.path.join(args.output_base, "optuna_shards_swin", f"shard_{args.shard}")
    os.makedirs(shard_dir, exist_ok=True)

    results: list[dict] = []
    best_qwk   = -1.0
    best_label = ""

    for idx, raw_case in enumerate(cases):
        cfg, label = merge_case(base, raw_case)
        run_dir = os.path.join(shard_dir, f"{idx:02d}_{label}")
        cfg["OUTPUT_DIR"]     = run_dir
        # W&B run name: rõ ràng hơn để dễ filter trên dashboard
        cfg["WANDB_RUN_NAME"] = f"s{args.shard}_{label}"
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

        # ── Kiểm tra config bắt buộc trước khi chạy ──────────────────────────
        assert cfg["FREEZE_STRATEGY"] == "none", \
            f"BUG: FREEZE_STRATEGY phải là 'none', có '{cfg['FREEZE_STRATEGY']}'"
        assert cfg.get("AUGMENT_VERSION", "v1") == "v1", \
            f"BUG: AUGMENT_VERSION phải là 'v1', có '{cfg.get('AUGMENT_VERSION')}'"
        assert cfg["MODEL_TYPE"] == "swinv2_base_384", \
            f"BUG: MODEL_TYPE sai: '{cfg['MODEL_TYPE']}'"
        assert cfg["IMAGE_SIZE"] == 384, \
            f"BUG: IMAGE_SIZE phải là 384, có {cfg['IMAGE_SIZE']}"
        # BATCH_SIZE không assert — cho phép override từ notebook

        print(
            f"\n=== SwinV2 | shard {args.shard} | case {idx + 1}/{len(cases)}: {label} ===\n"
            f"    LR={cfg['LR']:.1e}  BACKBONE_LR_SCALE={cfg['BACKBONE_LR_SCALE']}"
            f"  WD={cfg['WEIGHT_DECAY']:.0e}\n"
            f"    WARMUP={cfg['WARMUP_EPOCHS']}  T_0={cfg['T_0']}"
            f"  EPOCHS={cfg['EPOCHS']}"
            f"  BATCH={cfg['BATCH_SIZE']}×ACCUM={cfg.get('GRAD_ACCUM_STEPS',1)}"
            f"  (effective={cfg['BATCH_SIZE']*cfg.get('GRAD_ACCUM_STEPS',1)})\n"
            f"    [Config check] FREEZE={cfg['FREEZE_STRATEGY']}  "
            f"AUG={cfg.get('AUGMENT_VERSION','v1')}  HEAD={cfg['HEAD_TYPE']}  ✓"
            + (f"\n    W&B run: {cfg['WANDB_RUN_NAME']} (group: {cfg['WANDB_GROUP']})"
               if wandb_enabled else "  [W&B disabled]")
        )

        score = -1.0
        err   = None
        try:
            run_single_training(cfg)
            hist_path = os.path.join(run_dir, "history.json")
            if os.path.isfile(hist_path):
                with open(hist_path) as f:
                    hist = json.load(f)
                vq = hist.get("val_qwk", [])
                if vq:
                    score = float(max(vq))
        except Exception as e:
            err = str(e)
            print(f"FAILED: {e}")
        finally:
            # ── Giải phóng GPU memory sau mỗi case (tránh OOM case tiếp theo) ──
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(
                f"  [GPU cleanup] "
                + (
                    f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.0f} MiB  "
                    f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.0f} MiB"
                    if torch.cuda.is_available() else "no CUDA"
                )
            )

        results.append({
            "label":       label,
            "max_val_qwk": score,
            "output_dir":  run_dir,
            "cfg_snapshot": {
                k: cfg[k]
                for k in (
                    "LR", "BACKBONE_LR_SCALE", "WEIGHT_DECAY",
                    "ADAMW_BETA1", "ADAMW_BETA2",
                    "WARMUP_EPOCHS", "T_0", "EPOCHS",
                )
            },
            "error": err,
        })
        if score > best_qwk:
            best_qwk   = score
            best_label = label

    # ── Lưu summary JSON ─────────────────────────────────────────────────────
    summary = {
        "model":            "swinv2_base_384",
        "shard":            args.shard,
        "best_max_val_qwk": best_qwk,
        "best_label":       best_label,
        "cases":            results,
    }
    out_json = os.path.join(shard_dir, "shard_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # ── W&B: log bảng tổng kết shard ─────────────────────────────────────────
    if wandb_enabled:
        _log_shard_summary_to_wandb(
            shard_idx=args.shard,
            results=results,
            best_label=best_label,
            best_qwk=best_qwk,
            wandb_project=base["WANDB_PROJECT"],
        )

    print(f"\nShard {args.shard} xong.")
    print(f"  Best QWK ≈ {best_qwk:.4f}  ({best_label})")
    print(f"  Summary  : {out_json}")


if __name__ == "__main__":
    main()
