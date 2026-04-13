import os
import sys
import html
import zipfile
import json

import warnings
import torch
import torch.nn as nn
import pandas as pd
from torch.amp import autocast
from tqdm import tqdm

warnings.filterwarnings("ignore")
PIPELINE_CONSOLE_FILENAME = "pipeline_full_console.log"

_orig_stdout = None
_orig_stderr = None
_log_fp = None


class _TeeStream:
    """Tee stdout/stderr: vừa in ra terminal vừa ghi vào file."""

    __slots__ = ("_stream", "_log")

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        if isinstance(data, bytes):
            try:
                data = data.decode(getattr(self._stream, "encoding", None) or "utf-8", errors="replace")
            except Exception:
                data = data.decode("utf-8", errors="replace")
        self._stream.write(data)
        self._stream.flush()
        if self._log:
            self._log.write(data)
            self._log.flush()

    def flush(self):
        self._stream.flush()
        if self._log:
            self._log.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start_pipeline_console_capture(output_dir: str) -> str | None:
    global _orig_stdout, _orig_stderr, _log_fp
    if _orig_stdout is not None:
        return os.path.join(output_dir, PIPELINE_CONSOLE_FILENAME)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, PIPELINE_CONSOLE_FILENAME)
    _log_fp = open(path, "w", encoding="utf-8", buffering=1)
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = _TeeStream(_orig_stdout, _log_fp)
    sys.stderr = _TeeStream(_orig_stderr, _log_fp)
    _log_fp.write("=== PIPELINE CONSOLE LOG (stdout + stderr, notebook process) ===\n\n")
    _log_fp.flush()
    return path


def stop_pipeline_console_capture() -> None:
    global _orig_stdout, _orig_stderr, _log_fp
    if _orig_stdout is None:
        return
    try:
        if _log_fp:
            _log_fp.write(
                "\n=== END TEED CAPTURE (DDP lines đã append phía trên nếu có) ===\n"
            )
            _log_fp.flush()
    except Exception:
        pass
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    _orig_stdout, _orig_stderr = None, None
    if _log_fp:
        try:
            _log_fp.close()
        except Exception:
            pass
        _log_fp = None


def append_pipeline_log_line(output_dir: str, text: str, rank: int = 0) -> None:
    if rank != 0:
        return
    path = os.path.join(output_dir, PIPELINE_CONSOLE_FILENAME)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text if text.endswith("\n") else text + "\n")
    except Exception:
        pass


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics":         metrics,
    }, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Checkpoint loaded: {path}  (epoch {ckpt['epoch']}, metrics: {ckpt['metrics']})")
    return {"epoch": ckpt["epoch"], "metrics": ckpt["metrics"]}


# ── Submission CSV ─────────────────────────────────────────────────────────────

def generate_submission(
    model: nn.Module,
    submit_loader,
    image_ids: list,
    output_dir: str,
    device: torch.device,
    coef: list = None,
) -> str:
    from .models import predict_labels

    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in tqdm(submit_loader, desc="Generating submission"):
            images = images.to(device, non_blocking=True)
            with autocast("cuda"):
                outputs = model(images).reshape(-1)  # [B]
            preds = predict_labels(outputs, coef=coef)
            all_preds.extend(preds.cpu().tolist())

    os.makedirs(output_dir, exist_ok=True)
    sub_path = os.path.join(output_dir, "submission.csv")
    df = pd.DataFrame({"id_code": image_ids, "diagnosis": all_preds})
    df.to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path}  ({len(df)} rows)")
    return sub_path


# ── Zip outputs ───────────────────────────────────────────────────────────────

def zip_outputs(output_dir: str) -> str:
    zip_path = output_dir.rstrip("/\\") + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for fname in files:
                file_path = os.path.join(root, fname)
                arcname   = os.path.relpath(file_path, start=os.path.dirname(output_dir))
                zf.write(file_path, arcname)
    print(f"Outputs zipped: {zip_path}")
    return zip_path


# ── W&B setup ─────────────────────────────────────────────────────────────────

def setup_wandb(cfg: dict):
    try:
        import wandb
        api_key = os.environ.get("WANDB_API_KEY", "")
        if not api_key:
            print("WANDB_API_KEY trống — bỏ qua W&B logging.")
            return None

        wandb.login(key=api_key, relogin=True)
        init_kwargs = dict(
            project=cfg.get("WANDB_PROJECT", "fundus-baseline"),
            name=cfg.get("WANDB_RUN_NAME", cfg.get("MODEL_TYPE", "run")),
            config={
                k: v for k, v in cfg.items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool))
            },
            reinit=True,
        )
        # Hỗ trợ group (gom runs cùng shard) và job_type (hpo_case / shard_summary)
        if cfg.get("WANDB_GROUP"):
            init_kwargs["group"] = cfg["WANDB_GROUP"]
        if cfg.get("WANDB_JOB_TYPE"):
            init_kwargs["job_type"] = cfg["WANDB_JOB_TYPE"]
        run = wandb.init(**init_kwargs)
        print(f"W&B run initialized: {run.url}")
        return run
    except Exception as e:
        print(f"W&B setup thất bại: {e} — tiếp tục training không có W&B.")
        return None


# ── W&B: meta + resume full pipeline (eval → submission → figures) ───────────

def save_wandb_run_meta(output_dir: str, run) -> str | None:
    if run is None:
        return None
    meta = {
        "id": run.id,
        "project": run.project,
        "entity": getattr(run, "entity", None) or "",
        "name": run.name,
        "url": getattr(run, "url", "") or "",
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "wandb_run_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"W&B run meta saved: {path}")
    return path


def resume_wandb_run(output_dir: str, cfg: dict | None = None):
    path = os.path.join(output_dir, "wandb_run_meta.json")
    if not os.path.isfile(path) or not os.environ.get("WANDB_API_KEY"):
        print("Không resume W&B (thiếu wandb_run_meta.json hoặc WANDB_API_KEY).")
        return None
    try:
        import wandb
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        kwargs = {
            "project": meta["project"],
            "id": meta["id"],
            "resume": "allow",
        }
        if meta.get("entity"):
            kwargs["entity"] = meta["entity"]
        run = wandb.init(**kwargs)
        print(f"W&B run resumed (post-training pipeline): {getattr(run, 'url', run)}")
        return run
    except Exception as e:
        print(f"W&B resume thất bại: {e}")
        return None


def log_eval_phase_to_wandb(run, metrics: dict, eval_txt_path: str | None = None) -> None:
    if run is None:
        return
    import wandb
    log_d = {
        "pipeline/stage": "eval_internal_test",
        "eval/val_loss": metrics["val_loss"],
        "eval/qwk": metrics["qwk"],
        "eval/accuracy": metrics["accuracy"],
        "eval/macro_f1": metrics["macro_f1"],
        "eval/balanced_accuracy": metrics["balanced_accuracy"],
    }
    for i, r in enumerate(metrics.get("per_class_recall", [])):
        log_d[f"eval/recall_class_{i}"] = float(r)
    txt = metrics.get("classification_report_text", "")
    if txt:
        safe = html.escape(txt)
        log_d["eval/classification_report"] = wandb.Html(f"<pre style='font-size:11px'>{safe}</pre>")
    wandb.log(log_d)
    if eval_txt_path and os.path.isfile(eval_txt_path):
        try:
            art = wandb.Artifact("evaluation_metrics", type="metrics")
            art.add_file(eval_txt_path)
            run.log_artifact(art)
        except Exception as e:
            print(f"W&B artifact evaluation_metrics: {e}")


def log_submission_phase_to_wandb(run, submission_path: str, sub_df) -> None:
    if run is None:
        return
    import wandb
    log_d = {"pipeline/stage": "submission_inference", "submission/num_rows": len(sub_df)}
    counts = sub_df["diagnosis"].value_counts().sort_index()
    for k, v in counts.items():
        log_d[f"submission/count_class_{int(k)}"] = int(v)
    wandb.log(log_d)
    try:
        art = wandb.Artifact("submission_csv", type="predictions")
        art.add_file(submission_path)
        run.log_artifact(art)
    except Exception as e:
        print(f"W&B artifact submission: {e}")


def log_visualization_phase_to_wandb(
    run,
    figure_paths: list,
    output_dir: str,
    zip_path: str | None = None,
) -> None:
    if run is None:
        return
    import wandb

    console_path = os.path.join(output_dir, PIPELINE_CONSOLE_FILENAME)
    if os.path.isfile(console_path):
        try:
            c_art = wandb.Artifact("pipeline_console_log", type="console")
            c_art.add_file(console_path)
            run.log_artifact(c_art)
        except Exception as e:
            print(f"W&B artifact pipeline_console_log: {e}")

    # Một step W&B: stage + preview log + figure
    combined = {"pipeline/stage": "figures_and_artifacts"}
    if os.path.isfile(console_path):
        try:
            with open(console_path, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            preview = raw if len(raw) <= 120_000 else raw[:120_000] + "\n\n... [truncated for W&B HTML preview] ..."
            combined["pipeline/full_console_preview"] = wandb.Html(
                f"<pre style='font-size:10px;white-space:pre-wrap'>{html.escape(preview)}</pre>"
            )
        except Exception as e:
            print(f"W&B console preview: {e}")
    for p in figure_paths:
        if p and os.path.isfile(p):
            key = f"figures/{os.path.basename(p).replace('.', '_')}"
            combined[key] = wandb.Image(p)
    wandb.log(combined)

    try:
        art = wandb.Artifact("pipeline_outputs", type="run_outputs")
        if os.path.isdir(output_dir):
            art.add_dir(output_dir)
        run.log_artifact(art)
    except Exception as e:
        print(f"W&B artifact pipeline_outputs: {e}")
    if zip_path and os.path.isfile(zip_path):
        try:
            zart = wandb.Artifact("outputs_zip", type="archive")
            zart.add_file(zip_path)
            run.log_artifact(zart)
        except Exception as e:
            print(f"W&B artifact zip: {e}")


def wandb_finish_pipeline(run, message: str = "pipeline_complete") -> None:
    if run is None:
        return
    import wandb
    wandb.log({"pipeline/stage": message})
    wandb.finish()
    print("W&B run đã đóng — toàn bộ pipeline đã log.")


# ── Load history JSON ─────────────────────────────────────────────────────────

def load_history(output_dir: str) -> dict:
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "r") as f:
        history = json.load(f)
    return history
