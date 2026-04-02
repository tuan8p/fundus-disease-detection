"""
utils.py
--------
Các tiện ích hỗ trợ pipeline:

  - save_checkpoint      : Lưu model state_dict + optimizer + metadata
  - load_checkpoint      : Load lại checkpoint, trả về model + metadata
  - generate_submission  : Inference → submission.csv trong output_dir + bản copy ra thư mục cha (submit Kaggle)
  - zip_outputs          : Nén toàn bộ thư mục outputs/ thành outputs.zip
  - setup_wandb          : Khởi tạo W&B run (key từ Kaggle Secrets)
  - start/stop_pipeline_console_capture : Tee stdout/stderr → file (notebook)
  - append_pipeline_log_line              : Ghi thêm dòng từ DDP worker (rank 0)
"""

import os
import sys
import html
import json
import shutil
import zipfile

import warnings
import torch
import torch.nn as nn
import pandas as pd
from torch.amp import autocast
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Tên file ghi toàn bộ print (notebook + append từ DDP rank 0)
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
    """
    Bắt đầu ghi mọi stdout/stderr của notebook (Cell 2 trở đi) vào
    OUTPUT_DIR/pipeline_full_console.log. Gọi một lần sau khi tạo OUTPUT_DIR.

    Tiến trình DDP không kế thừa tee — dùng append_pipeline_log_line() trong train.py.
    """
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
    """Tắt tee, đóng file — gọi trước khi upload log lên W&B (thường là đầu Cell 8)."""
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
    """
    Ghi thêm một dòng vào pipeline_full_console.log (dùng trong DDP worker rank 0).
    Không đi qua parent tee nên phải append file riêng.
    """
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
    """
    Lưu checkpoint gồm model weights, optimizer state và metadata epoch.

    Args:
        model     : Model (không bọc DDP — dùng model.module khi DDP)
        optimizer : Optimizer hiện tại
        epoch     : Epoch hiện tại
        metrics   : Dict metrics (val_qwk, val_loss, ...)
        path      : Đường dẫn file .pth
    """
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
    """
    Load checkpoint vào model (và optimizer nếu cần tiếp tục training).

    Args:
        model     : Model đã được khởi tạo cùng kiến trúc
        path      : Đường dẫn file .pth
        device    : Thiết bị load tensor lên
        optimizer : Nếu không None, load luôn optimizer state

    Returns:
        dict: metadata (epoch, metrics)
    """
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
) -> str:
    """
    Chạy inference trên tập test không nhãn, lưu kết quả thành submission.csv.
    Format giống sample_submission.csv: cột id_code, cột diagnosis.

    Args:
        model        : Model đã load best checkpoint
        submit_loader: DataLoader của test.csv (không nhãn)
        image_ids    : Danh sách id_code từ test.csv (giữ đúng thứ tự)
        output_dir   : Thư mục lưu submission.csv
        device       : torch.device

    Returns:
        str: Đường dẫn file submission.csv
    """
    from .models import predict_labels

    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in tqdm(submit_loader, desc="Generating submission"):
            images = images.to(device, non_blocking=True)
            with autocast("cuda"):
                outputs = model(images).reshape(-1)  # [B]
            preds = predict_labels(outputs)
            all_preds.extend(preds.cpu().tolist())

    os.makedirs(output_dir, exist_ok=True)
    sub_path = os.path.join(output_dir, "submission.csv")
    df = pd.DataFrame({"id_code": image_ids, "diagnosis": all_preds})
    df.to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path}  ({len(df)} rows)")

    copy_path = copy_submission_for_kaggle_submit(output_dir)
    if copy_path:
        print(f"Copy để submit Competition (ngoài outputs/): {copy_path}")

    return sub_path


def copy_submission_for_kaggle_submit(output_dir: str, filename: str = "submission.csv") -> str | None:
    """
    Copy `output_dir/submission.csv` ra thư mục **cha** của `output_dir` (cùng cấp với folder outputs).

    Ví dụ:
      - OUTPUT_DIR = `/kaggle/working/outputs` → `/kaggle/working/submission.csv`
    Trên Kaggle: dùng tab Output hoặc chọn file `submission.csv` ở root working để submit.

    Returns:
        Đường dẫn file đã copy, hoặc None nếu file nguồn không tồn tại.
    """
    src = os.path.join(output_dir, filename)
    if not os.path.isfile(src):
        return None
    parent = os.path.dirname(os.path.abspath(output_dir.rstrip("/\\")))
    dst = os.path.join(parent, filename)
    shutil.copy2(src, dst)
    return dst


# ── Zip outputs ───────────────────────────────────────────────────────────────

def zip_outputs(output_dir: str) -> str:
    """
    Nén toàn bộ thư mục output_dir thành outputs.zip ở cùng cấp.

    Args:
        output_dir : Thư mục outputs/ cần nén

    Returns:
        str: Đường dẫn file .zip
    """
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
    """
    Khởi tạo W&B run. API key lấy từ biến môi trường WANDB_API_KEY
    (nên được set từ Kaggle Secrets trước khi gọi hàm này).

    Args:
        cfg (dict): Thông số config từ Cell 2 notebook

    Returns:
        wandb.Run hoặc None nếu wandb không khả dụng / key trống
    """
    try:
        import wandb
        api_key = os.environ.get("WANDB_API_KEY", "")
        if not api_key:
            print("WANDB_API_KEY trống — bỏ qua W&B logging.")
            return None

        wandb.login(key=api_key, relogin=True)
        run = wandb.init(
            project=cfg.get("WANDB_PROJECT", "fundus-baseline"),
            name=cfg.get("WANDB_RUN_NAME", cfg.get("MODEL_TYPE", "run")),
            config={
                k: v for k, v in cfg.items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool))
            },
            reinit=True,
        )
        print(f"W&B run initialized: {run.url}")
        return run
    except Exception as e:
        print(f"W&B setup thất bại: {e} — tiếp tục training không có W&B.")
        return None


# ── W&B: meta + resume full pipeline (eval → submission → figures) ───────────

def save_wandb_run_meta(output_dir: str, run) -> str | None:
    """
    Lưu id/project/entity của run hiện tại trước khi wandb.finish() ở training.
    Notebook sẽ đọc file này để resume và log tiếp các giai đoạn sau epoch cuối.
    """
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
    """
    Resume cùng một W&B run (sau khi training đã finish) để log eval / submission / artifact.
    Trả về wandb.Run hoặc None nếu không có meta / API key.
    """
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
    """Log metrics internal test + file evaluation_metrics.txt lên W&B."""
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
    """Log phân phối dự đoán trên tập submit + file CSV."""
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
    """
    Log ảnh figure + file pipeline_full_console.log + toàn bộ thư mục outputs + zip.

    Gọi **sau** stop_pipeline_console_capture() để file log đã đóng và đủ nội dung.
    """
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
    """Đánh dấu pipeline kết thúc và đóng W&B run (gọi ở cell cuối notebook)."""
    if run is None:
        return
    import wandb
    wandb.log({"pipeline/stage": message})
    wandb.finish()
    print("W&B run đã đóng — toàn bộ pipeline đã log.")


# ── Load history JSON ─────────────────────────────────────────────────────────

def load_history(output_dir: str) -> dict:
    """
    Đọc history.json được lưu bởi train_worker sau khi training.

    Args:
        output_dir : Thư mục outputs/

    Returns:
        dict với keys: train_loss, train_acc, train_qwk, val_loss, val_acc, val_qwk
    """
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "r") as f:
        history = json.load(f)
    return history
