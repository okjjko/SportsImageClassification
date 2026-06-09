"""FastAPI 路由 — 运动图片分类 Web API"""

import os
import threading
import time
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

from .config import BEST_MODEL_PATH, FIGURE_DIR
from .predict import load_model, predict_image

router = APIRouter(prefix="/api")

_model = None
_class_names = None
_device = None
_model_lock = threading.Lock()

_history = []
_history_counter = 0
_history_lock = threading.Lock()
_HISTORY_MAX = 200

HISTORY_IMAGE_DIR = (
    Path(__file__).resolve().parent.parent / "web" / "history_images"
)
EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "archive" / "test"


def _get_model():
    global _model, _class_names, _device
    if _model is None:
        with _model_lock:
            if _model is None:
                if not BEST_MODEL_PATH.exists():
                    raise FileNotFoundError("模型文件不存在，请先训练")
                _model, _class_names, _device = load_model()
    return _model, _class_names, _device


def _add_history(filepath, results, source):
    global _history, _history_counter
    with _history_lock:
        _history_counter += 1
        entry = {
            "id": _history_counter,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": os.path.basename(filepath),
            "filepath": filepath,
            "source": source,
            "results": results,
        }
        _history.append(entry)
        if len(_history) > _HISTORY_MAX:
            _history.pop(0)
        return entry


def _save_upload(file: UploadFile) -> str:
    HISTORY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_name = f"{ts}_{file.filename}"
    dest = HISTORY_IMAGE_DIR / safe_name
    content = file.file.read()
    with open(dest, "wb") as f:
        f.write(content)
    return str(dest)


def _classify(filepath, source):
    image = Image.open(filepath).convert("RGB")
    model, class_names, device = _get_model()
    preds = predict_image(model, image, class_names, device, top_k=3)
    _add_history(filepath, preds, source)
    return preds


# ─── 状态 ─────────────────────────────────────────

@router.get("/status")
def get_status():
    val_acc_str = ""
    try:
        import torch as _torch
        ckpt = _torch.load(
            str(BEST_MODEL_PATH), map_location="cpu", weights_only=False
        )
        if "val_acc" in ckpt:
            val_acc_str = f"{ckpt['val_acc']:.2f}%"
    except Exception:
        val_acc_str = "N/A"

    test_acc_str = ""
    report_path = FIGURE_DIR / "classification_report.txt"
    if report_path.exists():
        try:
            lines = report_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if line.startswith("Top-1 准确率:"):
                    test_acc_str = f"测试 Top-1: {line.split(':')[1].strip()}"
                elif line.startswith("Top-3 准确率:"):
                    test_acc_str += f" | Top-3: {line.split(':')[1].strip()}"
        except Exception:
            pass

    return {
        "model": "EfficientNet-B0",
        "classes": 100,
        "val_acc": val_acc_str,
        "test_acc": test_acc_str or None,
    }


# ─── 示例图片 ─────────────────────────────────────

@router.get("/examples")
def get_examples():
    if not EXAMPLE_DIR.exists():
        return {"examples": []}
    cats = sorted([d.name for d in EXAMPLE_DIR.iterdir() if d.is_dir()])
    if not cats:
        return {"examples": []}
    step = max(1, len(cats) // 6)
    selected = cats[::step][:6]
    examples = []
    for cat in selected:
        cat_dir = EXAMPLE_DIR / cat
        files = sorted(cat_dir.iterdir())
        if files:
            examples.append({
                "name": cat,
                "url": f"/api/examples/static/{cat}/{files[0].name}",
            })
    return {"examples": examples}


# ─── 单张识别 ────────────────────────────────────

@router.post("/predict/single")
async def predict_single(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(400, "请上传图片文件")
    try:
        saved_path = _save_upload(file)
        preds = _classify(saved_path, "单张")
        return {"filename": file.filename, "results": preds}
    except Exception as e:
        raise HTTPException(500, f"推理失败: {e}")


# ─── 批量识别 ────────────────────────────────────

@router.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "请上传图片文件")
    results = []
    for f in files:
        try:
            saved_path = _save_upload(f)
            preds = _classify(saved_path, "批量")
            results.append({
                "filename": f.filename,
                "results": preds,
            })
        except Exception as e:
            results.append({
                "filename": f.filename,
                "results": [],
                "error": str(e),
            })
    return {"results": results}


# ─── 历史记录 ────────────────────────────────────

@router.get("/history")
def get_history():
    with _history_lock:
        out = []
        for h in reversed(_history):
            top1 = h["results"][0] if h["results"] else None
            out.append({
                "id": h["id"],
                "filename": h["filename"],
                "timestamp": h["timestamp"],
                "source": h["source"],
                "top1": top1["class"] if top1 else None,
                "top1_conf": top1["confidence"] if top1 else None,
            })
        return {"history": out}


@router.get("/history/{entry_id}")
def get_history_detail(entry_id: int):
    with _history_lock:
        for h in _history:
            if h["id"] == entry_id:
                return {
                    "id": h["id"],
                    "filename": h["filename"],
                    "filepath": h["filepath"],
                    "timestamp": h["timestamp"],
                    "source": h["source"],
                    "results": h["results"],
                }
    raise HTTPException(404, "记录不存在")


@router.delete("/history")
def clear_history():
    global _history
    with _history_lock:
        _history.clear()
    return {"ok": True}
