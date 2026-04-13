from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from service.swin_service import predict_fundus
from service.efficient_service import predict_fundus_effb7
from service.xai_service import run_xai_for_web

router = APIRouter()

# 1. CẬP NHẬT VALID_MODES ĐỒNG BỘ VỚI FRONTEND
VALID_MODES = {"eff", "swin", "xai-eff", "xai-swin"}

async def _process_single(image_bytes: bytes, mode: str) -> dict:
    """Xử lý 1 ảnh theo mode, trả về dict kết quả chuẩn."""
    
    if mode == "eff":
        r = predict_fundus_effb7(image_bytes)
        return {
            "mode"       : "eff",
            "grade"      : r["grade"],
            "raw_score"  : r["raw_score"],
            "description": r["description"],
            "ai_answer"  : f"EfficientNet-B7 → Grade {r['grade']} — {r['description']}",
        }

    if mode == "swin":
        r = predict_fundus(image_bytes)
        return {
            "mode"       : "swin",
            "grade"      : r["grade"],
            "raw_score"  : r["raw_score"],
            "description": r["description"],
            "ai_answer"  : f"SwinV2-Base → Grade {r['grade']} — {r['description']}",
        }

    # 2. TÁCH RIÊNG XỬ LÝ CHO XAI-EFF VÀ XAI-SWIN
    if mode in ["xai-eff", "xai-swin"]:
        xai_data = run_xai_for_web(image_bytes)
        
        # Khởi tạo bộ khung JSON trả về
        result = {
            "mode"        : mode, # Trả về đúng 'xai-eff' hoặc 'xai-swin'
            "original_b64": xai_data.get("original_b64"),
        }
        
        # Gắn dữ liệu tương ứng với model được chọn
        if mode == "xai-eff":
            eff = xai_data["efficientnet"]
            result["ai_answer"] = f"Phân tích XAI (EfficientNet-B7): Grade {eff['grade']} — {eff['description']}"
            result["efficientnet"] = {
                "grade"      : eff["grade"],
                "raw_score"  : eff["raw_score"],
                "description": eff["description"],
                "heatmap_b64": eff["heatmap_b64"],
                "overlay_b64": eff["overlay_b64"],
            }
            
        elif mode == "xai-swin":
            swin = xai_data["swinv2"]
            result["ai_answer"] = f"Phân tích XAI (SwinV2-Base): Grade {swin['grade']} — {swin['description']}"
            result["swinv2"] = {
                "grade"      : swin["grade"],
                "raw_score"  : swin["raw_score"],
                "description": swin["description"],
                "heatmap_b64": swin["heatmap_b64"],
                "overlay_b64": swin["overlay_b64"],
            }
            
        return result


@router.post("/predict")
async def predict_endpoint(
    images    : list[UploadFile] = File(...),
    model_mode: str              = Form(...),
):
    if not images:
        raise HTTPException(status_code=400, detail="Không có ảnh được gửi lên.")
        
    if model_mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"model_mode không hợp lệ. Chọn: {VALID_MODES}")

    for img in images:
        if img.content_type not in ("image/jpeg", "image/png", "image/jpg"):
            raise HTTPException(status_code=400, detail=f"{img.filename}: chỉ chấp nhận định dạng JPEG/PNG.")

    try:
        # Xử lý tuần tự từng ảnh, giữ đúng thứ tự hiển thị
        results = []
        for img in images:
            image_bytes = await img.read()
            result      = await _process_single(image_bytes, model_mode)
            result["filename"] = img.filename
            results.append(result)

        return {"status": "success", "mode": model_mode, "results": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    return {"status": "ok", "message": "API đang hoạt động trơn tru."}