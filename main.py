# main.py - Phiên bản FastAPI ổn định cho Render
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import io
import json

# Tải model và dữ liệu khi server khởi động
try:
    model = YOLO("best.pt")
    with open("diseases.json", "r", encoding="utf-8") as f:
        disease_details = json.load(f)
    print("Model và dữ liệu đã được tải thành công!")
except Exception as e:
    model = None
    disease_details = {}
    print(f"Lỗi khi tải model hoặc dữ liệu: {e}")

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Skin Diagnosis AI API")

# Định nghĩa cấu trúc dữ liệu sẽ trả về
class DiagnosisResponse(BaseModel):
    label: str
    confidence: float
    display_name: str
    description: str
    treatment: str
    severity_text: str

# Tạo một endpoint duy nhất tại /diagnose/
@app.post("/diagnose/", response_model=DiagnosisResponse)
async def diagnose_skin(image_file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Lỗi: Model chưa được tải.")

    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))
    results = model(image)

    if results and len(results[0].boxes) > 0:
        top_box = results[0].boxes[0]
        confidence = top_box.conf[0].item()
        class_index = int(top_box.cls[0].item())
        label = model.names[class_index]
        details = disease_details.get(label, {})

        return DiagnosisResponse(
            label=label,
            confidence=confidence,
            display_name=details.get("display_name", label),
            description=details.get("description", "Không có mô tả."),
            treatment=details.get("treatment", "Không có gợi ý."),
            severity_text=details.get("severity_text", "Chưa xác định."),
        )
    else:
        return DiagnosisResponse(
            label="Không phát hiện", confidence=0.0,
            display_name="Không phát hiện bệnh",
            description="Không tìm thấy dấu hiệu bệnh rõ ràng trong ảnh.",
            treatment="Vui lòng thử lại với một ảnh khác rõ nét hơn.",
            severity_text="Chưa xác định",
        )

@app.get("/")
def read_root():
    return {"status": "Skin AI Server is running"}

