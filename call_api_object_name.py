import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI()

# Load model sẵn khi server khởi động
model = YOLO("models/yolo11n.pt").to("cuda")

# Hàm xử lý YOLO
def infer_api(image):
    results = model.predict(image, verbose=False)
    boxes = results[0].boxes
    objects = []
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        objects.append(class_name)
    return objects

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Đọc nội dung file thành mảng numpy
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Dự đoán bằng YOLO
    object_names = infer_api(image)

    return JSONResponse(content={"objects": object_names})
