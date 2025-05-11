from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from ultralytics import YOLO
import io

app = FastAPI()

model = YOLO("models/yolo11n.pt").to("cuda")


def visualize(img, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img


@app.post("/detect-image/")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model.predict(image, verbose=False)
    boxes = results[0].boxes
    image_with_boxes = visualize(image, boxes)

    # Encode ảnh đã xử lý sang dạng JPEG bytes để trả về
    _, img_encoded = cv2.imencode(".jpg", image_with_boxes)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
