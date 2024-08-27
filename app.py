from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import os
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins (for development purposes only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

CLASS_NAMES = {    
    0: "Bumper",
    1: "Paint Damage",
    2: "Body Dent",
    3: "Fender",
    4: "Cracked Windsheild",
    5: "Suspension",
    6: "Hood Damage",
    7: "Weather Damage",
    8: "Window Glass",
    9: "Trunk Boot",
    10: "Head Light"
}

# Load the trained model
model_path = "model/train6/weights/best.pt"  # Update this path to your model's path
model = YOLO(model_path)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run inference on the image
    results = model(image)

    # Extract predictions
    predictions = []
    for result in results[0].boxes:
        #print(result)
        x1, y1, x2, y2 = result.xyxy[0]
        score = result.conf[0]
        class_id = result.cls[0]
        predictions.append({
            "bbox": [x1.item(), y1.item(), x2.item(), y2.item()],
            "score": score.item(),
            "class_id": class_id.item()
        })
        # Add class names to predictions
        for prediction in predictions:
            class_id = prediction["class_id"]
            prediction["class_name"] = CLASS_NAMES.get(class_id, "Unknown")

    return JSONResponse(content={"predictions": predictions})

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="192.168.1.3", port=8000)
    #uvicorn.run(app, host="192.168.34.160", port=8000)
    uvicorn.run(app, host="172.0.40.25", port=8000)