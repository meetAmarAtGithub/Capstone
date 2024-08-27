import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import os
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import logging
import json

app = FastAPI()

origins = [
    #"http://192.168.1.3:8000"
    #"http://192.168.34.160:8000"
    "http://172.0.40.25:8000"
]

# Allow all origins (for development purposes only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

logging.basicConfig(level=logging.INFO)

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

#Find car make and model

CARQUERY_API_URL = 'https://www.carqueryapi.com/api/0.3/?callback=?&cmd='
'''
@app.get("/car-makes/")
async def get_car_makes(sold_in_country: str):
    url = f"{CARQUERY_API_URL}getMakes&sold_in_country={sold_in_country}"
    print("url - ", url)
    response = requests.get(url)
    print("Car Makes Response:", response.text)  # Add this line for logging
    data = response.json()
    return data.get('Makes', [])

@app.get("/car-makes/")
async def get_car_makes(sold_in_country: str):
    url = f"https://www.carqueryapi.com/api/0.3/?callback=?&cmd=getMakes&sold_in_country={sold_in_country}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        # Strip JSONP callback
        json_str = response.text[2:-2]
        data = json.loads(json_str)
        return data
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        raise HTTPException(status_code=500, detail="Error fetching car makes")
'''
# Updated endpoint to fetch car makes
@app.get("/car-makes/")
async def get_car_makes(sold_in_country: str):
    url = f"https://www.carqueryapi.com/api/0.3/?callback=&cmd=getMakes&sold_in_country={sold_in_country}"
    #logging.info(f"Request URL: {url}")
    try:
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses
        
        
        # Strip JSONP callback
        #json_str = response.text.strip('?()')  
        
        json_str = response.text[1:-2]     
        data = json.loads(json_str)
        #logging.info(f"data['Makes']: {data['Makes']}") 
        if 'Makes' in data:
            return data['Makes']
        else:
            raise HTTPException(status_code=500, detail="Invalid response format")        
        '''
        # Strip JSONP callback
        jsonp_prefix = "?("
        jsonp_suffix = ");"
        if response.text.startswith(jsonp_prefix) and response.text.endswith(jsonp_suffix):
            json_str = response.text[len(jsonp_prefix):-len(jsonp_suffix)]            
            data = json.loads(json_str)
            logging.info(f"Response Text: {data.get('Makes', [])}")
            return data.get('Makes', [])
        else:
            raise HTTPException(status_code=500, detail="Unexpected response format")
        '''
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        logging.error(f"Request exception: {e}")
        raise HTTPException(status_code=500, detail="Error fetching car makes")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="Error parsing JSON response")
        
@app.get("/car-models/")
async def get_car_models(make_id: str):
    try:
        url = f"{CARQUERY_API_URL}getModels&make={make_id}"
        logging.info(f"car model url: {url}")

        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses        
        logging.info(f"Car Models Response: {response.text}") 

        '''
        #print("Car Models Response:", response.text)  # Add this line for logging
        json_str = response.text[1:-2]     
        data = json.loads(json_str)
        return data.get('Models', data['Models'])    
        '''

        # Ensure correct JSONP prefix and suffix
        jsonp_prefix = "?("
        jsonp_suffix = ");"
        if response.text.startswith(jsonp_prefix) and response.text.endswith(jsonp_suffix):
            json_str = response.text[len(jsonp_prefix):-len(jsonp_suffix)]
            data = json.loads(json_str)
            return data.get('Models', [])
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        logging.error(f"Request exception: {e}")
        raise HTTPException(status_code=500, detail="Error fetching car makes")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="Error parsing JSON response")

#Predict car damage
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