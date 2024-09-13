#Capstone 1 implementation starts here
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

#Capstone 2 imports starts here
from fastapi import Form
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

app = FastAPI()

origins = [
    "http://192.168.1.3:8000"
    #"http://192.168.34.160:8000"
    #"http://172.0.40.25:8000"
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

#Capstone 2 implementation starts here

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# Initialize Chroma client
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()

# Define Pydantic model for API input
class QueryInput(BaseModel):
    query: str

# Helper function to generate multi-query answers
def generate_multi_query(query, context, model="gpt-3.5-turbo"):
    prompt = """
    You are a knowledgable insurance agent. 
    Your users are inquiring about car damage related insurance coverage. 
    """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Based on the following context:\n\n{context}\n\nAnswer the query: '{query}'"},
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    return content

# New endpoint for uploading the PDF and getting the answer
import time
import logging

@app.post("/get_answer/")
async def get_answer(query: str = Form(...), file: UploadFile = File(...)):
    # Start total processing time
    total_start_time = time.time()

    # Read the uploaded PDF file
    pdf_content = await file.read()

    # Log time for PDF reading
    pdf_read_start_time = time.time()
    
    # Load the PDF content using PdfReader
    reader = PdfReader(file.file)
    pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
    
    # Log PDF reading time
    pdf_read_time = time.time() - pdf_read_start_time
    logging.info(f"PDF reading time: {pdf_read_time:.4f} seconds")

    # Split text using RecursiveCharacterTextSplitter
    text_split_start_time = time.time()
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

    # Log text splitting time
    text_split_time = time.time() - text_split_start_time
    logging.info(f"Text splitting time: {text_split_time:.4f} seconds")

    # Split text into smaller chunks using SentenceTransformersTokenTextSplitter
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    # Document retrieval from ChromaDB
    retrieval_start_time = time.time()
    chroma_collection = chroma_client.get_or_create_collection(
        "cardamage-collect", embedding_function=embedding_function
    )
    ids = [str(i) for i in range(len(token_split_texts))]
    chroma_collection.add(ids=ids, documents=token_split_texts)

    # Query the collection
    results = chroma_collection.query(
        query_texts=[query], n_results=10, include=["documents", "embeddings"]
    )
    retrieved_documents = results["documents"][0]

    # Log document retrieval time
    retrieval_time = time.time() - retrieval_start_time
    logging.info(f"Document retrieval time from ChromaDB: {retrieval_time:.4f} seconds")

    # Use CrossEncoder to score documents
    cross_encoder_start_time = time.time()
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, doc] for doc in retrieved_documents]
    scores = cross_encoder.predict(pairs)

    # Log document scoring time
    cross_encoder_time = time.time() - cross_encoder_start_time
    logging.info(f"Document scoring time (CrossEncoder): {cross_encoder_time:.4f} seconds")

    # Sort and retrieve top documents
    top_indices = np.argsort(scores)[::-1][:5]
    top_documents = [retrieved_documents[i] for i in top_indices]
    context = "\n\n".join(top_documents)

    # Generate the final answer using the OpenAI model
    response_generation_start_time = time.time()
    final_answer = generate_multi_query(query=query, context=context)

    # Log response generation time
    response_generation_time = time.time() - response_generation_start_time
    logging.info(f"Response generation time (GPT-3.5): {response_generation_time:.4f} seconds")

    # Total processing time
    total_processing_time = time.time() - total_start_time
    logging.info(f"Total processing time: {total_processing_time:.4f} seconds")

    return JSONResponse(content={
        "query": query,
        "answer": final_answer,
        "processing_times": {
            "pdf_read_time": pdf_read_time,
            "text_split_time": text_split_time,
            "retrieval_time": retrieval_time,
            "document_scoring_time": cross_encoder_time,
            "response_generation_time": response_generation_time,
            "total_processing_time": total_processing_time
        }
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.3", port=8000)
    #uvicorn.run(app, host="192.168.34.160", port=8000)
    #uvicorn.run(app, host="172.0.40.25", port=8000)
    