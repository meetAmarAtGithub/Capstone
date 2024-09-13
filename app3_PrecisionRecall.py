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
from sklearn.metrics import precision_score, recall_score
from fuzzywuzzy import fuzz
import logging

# Helper function to normalize text (for better comparison)
def normalize_text(text):
    return text.strip().lower().replace("\n", " ").replace("\t", " ")

# Helper function for fuzzy matching
def fuzzy_match(doc, expected_docs, threshold=70):
    for expected_doc in expected_docs:
        # Fuzzy matching with a threshold (e.g., 70% match)
        if fuzz.partial_ratio(doc, expected_doc) >= threshold:
            return True
    return False

@app.post("/get_answer/")
async def get_answer(query: str = Form(...), file: UploadFile = File(...)):
    # Read the uploaded PDF file
    pdf_content = await file.read()
    
    # Load the PDF content using PdfReader
    reader = PdfReader(file.file)
    pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]

    # Split text using RecursiveCharacterTextSplitter
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

    # Split text into smaller chunks using SentenceTransformersTokenTextSplitter
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    # Create a collection in ChromaDB and add documents
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

    # Use CrossEncoder to score documents
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, doc] for doc in retrieved_documents]
    scores = cross_encoder.predict(pairs)

    # Sort and retrieve top documents
    top_indices = np.argsort(scores)[::-1][:5]
    top_documents = [retrieved_documents[i] for i in top_indices]
    context = "\n\n".join(top_documents)

    # Normalize retrieved documents
    normalized_top_documents = [normalize_text(doc) for doc in top_documents]

    # Updated Ground Truth (expected relevant documents for this query)
    expected_relevant_docs = [
        'roadside assistance',
        'death of or bodily injury',
        'damage to third party property - rs. 7.5 lakhs',
        'cover for owner-driver under section iii (csi) rs 1500000',
        'deductible under section - i: rs 1000'
    ]
    
    # Normalize the expected relevant documents
    normalized_expected_docs = [normalize_text(doc) for doc in expected_relevant_docs]

    # Debugging: Log the retrieved and expected documents to see what is being compared
    logging.info(f"Retrieved documents: {normalized_top_documents}")
    logging.info(f"Expected relevant documents: {normalized_expected_docs}")

    # Create binary vectors for precision and recall calculation using fuzzy matching
    y_true = []
    y_pred = []

    for retrieved_doc in normalized_top_documents:
        is_relevant = fuzzy_match(retrieved_doc, normalized_expected_docs, threshold=70)  # Set threshold as needed
        y_true.append(1 if is_relevant else 0)
        y_pred.append(1)  # We assume all retrieved documents are predicted relevant

    # Calculate precision and recall
    if len(y_true) > 0 and len(y_pred) > 0:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
    else:
        precision = 0.0
        recall = 0.0

    logging.info(f"Precision: {precision}, Recall: {recall}")
    print(f"Precision: {precision}, Recall: {recall}")

    # Generate the final answer using the OpenAI model
    final_answer = generate_multi_query(query=query, context=context)

    return JSONResponse(content={"query": query, "answer": final_answer, "precision": precision, "recall": recall})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.3", port=8000)
    #uvicorn.run(app, host="192.168.34.160", port=8000)
    #uvicorn.run(app, host="172.0.40.25", port=8000)
    