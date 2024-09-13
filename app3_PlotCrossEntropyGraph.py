#pip install umap-learn
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
from helper import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
import umap.umap_ as umap  # Correct UMAP import
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

reader = PdfReader("data/OG-24-9910-1801-00189571.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]

# Split the text into smaller chunks
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

# Extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

query = "Are external damages covered in Insurance Coverage ?"

results = chroma_collection.query(query_texts=[query], n_results=10)
retrieved_documents = results["documents"][0]

def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """You are a helpful expert financial research assistant. 
   Provide an example answer to the given question, that might be found in a document like an annual report."""
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

original_query = "I have a cracked windshiled, is it covered under my Insurance Coverage ?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))

results = chroma_collection.query(
    query_texts=joint_query, n_results=10, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0]

# Retrieve embeddings and project them
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

retrieved_embeddings = results["embeddings"][0]
original_query_embedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

# Plot the projected query and retrieved documents in the embedding space
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.3", port=8000)
    #uvicorn.run(app, host="192.168.34.160", port=8000)
    #uvicorn.run(app, host="172.0.40.25", port=8000)
    