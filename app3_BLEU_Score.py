#pip install umap-learn, nltk, chromadb, sentence-transformers, pypdf
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from dotenv import load_dotenv
from pypdf import PdfReader
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import umap.umap_ as umap  # Correct UMAP import
import matplotlib.pyplot as plt

# Initialize BLEU score package and download 'punkt' tokenizer
nltk.download('punkt')

# Initialize FastAPI app
app = FastAPI()

# Set CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=openai_key)

# Define class names for car damage detection
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

# Load pre-trained YOLO model for car damage detection
model_path = "model/train6/weights/best.pt"
model = YOLO(model_path)

# Initialize Chroma client
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()

# BLEU Score Calculation
def compute_bleu_score(reference, generated_response):
    reference_tokens = nltk.word_tokenize(reference)
    generated_tokens = nltk.word_tokenize(generated_response)
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothie)
    return score

# Generate the final answer using the OpenAI model and calculate BLEU score
def generate_multi_query(query, context, reference_answer, model="gpt-3.5-turbo"):
    prompt = """
    You are a knowledgeable insurance agent. 
    Your users are inquiring about car damage related insurance coverage. 
    """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Based on the following context:\n\n{context}\n\nAnswer the query: '{query}'"},
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content

    # Compute BLEU score
    bleu_score = compute_bleu_score(reference_answer, content)
    logging.info(f"Generated Answer: {content}")
    logging.info(f"BLEU Score: {bleu_score}")
    
    return content, bleu_score

# Endpoint for uploading the PDF and getting the answer
@app.post("/get_answer/")
async def get_answer(query: str = Form(...), file: UploadFile = File(...)):
    # Read the uploaded PDF file
    pdf_content = await file.read()
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

    # Reference answer for BLEU comparison (Example: Replace with correct reference answer)
    reference_answer = "Yes, cracked windshields are typically covered under comprehensive insurance policies."

    # Generate the final answer using OpenAI GPT-3.5 Turbo and calculate BLEU score
    final_answer, bleu_score = generate_multi_query(query=query, context="\n\n".join(retrieved_documents), reference_answer=reference_answer)

    return JSONResponse(content={"query": query, "answer": final_answer, "bleu_score": bleu_score})

# Function to visualize embeddings (Optional but used in your code)
def project_embeddings(embeddings, transform):
    return transform.transform(embeddings)

# Plot the projected query and retrieved documents in the embedding space
def plot_projection(query_embedding, augmented_query_embedding, dataset_embeddings, retrieved_embeddings):
    plt.figure()
    plt.scatter(dataset_embeddings[:, 0], dataset_embeddings[:, 1], s=10, color="gray")
    plt.scatter(retrieved_embeddings[:, 0], retrieved_embeddings[:, 1], s=100, facecolors="none", edgecolors="g")
    plt.scatter(query_embedding[:, 0], query_embedding[:, 1], s=150, marker="X", color="r")
    plt.scatter(augmented_query_embedding[:, 0], augmented_query_embedding[:, 1], s=150, marker="X", color="orange")
    plt.gca().set_aspect("equal", "datalim")
    plt.title("Query and Document Embeddings")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.3", port=8000)