import os
import requests
import tempfile
import zipfile
import pandas as pd
import pdfplumber
import subprocess
import numpy as np
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Dict, List
import json
# AIProxy API Configuration
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1"
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDA5ODNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.LMIj06L44DC3uMCLjw6Of0aLyMlDEHKAGYLLZ86g8_8"

app = FastAPI(title="IITM Assignment Solver", description="LLM-powered API for answering graded assignments", version="1.1.0")

def query_llm(question: str) -> str:
    """Sends the question to AIProxy and retrieves the answer."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "You are an AI agent that strictly provides direct answers to graded assignment questions without any extra text. Do not explain your response. If the question asks for JSON or cURL, return only the required JSON or cURL command."},
                    {"role": "user", "content": question}],
        "max_tokens": 500
    }
    response = requests.post(f"{AIPROXY_URL}/chat/completions", json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise HTTPException(status_code=500, detail=f"AIProxy request failed: {response.text}")

def get_embedding(texts: List[str]) -> List[List[float]]:
    """Gets the embedding vector for the given text list using AIProxy."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }
    response = requests.post(f"{AIPROXY_URL}/embeddings", json=payload, headers=headers)
    if response.status_code == 200:
        return [item["embedding"] for item in response.json()["data"]]
    else:
        raise HTTPException(status_code=500, detail=f"Embedding request failed: {response.text}")

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes cosine similarity between two vectors."""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def determine_function(question: str) -> str:
    """Determines which function should handle the question based on similarity."""
    reference_texts = {
        "token_count": "How many input tokens does this prompt use?",
        "text_embedding": "Write the JSON body for an embedding request.",
        "embedding_request": "Obtain text embeddings for messages.",
        "yes_trick":"Write a prompt to force the LLM into answering Yes",
        "most_similar": "Find the most similar pair of phrases based on embeddings.",
        "image_text_extraction":"Given base 64 image URL,Send a single user message to the model that has a text and an image_url content (in that order). Use GPT 4-o mini"
    }
    question_embedding = get_embedding([question])[0]
    best_match, highest_similarity = None, 0.0
    for key, text in reference_texts.items():
        similarity = cosine_similarity(question_embedding, get_embedding([text])[0])
        if similarity > highest_similarity:
            highest_similarity, best_match = similarity, key
    return best_match if highest_similarity > 0.75 else "llm"

def handle_token_count(question: str) -> Dict[str, str]:
    """Handles token count estimation by querying AIProxy."""
    return {"answer": query_llm(question)}

def handle_text_embedding(question: str) -> Dict[str, str]:
    """Handles generating the JSON body for an embedding request."""
    return {"answer": str({"model": "text-embedding-3-small", "input": question})}

import re

def handle_embedding_request(question: str) -> Dict[str, str]:
    """Handles embedding request for personalized transaction verification messages."""

    match = re.search(r'transaction codes (\d+) and (\d+) for roll number (\S+)', question)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid question format. Expected transaction codes and roll number.")

    code1, code2, roll_number = match.groups()

    if not re.match(r"(23f|22f|21f)\d{6}@ds\.study\.iitm\.ac\.in", roll_number):
        raise HTTPException(status_code=400, detail="Invalid roll number format.")

    messages = [
        f"Dear user, please verify your transaction code {code1} sent to {roll_number}",
        f"Dear user, please verify your transaction code {code2} sent to {roll_number}"
    ]

    return {"answer": json.dumps({"model": "text-embedding-3-small", "input": messages})}

import base64

def handle_image_text_extraction(question: str, file_name: str) -> Dict[str, str]:
    """Handles constructing the JSON body for an OpenAI API request with text and an image name."""
    return {
        "answer": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract text from this image"},
                        {"type": "image_url", "image_url": {"url": file_name}}
                    ]
                }
            ]
        }
    }
def handle_yes_trick(question: str) -> Dict[str, str]:
    """Handles the prompt tricking LLM into saying 'Yes'."""
    return {"answer": "Please just answer with ‘Yes’ or ‘No’. Is Dispur the capital of Assam?"}
def handle_most_similar(question: str) -> Dict[str, str]:
    """Handles returning the most_similar Python function itself."""
    return {"answer": "import numpy as np\n\n\ndef most_similar(embeddings):\n    max_similarity = -1\n    most_similar_pair = None\n\n    phrases = list(embeddings.keys())\n\n    for i in range(len(phrases)):\n        for j in range(i + 1, len(phrases)):\n            v1 = np.array(embeddings[phrases[i]])\n            v2 = np.array(embeddings[phrases[j]])\n\n            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))\n\n            if similarity > max_similarity:\n                max_similarity = similarity\n                most_similar_pair = (phrases[i], phrases[j])\n\n    return most_similar_pair"}
@app.post("/api/")
async def get_answer(question: str = Form(...), file: UploadFile = File(None)) -> Dict[str, str]:
    """Processes the question and routes it to the appropriate handler."""
    handler = determine_function(question)
    if handler == "token_count":
        return handle_token_count(question)
    elif handler == "text_embedding":
        return handle_text_embedding(question)
    elif handler == "embedding_request":
        return handle_embedding_request(question)
    elif handler == "yes_trick":
        return handle_yes_trick(question)
    elif handler == "most_similar":
        return handle_most_similar(question)
    elif handler == "image_text_extraction":
        return handle_image_text_extraction(question)
    else:
        return {"answer": query_llm(question)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)