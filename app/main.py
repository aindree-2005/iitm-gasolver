from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional
from app.utils.aiproxy import get_openai_response
import shutil
import tempfile
from fastapi import UploadFile
from app.utils.functions import *
import httpx
from dotenv import load_dotenv
load_dotenv()

AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDA5ODNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.LMIj06L44DC3uMCLjw6Of0aLyMlDEHKAGYLLZ86g8_8"
AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"


app = FastAPI(title="IITM Assignment API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
async def save_upload_file_temporarily(upload_file: UploadFile) -> str:
    """
    Save an upload file temporarily and return the path to the saved file.
    """
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create a path to save the file
        file_path = os.path.join(temp_dir, upload_file.filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            contents = await upload_file.read()
            f.write(contents)
        
        # Return the path to the saved file
        return file_path
    except Exception as e:
        # Clean up the temporary directory if an error occurs
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e

@app.post("/api/")
async def process_question(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    try:
        # Save file temporarily if provided
        temp_file_path = None
        if file:
            temp_file_path = await save_upload_file_temporarily(file)
        
        # Get answer from OpenAI
        answer = await get_openai_response(question, temp_file_path)
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
