from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional
import zipfile
import shutil
import json
from typing import Optional, Tuple, Dict, Any, Union
from dotenv import load_dotenv
from mangum import Mangum  # Required for AWS Lambda/Vercel compatibility
import tempfile
from app.utils.aiproxy import get_openai_response
from app.utils.functions import *
# Load environment variables
load_dotenv()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
UPLOAD_DIR = "static/uploads"
EXTRACT_DIR = "static/extracted"
OUTPUT_DIR = "static/outputs"

# Ensure all required directories exist
for directory in [UPLOAD_DIR, EXTRACT_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# FastAPI App
app = FastAPI(title="IITM Assignment API")
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
@app.get("/")
def read_root():
    return {"message": "FastAPI deployed on Vercel"}
@app.post("/api/")
async def process_question(
    question: str = Form(...), file: Optional[UploadFile] = File(None)
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


# New endpoint for testing specific functions
@app.post("/debug/{function_name}")
async def debug_function(
    function_name: str,
    file: Optional[UploadFile] = File(None),
    params: str = Form("{}"),
):
    """
    Debug endpoint to test specific functions directly

    Args:
        function_name: Name of the function to test
        file: Optional file upload
        params: JSON string of parameters to pass to the function
    """
    try:
        # Save file temporarily if provided
        temp_file_path = None
        if file:
            temp_file_path = await save_upload_file_temporarily(file)

        # Parse parameters
        parameters = json.loads(params)

        # Add file path to parameters if file was uploaded
        if temp_file_path:
            parameters["file_path"] = temp_file_path

        # Call the appropriate function based on function_name
        if function_name == "analyze_sales_with_phonetic_clustering":
            result = await analyze_sales_with_phonetic_clustering(**parameters)
            return {"result": result}
        elif function_name == "calculate_prettier_sha256":
            # For calculate_prettier_sha256, we need to pass the filename parameter
            if temp_file_path:
                result = await calculate_prettier_sha256(temp_file_path)
                return {"result": result}
            else:
                return {"error": "No file provided for calculate_prettier_sha256"}
        else:
            return {
                "error": f"Function {function_name} not supported for direct testing"
            }

    except Exception as e:
        import traceback

        return {"error": str(e), "traceback": traceback.format_exc()}

# ✅ Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.1"}

# ✅ Vercel compatibility (Place at the bottom)
handler = Mangum(app)

# ✅ Local Development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)