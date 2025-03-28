from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import os
from typing import Optional
from app.utils.aiproxy import get_openai_response
import shutil
import tempfile
from dotenv import load_dotenv
from mangum import Mangum  # Required for AWS Lambda/Vercel compatibility
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"

# FastAPI App
app = FastAPI(title="IITM Assignment API")

# Directory for storing uploaded and output files
UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "static/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files for access
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return {"message": "FastAPI deployed on Vercel"}

# ✅ API for file uploads & questions
@app.post("/api/")
async def process_question(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    try:
        temp_file_path = None
        if file:
            temp_file_path = await save_upload_file(file)

        answer = await get_openai_response(question, temp_file_path)

        # If the output is a file, provide a downloadable link
        if os.path.exists(answer):  # Assuming answer is a file path
            output_filename = os.path.basename(answer)
            output_url = f"/static/outputs/{output_filename}"
            shutil.move(answer, os.path.join(OUTPUT_DIR, output_filename))
            return {"file_url": output_url}

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Serve HTML+JS frontend at /web
@app.get("/web", response_class=HTMLResponse)
async def serve_web():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IITM Assignment Solver</title>
    </head>
    <body>
        <h2>Ask a Question with File Upload</h2>
        <form id="upload-form">
            <input type="text" id="question" name="question" placeholder="Enter your question" required>
            <input type="file" id="file" name="file">
            <button type="submit">Submit</button>
        </form>
        <p id="result"></p>

        <script>
            document.getElementById("upload-form").onsubmit = async function(event) {
                event.preventDefault();
                let formData = new FormData();
                formData.append("question", document.getElementById("question").value);
                let fileInput = document.getElementById("file").files[0];
                if (fileInput) formData.append("file", fileInput);

                let response = await fetch("/api/", {
                    method: "POST",
                    body: formData
                });

                let data = await response.json();
                let resultElement = document.getElementById("result");

                if (data.file_url) {
                    resultElement.innerHTML = '<a href="' + data.file_url + '" target="_blank">Download Output File</a>';
                } else {
                    resultElement.innerText = "Answer: " + (data.answer || data.detail);
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ✅ Serve a stored file if needed
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def save_upload_file(upload_file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_DIR, upload_file.filename)
    with open(file_path, "wb") as f:
        f.write(await upload_file.read())
    return file_path

@app.get("/debug-env")
def debug_env():
    return {"AIPROXY_TOKEN": os.getenv("AIPROXY_TOKEN")}

# Vercel compatibility (Place at the bottom)
handler = Mangum(app)

# Local Development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
