from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os
from typing import Optional
from app.utils.aiproxy import get_openai_response
import shutil
import tempfile
from dotenv import load_dotenv
from mangum import Mangum  # Required for AWS Lambda/Vercel compatibility

# Load environment variables
load_dotenv()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"

# FastAPI App
app = FastAPI(title="IITM Assignment API")

@app.get("/")
def read_root():
    return {"message": "FastAPI deployed on Vercel"}

# ✅ API that supports file uploads & questions
@app.post("/api/")
async def process_question(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    try:
        temp_file_path = None
        if file:
            temp_file_path = await save_upload_file_temporarily(file)
        
        answer = await get_openai_response(question, temp_file_path)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Serve an HTML+JS frontend at /web
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
                document.getElementById("result").innerText = "Answer: " + (data.answer || data.detail);
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def save_upload_file_temporarily(upload_file: UploadFile) -> str:
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, upload_file.filename)
        with open(file_path, "wb") as f:
            contents = await upload_file.read()
            f.write(contents)
        return file_path
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e

@app.get("/debug-env")
def debug_env():
    return {"AIPROXY_TOKEN": os.getenv("AIPROXY_TOKEN")}

# Vercel compatibility (Place at the bottom)
handler = Mangum(app)

# Local Development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
