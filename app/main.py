from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import os
from typing import Optional
import shutil
from dotenv import load_dotenv
from mangum import Mangum  # Required for AWS Lambda/Vercel compatibility

# Load environment variables
load_dotenv()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure upload directory exists

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
        saved_file_path = None
        if file:
            saved_file_path = await save_uploaded_file(file)

        # 🔥 Mocked function (Replace this with real AI processing logic)
        output_path, answer = process_question_logic(question, saved_file_path)

        # 📂 If output is a file, return a download link
        if output_path:
            return {"file_url": f"/download/{os.path.basename(output_path)}"}
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
                if (data.file_url) {
                    document.getElementById("result").innerHTML = '<a href="' + data.file_url + '" download>Download Output File</a>';
                } else {
                    document.getElementById("result").innerText = "Answer: " + (data.answer || data.detail);
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ✅ Serve downloaded files
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)

# ✅ Save uploaded file
async def save_uploaded_file(upload_file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_DIR, upload_file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return file_path

# ✅ Mocked AI Processing (Replace with real logic)
def process_question_logic(question: str, file_path: Optional[str]):
    # Simulate AI processing output
    if file_path:  # If a file is uploaded, assume output is a new processed file
        output_path = os.path.join(UPLOAD_DIR, "processed_" + os.path.basename(file_path))
        shutil.copy(file_path, output_path)  # Just simulating file processing
        return output_path, None
    else:
        return None, f"Mock AI Response for: {question}"

# ✅ Debugging: Check Env Vars
@app.get("/debug-env")
def debug_env():
    return {"AIPROXY_TOKEN": os.getenv("AIPROXY_TOKEN")}

# ✅ Vercel compatibility (Place at the bottom)
handler = Mangum(app)

# ✅ Local Development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
