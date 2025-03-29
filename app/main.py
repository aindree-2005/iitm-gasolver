from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import os
import zipfile
import pandas as pd
import shutil
import uuid
from typing import Optional, Tuple, Dict, Any, Union
from dotenv import load_dotenv
from mangum import Mangum  # Required for AWS Lambda/Vercel compatibility

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

@app.get("/")
def read_root():
    return {"message": "FastAPI deployed on Vercel"}

# ✅ API that supports file uploads & questions
@app.post("/api/")
async def process_question(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    try:
        # Generate a unique session ID for this request
        session_id = str(uuid.uuid4())
        
        # Process the uploaded file if it exists
        file_info = None
        if file:
            file_info = await handle_uploaded_file(file, session_id)
        
        # Process the question and file
        result = process_query(question, file_info, session_id)
        
        # Return appropriate response based on result type
        if isinstance(result, str) and os.path.isfile(result):
            # If the result is a file path, return a download link
            filename = os.path.basename(result)
            return {"file_url": f"/download/{filename}", "type": "file"}
        else:
            # Otherwise, return the text answer
            return {"answer": result, "type": "text"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# ✅ Enhanced file handling function
async def handle_uploaded_file(upload_file: UploadFile, session_id: str) -> Dict[str, Any]:
    """
    Handle the uploaded file including extraction if it's a zip file.
    Returns file information dictionary.
    """
    # Generate unique filename to prevent collisions
    original_filename = upload_file.filename
    file_extension = os.path.splitext(original_filename)[1].lower()
    unique_filename = f"{session_id}{file_extension}"
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    
    file_info = {
        "original_filename": original_filename,
        "saved_path": file_path,
        "file_type": file_extension,
        "session_id": session_id,
        "extracted_files": []
    }
    
    # If it's a zip file, extract it
    if file_extension == ".zip":
        extract_path = os.path.join(EXTRACT_DIR, session_id)
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        # Record extracted files
        extracted_files = []
        for root, _, files in os.walk(extract_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, extract_path)
                extracted_files.append({
                    "filename": filename,
                    "path": file_path,
                    "relative_path": rel_path
                })
        
        file_info["extracted_files"] = extracted_files
        file_info["extraction_dir"] = extract_path
    
    return file_info

# ✅ Improved query processing function
def process_query(question: str, file_info: Optional[Dict[str, Any]], session_id: str) -> Union[str, Dict[str, Any]]:
    """
    Process the query based on the question and file information.
    Returns either a file path (string) or a text answer.
    """
    # Check if the question is asking to extract a zip file and read CSV
    if file_info and "extract" in question.lower() and "zip" in question.lower() and "csv" in question.lower():
        # Check if we have already extracted files
        if file_info.get("extracted_files"):
            # Look for CSV files in the extracted contents
            csv_files = [f for f in file_info["extracted_files"] if f["filename"].lower().endswith(".csv")]
            
            if csv_files:
                # Process the first CSV file found
                csv_path = csv_files[0]["path"]
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Check if the question is asking about specific columns
                    if "answer column" in question.lower():
                        if "answer" in df.columns:
                            return f"The values in the 'answer' column are: {', '.join(map(str, df['answer'].tolist()))}"
                        else:
                            return f"The CSV file does not contain an 'answer' column. Available columns are: {', '.join(df.columns)}"
                    
                    # If we need to return the processed file
                    output_file = os.path.join(OUTPUT_DIR, f"processed_{session_id}.csv")
                    df.to_csv(output_file, index=False)
                    return output_file
                    
                except Exception as e:
                    return f"Error processing CSV file: {str(e)}"
            else:
                return "No CSV files found in the extracted zip file."
        else:
            return "No files were extracted or the uploaded file is not a zip file."
            
    # Handle other types of questions or file operations
    # This is where you would integrate your actual AI processing logic
    
    if file_info:
        # Create a sample output file as a demonstration
        output_path = os.path.join(OUTPUT_DIR, f"result_{session_id}.txt")
        with open(output_path, "w") as f:
            f.write(f"Processed result for question: {question}\n")
            f.write(f"Based on file: {file_info.get('original_filename', 'unknown')}")
        return output_path
    else:
        return f"Answer to your question: {question}\n(This is where your AI model would process the question)"

# ✅ Serve an HTML+JS frontend at /web with improved UI
@app.get("/web", response_class=HTMLResponse)
async def serve_web():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IITM Assignment Solver</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
            }
            input[type="text"] {
                width: 100%;
                padding: 8px;
                box-sizing: border-box;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                cursor: pointer;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                min-height: 50px;
            }
            .loading {
                display: none;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>IITM Assignment Solver</h1>
        <p>Upload a file and ask questions about it</p>
        
        <form id="upload-form">
            <div class="form-group">
                <label for="question">Question:</label>
                <input type="text" id="question" name="question" placeholder="E.g., Extract data from this zip file and analyze the CSV inside" required>
            </div>
            
            <div class="form-group">
                <label for="file">Upload File (optional):</label>
                <input type="file" id="file" name="file">
            </div>
            
            <button type="submit">Submit</button>
        </form>
        
        <div class="loading" id="loading">Processing your request...</div>
        
        <div id="result"></div>

        <script>
            document.getElementById("upload-form").onsubmit = async function(event) {
                event.preventDefault();
                
                // Show loading indicator
                document.getElementById("loading").style.display = "block";
                document.getElementById("result").innerHTML = "";
                
                let formData = new FormData();
                formData.append("question", document.getElementById("question").value);
                let fileInput = document.getElementById("file").files[0];
                if (fileInput) formData.append("file", fileInput);

                try {
                    let response = await fetch("/api/", {
                        method: "POST",
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
                    }

                    let data = await response.json();
                    
                    // Hide loading indicator
                    document.getElementById("loading").style.display = "none";
                    
                    if (data.file_url) {
                        document.getElementById("result").innerHTML = '<p>Process complete! <a href="' + data.file_url + '" download>Download Output File</a></p>';
                    } else if (data.answer) {
                        document.getElementById("result").innerHTML = '<p><strong>Answer:</strong></p><pre>' + data.answer + '</pre>';
                    } else {
                        document.getElementById("result").innerHTML = '<p>Error: Unexpected response format</p>';
                    }
                } catch (error) {
                    document.getElementById("loading").style.display = "none";
                    document.getElementById("result").innerHTML = '<p>Error: ' + error.message + '</p>';
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ✅ Serve downloaded files with improved error handling
@app.get("/download/{filename}")
async def download_file(filename: str):
    # Look in all possible directories for the file
    possible_paths = [
        os.path.join(UPLOAD_DIR, filename),
        os.path.join(OUTPUT_DIR, filename)
    ]
    
    for file_path in possible_paths:
        if os.path.exists(file_path):
            return FileResponse(
                file_path, 
                media_type="application/octet-stream", 
                filename=filename
            )
    
    raise HTTPException(status_code=404, detail="File not found")

# ✅ Debugging: Check Env Vars
@app.get("/debug-env")
def debug_env():
    return {
        "AIPROXY_TOKEN": bool(os.getenv("AIPROXY_TOKEN")),
        "UPLOAD_DIR": UPLOAD_DIR,
        "EXTRACT_DIR": EXTRACT_DIR,
        "OUTPUT_DIR": OUTPUT_DIR
    }

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