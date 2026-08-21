import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from orchestrator import RAGOrchestrator
from analytics import log_latency, get_analytics

app = FastAPI(title="Voice RAG API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = None

@app.on_event("startup")
async def startup_event():
    global orchestrator
    orchestrator = RAGOrchestrator()

@app.post("/query")
async def query_audio(file: UploadFile = File(...)):
    # Save the audio file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Process through RAG pipeline
        result = await orchestrator.process(temp_path)
        
        # Log latency if successful
        if "error" not in result or result.get("error") == "Query flagged as unsafe or off-topic.":
            log_latency(result["timings"])
            
        return result
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/analytics")
async def analytics():
    return get_analytics()
