import sys
import os
import spaces

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

# Import the FastAPI app
from main import app

# Hugging Face ZeroGPU requires at least one function decorated with @spaces.GPU
# We add a dummy FastAPI endpoint directly in app.py so the HF AST parser finds it!
@app.get("/hf_gpu_check")
@spaces.GPU
def dummy_gpu_check():
    return {"status": "GPU function registered successfully!"}
