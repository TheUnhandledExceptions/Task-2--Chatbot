import sys
import os

# Add backend directory to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

import gradio as gr
from main import app as fastapi_app

# Create a dummy Gradio interface to satisfy Hugging Face Spaces if it looks for one
def dummy():
    return "Voice RAG FastAPI Backend is live!"

demo = gr.Interface(fn=dummy, inputs=None, outputs="text")

# Mount the FastAPI app onto the Gradio interface
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces expects the app to bind to port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
