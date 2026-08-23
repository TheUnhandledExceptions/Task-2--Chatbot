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

# Hugging Face Spaces automatically runs the `app` ASGI instance, no need for manual uvicorn.run()
