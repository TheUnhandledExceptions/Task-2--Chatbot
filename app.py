import sys
import os

# Add backend directory to path so imports inside backend work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

import gradio as gr
from main import app as fastapi_app

def dummy():
    return "Voice RAG FastAPI Backend is live!"

demo = gr.Interface(fn=dummy, inputs=None, outputs="text")

# Hugging Face looks for a variable named 'app'
# We mount Gradio to /ui so our /query FastAPI endpoint remains untouched
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
