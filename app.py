import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

import gradio as gr
import spaces
from main import app as fastapi_app

@spaces.GPU
def dummy():
    return "Voice RAG FastAPI Backend is live!"

demo = gr.Interface(fn=dummy, inputs=None, outputs="text")
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
