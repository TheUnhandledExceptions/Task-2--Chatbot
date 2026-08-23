from app.retriever import warmup, model

# Ensure the model is loaded when this module is imported by the eval loop
warmup()

def embed(text: str) -> list[float]:
    """
    Returns the dense vector embedding for a given text.
    This function is required by the rag-local-eval-loop for the RETRIEVAL checks.
    """
    return list(model.embed([text]))[0].tolist()
