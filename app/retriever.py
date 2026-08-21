import time
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class SearchResponse:
    def __init__(self, total_ms, embed_ms, search_ms):
        self.total_ms = total_ms
        self.embed_ms = embed_ms
        self.search_ms = search_ms

model = None
qdrant_client = None

def warmup():
    global model, qdrant_client
    if model is None:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    if qdrant_client is None:
        import shutil
        src_db = os.path.join(os.path.dirname(__file__), "..", "backend", "qdrant_db")
        benchmark_db = os.path.join(os.path.dirname(__file__), "..", "backend", "qdrant_db_benchmark")
        
        try:
            shutil.copytree(src_db, benchmark_db, dirs_exist_ok=True)
            # Remove the copied .lock file if it exists
            lock_file = os.path.join(benchmark_db, ".lock")
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception as e:
            print("Failed to copy database:", e)
            
        qdrant_client = QdrantClient(path=benchmark_db)
    
    # Warmup inference
    model.encode("warmup")
    qdrant_client.query_points(collection_name="msmarco_xi_indic", query=[0.0]*384, limit=1)

def search(query: str, top_k: int = 5) -> SearchResponse:
    start_total = time.time()
    
    start_embed = time.time()
    query_vector = model.encode(query).tolist()
    embed_ms = (time.time() - start_embed) * 1000
    
    start_search = time.time()
    qdrant_client.query_points(collection_name="msmarco_xi_indic", query=query_vector, limit=top_k)
    search_ms = (time.time() - start_search) * 1000
    
    total_ms = (time.time() - start_total) * 1000
    return SearchResponse(total_ms, embed_ms, search_ms)
