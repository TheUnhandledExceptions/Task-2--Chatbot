import os
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from chunking import AdvancedChunker

# Load Multilingual Embedding Model
# We use multilingual because MSMARCO-XI contains Indic languages
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
print(f"Loading embedding model: {model_name}")
model = SentenceTransformer(model_name)

# Initialize Qdrant client (local disk storage)
qdrant_path = "./qdrant_db"
client = QdrantClient(path=qdrant_path)

collection_name = "msmarco_xi_indic"

# Create collection if it doesn't exist
try:
    client.get_collection(collection_name)
    print(f"Collection {collection_name} already exists.")
except Exception:
    print(f"Creating collection {collection_name}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
    )

def index_dataset(subset_size=1000, lang="hi"):
    print(f"Loading dataset ai4bharat/MSMARCO-XI (lang: {lang})...")
    # Using streaming or loading a small subset for demonstration
    dataset = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
    
    from chunking import SemanticChunker, MetadataAwareChunker
    base_chunker = SemanticChunker(model=model, max_chunk_size=400, similarity_threshold=0.5)
    chunker = MetadataAwareChunker(base_chunker)
    
    points = []
    point_id = 0
    
    print(f"Processing and chunking first {subset_size} records...")
    for i, row in enumerate(dataset):
        if i >= subset_size:
            break
            
        text = row.get("passage", "") or row.get("text", "")
        # The dataset format for MSMARCO typically has 'query', 'passages', etc.
        # AI4Bharat format: 'query_id', 'query', 'passage_id', 'passage', 'is_selected'
        if "passage" in row:
            text = row["passage"]
        elif "text" in row:
            text = row["text"]
            
        if not text:
            continue
            
        metadata = {
            "id": row.get("passage_id", str(i)),
            "lang": lang,
            "original_query": row.get("query", "")
        }
        
        chunks = chunker.chunk(text, metadata)
        
        for c in chunks:
            chunk_text = c["text"]
            chunk_metadata = c["metadata"]
            
            # Create embedding
            embedding = model.encode(chunk_text).tolist()
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": chunk_text, **chunk_metadata}
                )
            )
            point_id += 1
            
            # Batch upload
            if len(points) >= 100:
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"Upserted 100 points. Total: {point_id}")
                points = []
                
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Upserted remaining {len(points)} points. Total: {point_id}")
        
    print("Indexing complete.")

if __name__ == "__main__":
    index_dataset(subset_size=1000, lang="hi")
