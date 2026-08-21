import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from chunking import AdvancedChunker

model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
print(f"Loading embedding model: {model_name}")
model = SentenceTransformer(model_name)

qdrant_path = "./qdrant_db"
client = QdrantClient(path=qdrant_path)
collection_name = "msmarco_xi_indic"

try:
    client.get_collection(collection_name)
    print(f"Collection {collection_name} already exists.")
except Exception:
    print(f"Creating collection {collection_name}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
    )

def index_mock_data():
    print("Indexing mock data for quick testing...")
    dummy_texts = [
        "The capital of India is New Delhi. It is a vibrant city with a rich history.",
        "Sachin Tendulkar is known as the God of Cricket in India.",
        "The Taj Mahal is located in Agra and was built by Shah Jahan.",
        "Sarvam AI is an Indian AI startup building models for Indic languages.",
        "HackerHouse Goa is a great place to build cool AI projects."
    ]
    
    from chunking import SemanticChunker, MetadataAwareChunker
    base_chunker = SemanticChunker(model=model, max_chunk_size=400, similarity_threshold=0.5)
    chunker = MetadataAwareChunker(base_chunker)
    points = []
    point_id = 0
    
    for i, text in enumerate(dummy_texts):
        metadata = {"id": str(i), "lang": "en", "original_query": ""}
        chunks = chunker.chunk(text, metadata)
        
        for c in chunks:
            chunk_text = c["text"]
            chunk_metadata = c["metadata"]
            embedding = model.encode(chunk_text).tolist()
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": chunk_text, **chunk_metadata}
                )
            )
            point_id += 1
            
    client.upsert(collection_name=collection_name, points=points)
    print("Mock indexing complete.")

if __name__ == "__main__":
    index_mock_data()
