import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class ChunkingStrategy:
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

class AdvancedChunker(ChunkingStrategy):
    def __init__(self, max_chunk_size: int = 500, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        sentences = re.split(r'(?<=[.!?|।])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append({"text": current_chunk.strip(), "metadata": metadata.copy()})
                overlap_text = ""
                if len(current_chunk) > self.overlap:
                    overlap_text = current_chunk[-self.overlap:]
                    space_idx = overlap_text.find(" ")
                    if space_idx != -1:
                        overlap_text = overlap_text[space_idx+1:]
                current_chunk = overlap_text + sentence + " "
                
        if current_chunk.strip():
            chunks.append({"text": current_chunk.strip(), "metadata": metadata.copy()})
            
        for i, c in enumerate(chunks):
            c["metadata"]["chunk_id"] = f"{metadata.get('id', 'unknown')}_{i}"
            c["metadata"]["total_chunks"] = len(chunks)
            c["metadata"]["strategy"] = "advanced_overlap"
            
        return chunks

class SemanticChunker(ChunkingStrategy):
    """Chunks text by grouping semantically similar sentences."""
    def __init__(self, model: SentenceTransformer, max_chunk_size: int = 500, similarity_threshold: float = 0.5):
        self.model = model
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?|।])\s+', text) if s.strip()]
        if not sentences:
            return []
            
        embeddings = self.model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            emb = embeddings[i]
            
            # Cosine similarity
            sim = np.dot(current_embedding, emb) / (np.linalg.norm(current_embedding) * np.linalg.norm(emb))
            current_len = sum(len(s) for s in current_chunk)
            
            if sim >= self.similarity_threshold and (current_len + len(sentence)) <= self.max_chunk_size:
                current_chunk.append(sentence)
                # Update current embedding as mean of the chunk so far
                current_embedding = (current_embedding * len(current_chunk) + emb) / (len(current_chunk) + 1)
            else:
                chunks.append({"text": " ".join(current_chunk), "metadata": metadata.copy()})
                current_chunk = [sentence]
                current_embedding = emb
                
        if current_chunk:
            chunks.append({"text": " ".join(current_chunk), "metadata": metadata.copy()})
            
        for i, c in enumerate(chunks):
            c["metadata"]["chunk_id"] = f"{metadata.get('id', 'unknown')}_sem_{i}"
            c["metadata"]["total_chunks"] = len(chunks)
            c["metadata"]["strategy"] = "semantic"
            
        return chunks

class MetadataAwareChunker(ChunkingStrategy):
    """Wraps another chunker and prepends metadata to the text for better retrieval."""
    def __init__(self, base_chunker: ChunkingStrategy):
        self.base_chunker = base_chunker
        
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = self.base_chunker.chunk(text, metadata)
        for c in chunks:
            prefix = ""
            if "original_query" in c["metadata"] and c["metadata"]["original_query"]:
                prefix += f"Topic: {c['metadata']['original_query']}. "
            if "lang" in c["metadata"]:
                prefix += f"Language: {c['metadata']['lang']}. "
            c["text"] = prefix + c["text"]
            c["metadata"]["strategy"] += "_metadata_aware"
        return chunks
