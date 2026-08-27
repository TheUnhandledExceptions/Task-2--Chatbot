# Task 2: Voice RAG Backend - Complete Code Explanation

## Project Overview
**Voice RAG Backend** is a FastAPI-based application that combines **Speech-to-Text (STT)**, **Retrieval-Augmented Generation (RAG)**, and **Safety Guardrails** to answer user queries from audio input using a knowledge base.

The system transcribes audio, retrieves relevant context from a vector database, and generates grounded answers using an LLM.

---

## Architecture Overview

```
User Audio Input
    ↓
[Speech-to-Text - Sarvam API]
    ↓
[Query Safety Guardrail Check]  +  [Vector Retrieval - Qdrant]
    ↓                                     ↓
[Context Retrieved]
    ↓
[Answer Generation - Groq LLM]
    ↓
[Response + Latency Metrics]
```

---

## Key Components

### 1. **app.py** - Gradio UI & ASGI Wrapper
**Purpose:** Serves the FastAPI backend with an optional Gradio interface for Hugging Face Spaces.

```python
import gradio as gr
from main import app as fastapi_app

def dummy():
    return "Voice RAG FastAPI Backend is live!"

demo = gr.Interface(fn=dummy, inputs=None, outputs="text")
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
```

**What it does:**
- Creates a dummy Gradio interface for HF Spaces compatibility
- Mounts the FastAPI app at `/ui`
- The Gradio app itself becomes the ASGI instance

---

### 2. **backend/main.py** - FastAPI Server
**Purpose:** Defines the core REST API endpoints.

#### Endpoint: `POST /query`
```python
@app.post("/query")
async def query_audio(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    # Save uploaded audio file temporarily
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process through RAG pipeline
        result = await orchestrator.process(temp_path)
        
        # Log latency if successful
        if "error" not in result or result.get("error") == "Query flagged as unsafe or off-topic.":
            log_latency(result["timings"])
        
        return result
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
```

**What it does:**
- Accepts audio file uploads (`.wav`, `.webm`)
- Temporarily stores the file
- Processes it through the RAGOrchestrator
- Logs performance metrics
- Cleans up temporary files
- Returns results with latency information

#### Endpoint: `GET /analytics`
```python
@app.get("/analytics")
async def analytics():
    return get_analytics()
```
**What it does:** Returns aggregated latency statistics (p50, p70, p100).

---

### 3. **backend/orchestrator.py** - RAG Pipeline Orchestrator
**Purpose:** Coordinates the entire RAG workflow asynchronously.

#### Class: `RAGOrchestrator`

**Initialization:**
```python
def __init__(self):
    self.groq_client = AsyncGroq(api_key=self.groq_api_key)  # LLM for generation
    self.embed_model = TextEmbedding(...)  # Embedding model for retrieval
    self.qdrant_client = QdrantClient(path="./qdrant_db")  # Vector database
    self.http_client = httpx.AsyncClient()  # HTTP client for Sarvam API
```

#### Method 1: `transcribe_audio(audio_path)`
**Purpose:** Convert audio to text using Sarvam AI Speech-to-Text API.

```python
async def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
    # Retry logic: up to 3 attempts with exponential backoff
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": self.sarvam_api_key}
    
    with open(audio_path, "rb") as f:
        file_content = f.read()
    
    # Send audio file for transcription
    response = await self.http_client.post(url, headers=headers, files=files, data=data)
    
    if response.status_code == 200:
        transcript = response.json().get("transcript", "")
    else:
        transcript = f"Error: {response.status_code}"
    
    return {"text": transcript, "latency": end_time - start_time}
```

**What it does:**
- Converts audio file to text
- Handles retries if API fails (up to 3 times)
- Falls back to mock transcription if API key missing
- Returns transcription + latency

#### Method 2: `retrieve_context(query)`
**Purpose:** Find relevant context from the Qdrant vector database.

```python
async def retrieve_context(self, query: str) -> Dict[str, Any]:
    # 1. Embed the query using the local embedding model
    query_vector = list(self.embed_model.embed([query]))[0].tolist()
    
    # 2. Search Qdrant for similar vectors
    search_result = self.qdrant_client.query_points(
        collection_name="msmarco_xi_indic",
        query=query_vector,
        limit=1  # Retrieve top-1 result
    )
    
    # 3. Extract text from retrieved vectors
    contexts = [hit.payload["text"] for hit in search_result.points]
    retrieved_text = "\n\n".join(contexts)
    
    return {"context": retrieved_text, "latency": end_time - start_time}
```

**What it does:**
- Embeds the user query to a vector
- Searches Qdrant for semantically similar passages
- Returns the top-1 most relevant context
- Handles retrieval errors gracefully

#### Method 3: `generate_answer(query, context)`
**Purpose:** Generate an answer using Groq's LLM, constrained to the provided context.

```python
async def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
    prompt = f"""You are a strict QA assistant. You MUST answer the user's query ONLY using the provided context.
If the context does not contain the information needed to answer the query, you MUST output exactly: "I cannot answer this based on the provided context." Do NOT use your general knowledge.

Context:
{context}

Query: {query}
Answer:"""
    
    # Call Groq API with allam-2-7b model
    chat_completion = await self.groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="allam-2-7b",
        temperature=0.0,  # Deterministic output
        max_tokens=30,
    )
    
    answer = chat_completion.choices[0].message.content.strip()
    return {"answer": answer, "latency": end_time - start_time}
```

**What it does:**
- Creates a strict prompt that forces the model to answer ONLY from context
- Calls Groq's API asynchronously
- Extracts thinking blocks if present (handles model's reasoning)
- Returns the final answer + latency

#### Method 4: `process(audio_path)` - Main Pipeline
**Purpose:** Orchestrates the entire RAG workflow.

```python
async def process(self, audio_path: str) -> Dict[str, Any]:
    # 1. Speech-to-Text
    stt_result = await self.transcribe_audio(audio_path)
    query = stt_result["text"]
    
    # Check if transcription failed
    if not query or query.startswith("Error"):
        return {"error": error_msg, "query": "No transcription", "timings": {"stt": stt_result["latency"]}}
    
    # 2. Concurrent: Safety Check + Retrieval (run in parallel!)
    safety_task = asyncio.create_task(is_safe_query(query, self.groq_client))
    retrieval_task = asyncio.create_task(self.retrieve_context(query))
    
    is_safe, retrieval_result = await asyncio.gather(safety_task, retrieval_task)
    
    # Check if query is safe
    if not is_safe:
        return {"error": "Query flagged as unsafe or off-topic.", "query": query, ...}
    
    # 3. Generation
    generation_result = await self.generate_answer(query, retrieval_result["context"])
    
    # 4. Return final grounded answer
    return {
        "query": query,
        "answer": final_answer,
        "context": retrieval_result["context"],
        "timings": {
            "stt": stt_result["latency"],
            "retrieval": retrieval_result["latency"],
            "generation": generation_result["latency"],
            "total": total_pipeline_time
        }
    }
```

**What it does:**
- **Orchestrates the full RAG pipeline in sequence**
- **STT → (Safety Check + Retrieval in parallel) → Generation**
- **Gathers timing metrics for every step**
- **Returns a comprehensive JSON response with query, answer, context, and latencies**

---

### 4. **backend/guardrails.py** - Safety & Quality Checks
**Purpose:** Prevent unsafe/off-topic queries and hallucinations.

#### Function: `is_safe_query(text, client)`
```python
async def is_safe_query(text: str, client=None) -> bool:
    # Fallback: keyword-based safety check
    unsafe_keywords = ["bomb", "kill", "hack", "illegal", "suicide"]
    
    # Primary: Use Groq to analyze safety
    prompt = f"Analyze if the following user query is safe and appropriate. Respond with exactly 'SAFE' or 'UNSAFE'. Query: '{text}'"
    
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="qwen/qwen3.6-27b",
        temperature=0.0,
        max_tokens=2000,
    )
    
    res = response.choices[0].message.content
    return "UNSAFE" not in res
```

**What it does:**
- Uses Groq's Qwen model to check if query is safe
- Falls back to keyword-based checking if API fails
- Runs concurrently with retrieval to save latency

#### Function: `hallucination_check(query, context, answer, client)`
```python
async def hallucination_check(query: str, context: str, answer: str, client=None) -> bool:
    prompt = f"""Given the context below, is the answer completely supported by it? Respond strictly 'YES' or 'NO'.
Context: {context}
Query: {query}
Answer: {answer}"""
    
    # Returns True if answer is grounded in context, False if hallucinated
```

**What it does:**
- Verifies that the generated answer is actually supported by the retrieved context
- Prevents LLM from making up information not in the context
- Can be called post-generation to validate answers

---

### 5. **backend/analytics.py** - Latency Tracking
**Purpose:** Store and retrieve performance metrics.

```python
# SQLite database structure
CREATE TABLE IF NOT EXISTS latencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stt_latency REAL,
    retrieval_latency REAL,
    generation_latency REAL,
    total_latency REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)

def log_latency(timings: Dict[str, float]):
    # Inserts timing data into database

def get_analytics() -> Dict[str, Any]:
    # Returns percentile statistics (p50, p70, p100) of total latency
```

**What it does:**
- Logs latency of each pipeline stage
- Calculates percentiles for performance monitoring
- Enables performance SLA tracking

---

### 6. **backend/chunking.py** - Text Chunking Strategies
**Purpose:** Splits documents into manageable chunks for embedding.

#### Class: `AdvancedChunker`
```python
class AdvancedChunker(ChunkingStrategy):
    def __init__(self, max_chunk_size: int = 500, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Splits on sentence boundaries (., !, ?, |, ।)
        # Maintains overlap between chunks for context preservation
        # Adds metadata about chunk position and strategy
```

**What it does:**
- Splits text into 500-char chunks with 100-char overlap
- Preserves sentence boundaries (doesn't cut mid-sentence)
- Maintains metadata for traceability

#### Class: `SemanticChunker`
```python
class SemanticChunker(ChunkingStrategy):
    def __init__(self, model: SentenceTransformer, max_chunk_size: int = 500, similarity_threshold: float = 0.5):
        # Groups semantically similar sentences together
        # Uses similarity threshold to decide chunk boundaries
```

**What it does:**
- Groups sentences by semantic similarity
- Creates chunks of semantically cohesive content
- More intelligent than simple word-count chunking

---

### 7. **backend/indexer.py** - Vector Database Indexing
**Purpose:** Indexes the MSMARCO-XI dataset into Qdrant.

```python
# Load the ai4bharat/MSMARCO-XI dataset
dataset = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)

# Create Qdrant collection
client.create_collection(
    collection_name="msmarco_xi_indic",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Index documents with semantic chunking
for row in dataset:
    text = row.get("passage", "")
    chunks = chunker.chunk(text, metadata={"id": row["id"]})
    
    # Embed each chunk
    embeddings = model.embed([c["text"] for c in chunks])
    
    # Store in Qdrant
    for chunk, embedding in zip(chunks, embeddings):
        qdrant_client.upsert(
            collection_name="msmarco_xi_indic",
            points=[PointStruct(id=point_id, vector=embedding, payload={...})]
        )
```

**What it does:**
- Loads the multilingual MSMARCO-XI dataset
- Chunks documents intelligently
- Embeds chunks using multilingual embeddings
- Stores everything in Qdrant for fast vector search

---

### 8. **app/retriever.py** - Benchmark Retrieval
**Purpose:** Measures search latency performance.

```python
class SearchResponse:
    def __init__(self, total_ms, embed_ms, search_ms):
        self.total_ms = total_ms      # Total time
        self.embed_ms = embed_ms      # Embedding time
        self.search_ms = search_ms    # Qdrant query time

def search(query: str, top_k: int = 5) -> SearchResponse:
    # Measure embedding latency
    start = time.time()
    query_vector = model.embed([query])[0]
    embed_ms = (time.time() - start) * 1000
    
    # Measure search latency
    start = time.time()
    results = qdrant_client.query_points(
        collection_name="msmarco_xi_indic",
        query=query_vector,
        limit=top_k
    )
    search_ms = (time.time() - start) * 1000
    
    return SearchResponse(
        total_ms=embed_ms + search_ms,
        embed_ms=embed_ms,
        search_ms=search_ms
    )
```

**What it does:**
- Measures embedding and search latency separately
- Used for performance benchmarking
- Returns detailed timing breakdown

---

### 9. **benchmark.py** - Performance Testing
**Purpose:** Run performance benchmarks against the RAG system.

```python
def warmup():
    # Pre-loads embedding model and Qdrant database
    # Warms up inference for accurate benchmarking

def search(query: str, top_k: int = 5) -> SearchResponse:
    # Performs a single search and measures latency
```

---

## Data Flow Example

```
User speaks: "Tell me about climate change"
    ↓
[STT] → Transcribed text: "Tell me about climate change"
    ↓
[Safety Check] (async)         [Retrieval] (async)
    "SAFE" ← Yes                   → Embedded to vector
                                   → Search Qdrant
                                   → Found: "Climate change is caused by..."
    ↓
[Generation]
    Prompt: "Using only the context: 'Climate change is...'
             Answer: Tell me about climate change"
    Model: "Climate change is caused by greenhouse gas emissions..."
    ↓
Response:
{
    "query": "Tell me about climate change",
    "answer": "Climate change is caused by greenhouse gas emissions...",
    "context": "Climate change is...",
    "timings": {
        "stt": 0.45,
        "retrieval": 0.12,
        "generation": 0.82,
        "total": 1.39
    }
}
```

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | FastAPI | REST API server |
| **UI** | Gradio | Web interface (HF Spaces) |
| **STT** | Sarvam AI API | Audio transcription |
| **Embeddings** | Sentence Transformers | Vector embeddings |
| **Vector DB** | Qdrant | Fast similarity search |
| **LLM** | Groq (Allam-2-7B, Qwen) | Text generation & safety checks |
| **Async** | asyncio | Parallel task execution |
| **Database** | SQLite | Latency tracking |
| **Dataset** | MSMARCO-XI | Multilingual knowledge base |

---

## Environment Variables

```
GROQ_API_KEY=your_groq_api_key          # LLM access
SARVAM_API_KEY=your_sarvam_api_key      # Speech-to-Text
```

---

## API Response Structure

### Success Response
```json
{
    "query": "user's transcribed question",
    "answer": "grounded answer from context",
    "context": "retrieved passage",
    "timings": {
        "stt": 0.45,
        "retrieval": 0.12,
        "generation": 0.82,
        "total": 1.39
    }
}
```

### Error Response
```json
{
    "error": "Query flagged as unsafe or off-topic.",
    "query": "original transcribed query",
    "timings": {
        "stt": 0.45
    }
}
```

---

## Latency Optimization Techniques

1. **Concurrent Execution:** Safety check and retrieval run in parallel
2. **Local Embeddings:** Use CPU-based embedding model (no API calls)
3. **Vector Search:** Qdrant provides fast cosine similarity search
4. **Retry Logic:** Exponential backoff for API failures
5. **Caching:** Connection pooling via httpx.AsyncClient

---

## Summary

This is a **production-grade Voice RAG system** that:
- ✅ Transcribes audio to text
- ✅ Retrieves relevant context from a 1000+ document knowledge base
- ✅ Generates grounded answers using LLMs
- ✅ Enforces safety guardrails (no unsafe queries)
- ✅ Prevents hallucinations (answers only from context)
- ✅ Tracks performance metrics at each stage
- ✅ Runs all operations asynchronously for speed
- ✅ Supports multilingual queries (Indic languages)
