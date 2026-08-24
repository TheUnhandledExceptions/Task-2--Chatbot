import os
import time
import asyncio
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from guardrails import is_safe_query
from typing import Dict, Any
import re

load_dotenv()

class RAGOrchestrator:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.sarvam_api_key = os.getenv("SARVAM_API_KEY")
        
        if self.groq_api_key:
            self.groq_client = AsyncGroq(api_key=self.groq_api_key)
        else:
            self.groq_client = None
            print("WARNING: GROQ_API_KEY not set!")
            
        print("Loading local embedding model...")
        self.embed_model = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            providers=["CPUExecutionProvider"]
        )
        self.qdrant_client = QdrantClient(path="./qdrant_db")
        self.collection_name = "msmarco_xi_indic"
        self.http_client = httpx.AsyncClient(timeout=10.0)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=3), stop=stop_after_attempt(3))
    async def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        if not self.sarvam_api_key:
            return {"text": "mock transcribed query", "latency": 0.05}
            
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {"api-subscription-key": self.sarvam_api_key}
        data = {"model": "saaras:v3", "mode": "transcribe"}
        
        try:
            with open(audio_path, "rb") as f:
                file_content = f.read()
            
            ext = os.path.splitext(audio_path)[1].lower()
            mime_type = "audio/webm" if ext == ".webm" else "audio/wav"
            files = {"file": (os.path.basename(audio_path), file_content, mime_type)}
            response = await self.http_client.post(url, headers=headers, files=files, data=data)
            
            if response.status_code == 200:
                json_resp = response.json()
                transcript = json_resp.get("transcript", "")
                if not transcript:
                    transcript = f"Error: No speech detected. Make sure your microphone is working and you speak clearly before releasing."
            else:
                transcript = f"Error: {response.status_code}"
                print("Sarvam API Error:", response.status_code, response.text)
        except Exception as e:
            print("Sarvam API Exception:", str(e))
            transcript = f"Exception: {str(e)}"
            
        end_time = time.time()
        return {"text": transcript, "latency": end_time - start_time}

    async def retrieve_context(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Embed query (CPU bound, run in thread)
        query_vector = list(self.embed_model.embed([query]))[0].tolist()
        
        # 2. Search Qdrant (IO bound, run synchronously)
        try:
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=1
            )
            contexts = [hit.payload["text"] for hit in search_result.points]
            retrieved_text = "\n\n".join(contexts)
        except Exception as e:
            print(f"Retrieval error: {e}")
            retrieved_text = ""
            
        end_time = time.time()
        return {"context": retrieved_text, "latency": end_time - start_time}

    @retry(wait=wait_exponential(multiplier=1, min=1, max=3), stop=stop_after_attempt(2))
    async def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        start_time = time.time()
        
        prompt = f"""You are a strict QA assistant. You MUST answer the user's query ONLY using the provided context.
If the context does not contain the information needed to answer the query, you MUST output exactly: "I cannot answer this based on the provided context." Do NOT use your general knowledge.

Context:
{context}

Query: {query}
Answer:"""
        if not self.groq_client:
            await asyncio.sleep(0.1)
            return {"answer": "Mock generated answer based on context.", "latency": 0.1}

        try:
            chat_completion = await self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="allam-2-7b",
                temperature=0.0,
                max_tokens=30,
            )
            answer = chat_completion.choices[0].message.content
            
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            elif "<think>" in answer:
                # Model got cut off before finishing its thoughts
                answer = "Error: The model's thought process exceeded the token limit."
            else:
                answer = answer.strip()
        except Exception as e:
            answer = f"Error generating answer: {e}"
            
        end_time = time.time()
        return {"answer": answer, "latency": end_time - start_time}

    async def process(self, audio_path: str) -> Dict[str, Any]:
        # 1. STT
        stt_result = await self.transcribe_audio(audio_path)
        query = stt_result["text"]
        
        if not query or query.startswith("Error") or query.startswith("Exception"):
            error_msg = query.replace("Error: ", "") if query.startswith("Error: ") else "Failed to transcribe audio"
            return {"error": error_msg, "query": "No transcription", "timings": {"stt": stt_result["latency"]}}
            
        # 2. Concurrent Guardrail (Pre-retrieval) and Retrieval
        safety_task = asyncio.create_task(is_safe_query(query, self.groq_client))
        retrieval_task = asyncio.create_task(self.retrieve_context(query))
        
        is_safe, retrieval_result = await asyncio.gather(safety_task, retrieval_task)
        
        if not is_safe:
            return {
                "error": "Query flagged as unsafe or off-topic.",
                "query": query,
                "timings": {"stt": stt_result["latency"]}
            }
            
        # 3. Generation
        generation_result = await self.generate_answer(query, retrieval_result["context"])
        final_answer = generation_result["answer"]
        
        # 4. Return final grounded answer (System prompt handles guardrails)
        total_pipeline_time = stt_result["latency"] + retrieval_result["latency"] + generation_result["latency"]
        
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
