import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv('backend/.env')
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query: str, context: str) -> str:
    """
    Generates an answer based on the provided context.
    This function is required by the rag-local-eval-loop for the CORRECTNESS, 
    FAITHFULNESS, and RELIABILITY checks.
    """
    prompt = f"""You are a helpful AI assistant. Answer the query using ONLY the provided context.
If the context does not contain the answer, output exactly: "I cannot answer based on the provided context."
Keep the answer concise and direct.

Context:
{context}

Query: {query}

Answer:"""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="qwen/qwen3.6-27b",
            temperature=0.1,
            max_tokens=1000,
        )
        answer = response.choices[0].message.content
        
        # Strip out any <think> reasoning blocks outputted by the model
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        elif "<think>" in answer:
            answer = "Error: The model's thought process exceeded the token limit."
        else:
            answer = answer.strip()
            
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"
