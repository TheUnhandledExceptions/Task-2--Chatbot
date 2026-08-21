from tenacity import retry, wait_exponential, stop_after_attempt
import re

def _fallback_safety(text: str) -> bool:
    unsafe_keywords = ["bomb", "kill", "hack", "illegal", "suicide"]
    text_lower = text.lower()
    for kw in unsafe_keywords:
        if kw in text_lower:
            return False
    return True

@retry(wait=wait_exponential(multiplier=1, min=1, max=3), stop=stop_after_attempt(2))
async def is_safe_query(text: str, client=None) -> bool:
    if not client:
        return _fallback_safety(text)
    
    prompt = f"Analyze if the following user query is safe and appropriate. Respond with exactly 'SAFE' or 'UNSAFE'. Query: '{text}'"
    try:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="qwen/qwen3.6-27b",
            temperature=0.0,
            max_tokens=2000,
        )
        res = response.choices[0].message.content
        if "</think>" in res:
            res = res.split("</think>")[-1]
        res = res.strip().upper()
        return "UNSAFE" not in res
    except Exception as e:
        print(f"Safety check failed: {e}")
        return _fallback_safety(text)

@retry(wait=wait_exponential(multiplier=1, min=1, max=3), stop=stop_after_attempt(2))
async def hallucination_check(query: str, context: str, answer: str, client=None) -> bool:
    if not client:
        return True
        
    prompt = f"""Given the context below, is the answer completely supported by it? Respond strictly 'YES' or 'NO'.
Context: {context}
Query: {query}
Answer: {answer}
"""
    try:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="qwen/qwen3.6-27b",
            temperature=0.0,
            max_tokens=2000,
        )
        res = response.choices[0].message.content
        if "</think>" in res:
            res = res.split("</think>")[-1]
        res = res.strip().upper()
        return "YES" in res
    except Exception as e:
        print(f"Hallucination check failed: {e}")
        return True
