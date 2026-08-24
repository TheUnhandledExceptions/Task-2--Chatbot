import asyncio
import os
import time
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()
client = AsyncGroq(api_key=os.environ['GROQ_API_KEY'])

async def test_model(model_id):
    try:
        start = time.time()
        res = await client.chat.completions.create(
            model=model_id, 
            messages=[{'role': 'user', 'content': 'What is the capital of France?'}]
        )
        latency = time.time() - start
        print(f"SUCCESS {model_id} - Latency: {latency:.2f}s - Response: {res.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"FAILED {model_id} - Error: {e}")

async def main():
    models = ['allam-2-7b', 'canopylabs/orpheus-v1-english', 'qwen/qwen3.6-27b']
    for m in models:
        await test_model(m)

asyncio.run(main())
