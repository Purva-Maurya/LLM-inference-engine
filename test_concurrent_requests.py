import asyncio
import httpx

PROMPTS = [
    "Hello everyone",
    "The future of artificial intelligence is",
    "I really enjoy programming because",
    "The history of computers began",
]

async def send_request(client, prompt):
    resp = await client.post("http://127.0.0.1:8000/generate",
                              json={"prompt": prompt, "max_new_tokens": 20})
    return resp.json()

async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        results = await asyncio.gather(*[send_request(client, p) for p in PROMPTS])
        for r in results:
            print(f"\nPrompt: {r['prompt']!r}\nOutput: {r['generated_text']!r}")

if __name__ == "__main__":
    asyncio.run(main())