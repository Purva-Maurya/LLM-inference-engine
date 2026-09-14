import itertools
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from paged_generate import generate_paged
from paged_cache import BlockPool

app = FastAPI(title="TERA", description="A hand-built GPT-2 inference engine with paged KV caching.")

# One shared block pool for the whole server's lifetime — this is what makes
# paging meaningful: multiple requests draw from the same physical memory pool.
SHARED_POOL = BlockPool(num_blocks=256, block_size=4)

# Hands out a unique ID to every incoming request, so concurrent requests
# never collide over the same block-table bookkeeping in the shared pool.
_request_id_counter = itertools.count()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 30
    temperature: float = 0.8
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/api")
def api_info():
    return {
        "message": "TERA inference engine is running.",
        "docs": "/docs",
        "endpoint": "POST /generate"
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    output = generate_paged(
        request.prompt,
        max_new_tokens=request.max_new_tokens,
        pool=SHARED_POOL,
        temperature=request.temperature,
        top_p=request.top_p,
        request_id=next(_request_id_counter)
    )
    return GenerateResponse(prompt=request.prompt, generated_text=output)


app.mount("/static", StaticFiles(directory="static"), name="static")