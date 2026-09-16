import asyncio
import itertools
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from paged_cache import BlockPool
from paged_scheduler import PagedScheduler, ScheduledRequest

app = FastAPI(title="TERA", description="A hand-built GPT-2 inference engine with continuous batching over a paged KV cache.")

# One shared block pool for the whole server's lifetime — every request draws
# from the same physical memory pool via its own block table.
SHARED_POOL = BlockPool(num_blocks=256, block_size=4)

# The background scheduler: batches whatever requests are currently active
# into one shared forward pass per step, admitting/evicting continuously.
scheduler = PagedScheduler(SHARED_POOL, max_batch_size=4)

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


@app.on_event("startup")
async def startup_event():
    # Launches the scheduler's infinite loop as a background task, running
    # for the entire lifetime of the server alongside FastAPI's own event loop.
    asyncio.create_task(scheduler.run_forever())


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
async def generate(request: GenerateRequest):
    req = ScheduledRequest(
        request_id=next(_request_id_counter),
        prompt=request.prompt,
        pool=SHARED_POOL,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )
    scheduler.submit(req)
    output = await req.future  # waits here until the scheduler finishes THIS request
    return GenerateResponse(prompt=request.prompt, generated_text=output)


app.mount("/static", StaticFiles(directory="static"), name="static")
