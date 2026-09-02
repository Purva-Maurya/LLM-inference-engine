from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model_cached import generate_cached

app = FastAPI(title="LLM Inference Engine", description="A hand-built GPT-2 inference engine with KV caching.")


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
        "message": "LLM Inference Engine is running.",
        "docs": "/docs",
        "endpoint": "POST /generate"
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    output = generate_cached(
        request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )
    return GenerateResponse(prompt=request.prompt, generated_text=output)


app.mount("/static", StaticFiles(directory="static"), name="static")