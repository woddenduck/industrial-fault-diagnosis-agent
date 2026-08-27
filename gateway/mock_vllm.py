import asyncio

from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = FastAPI()


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "python-assistant",
                "object": "model",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat(payload: dict):
    model = payload.get("model")

    if model == "not-exist-model":
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": "model not found",
                    "private_path": "/root/private/model",
                }
            },
        )

    if model == "slow-model":
        await asyncio.sleep(10)

    if model == "broken-model":
        return JSONResponse(
            status_code=500,
            content={
                "traceback": "sensitive traceback",
                "path": "/root/private/model",
            },
        )

    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "模拟模型回答",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
