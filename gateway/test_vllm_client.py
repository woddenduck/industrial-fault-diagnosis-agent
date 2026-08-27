import asyncio

from app.clients.vllm_client import vllm_client


async def main() -> None:
    payload = {
        "model": "python-assistant",
        "messages": [
            {
                "role": "system",
                "content": "你是一位严谨的 Python 技术助手。",
            },
            {
                "role": "user",
                "content": "请简要解释 Python 列表和元组的区别。",
            },
        ],
        "temperature": 0,
        "max_tokens": 256,
    }

    try:
        print("=" * 60)
        print("测试 get_models()")
        print("=" * 60)

        models_result = await vllm_client.get_models()
        print(models_result)

        print()
        print("=" * 60)
        print("测试 chat()")
        print("=" * 60)

        chat_result = await vllm_client.chat(payload)
        content = chat_result["choices"][0]["message"]["content"]
        print(content)

        print()
        print("=" * 60)
        print("测试 chat_stream()")
        print("=" * 60)

        async for text in vllm_client.chat_stream(payload):
            print(text, end="", flush=True)

        print()

    finally:
        await vllm_client.close()


if __name__ == "__main__":
    asyncio.run(main())
