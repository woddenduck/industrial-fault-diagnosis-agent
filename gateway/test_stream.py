import asyncio
import json
import time

import httpx


URL = "http://127.0.0.1:6008/chat"

PAYLOAD = {
    "model": "python-assistant",
    "messages": [
        {
            "role": "system",
            "content": "你是一位严谨的 Python 技术助手。",
        },
        {
            "role": "user",
            "content": (
                "请详细解释 Python 列表、元组和集合的区别，"
                "并给出代码示例。"
            ),
        },
    ],
    "temperature": 0,
    "max_tokens": 512,
    "stream": True,
}


async def main() -> None:
    start_time = time.perf_counter()
    first_content_time: float | None = None

    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream(
            method="POST",
            url=URL,
            json=PAYLOAD,
        ) as response:
            response.raise_for_status()

            print(
                f"HTTP 状态码：{response.status_code}"
            )

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue

                data = line.removeprefix("data:").strip()

                if not data:
                    continue

                if data == "[DONE]":
                    break

                chunk = json.loads(data)

                if "error" in chunk:
                    raise RuntimeError(
                        f"流式响应发生错误：{chunk['error']}"
                    )

                choices = chunk.get("choices") or []

                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""

                if not content:
                    continue

                if first_content_time is None:
                    first_content_time = (
                        time.perf_counter() - start_time
                    )

                print(
                    content,
                    end="",
                    flush=True,
                )

    total_time = time.perf_counter() - start_time

    print("\n")
    print(
        "首个内容延迟 TTFT：",
        (
            f"{first_content_time:.3f}s"
            if first_content_time is not None
            else "未收到内容"
        ),
    )
    print(f"总耗时：{total_time:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
