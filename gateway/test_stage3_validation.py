from copy import deepcopy

import httpx


GATEWAY_URL = "http://127.0.0.1:6008/chat"


BASE_REQUEST = {
    "model": "python-assistant",
    "messages": [
        {
            "role": "user",
            "content": "Python列表和元组有什么区别？",
        }
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": False,
}


def build_test_cases() -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []

    payload = deepcopy(BASE_REQUEST)
    payload["messages"] = []
    cases.append(("messages为空", payload))

    payload = deepcopy(BASE_REQUEST)
    payload["messages"][0]["content"] = ""
    cases.append(("content为空", payload))

    payload = deepcopy(BASE_REQUEST)
    payload["messages"][0]["content"] = "   "
    cases.append(("content只有空格", payload))

    payload = deepcopy(BASE_REQUEST)
    payload["temperature"] = 2.5
    cases.append(("temperature超出范围", payload))

    payload = deepcopy(BASE_REQUEST)
    payload["max_tokens"] = -1
    cases.append(("max_tokens小于0", payload))

    payload = deepcopy(BASE_REQUEST)
    payload["stream"] = "false"
    cases.append(("stream为字符串", payload))

    payload = deepcopy(BASE_REQUEST)
    payload["model"] = ""
    cases.append(("model为空", payload))

    return cases


def main() -> None:
    passed = 0
    cases = build_test_cases()

    with httpx.Client(timeout=10.0) as client:
        for name, payload in cases:
            try:
                response = client.post(
                    GATEWAY_URL,
                    json=payload,
                )
            except httpx.RequestError as exc:
                print(f"[连接失败] {name}：{exc}")
                continue

            success = response.status_code == 422

            if success:
                passed += 1
                result = "通过"
            else:
                result = "失败"

            print("=" * 70)
            print(f"测试项目：{name}")
            print(f"状态码：{response.status_code}")
            print(f"测试结果：{result}")

            try:
                print("响应内容：", response.json())
            except ValueError:
                print("响应内容：", response.text)

    print("=" * 70)
    print(f"测试完成：{passed}/{len(cases)} 项通过")

    if passed == len(cases):
        print("阶段三参数校验全部通过")
    else:
        print("仍有校验规则需要检查")


if __name__ == "__main__":
    main()
