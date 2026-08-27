"""
Day18 Stage4
Tokenizer Manager

使用真实LLM tokenizer统计Token数量

当前LLM:
Qwen/Qwen3-8B
"""


import os

from transformers import AutoTokenizer


MODEL_PATH = os.getenv(
    "TOKENIZER_MODEL_PATH",
    os.getenv("CHAT_MODEL", "Qwen/Qwen3-8B"),
)


print(
    f"Loading tokenizer: {MODEL_PATH}"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)



def count_tokens(text):
    """
    统计真实token数量

    参数:
        text:str

    返回:
        token数量
    """

    if not text:
        return 0


    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )


    return len(tokens)



if __name__ == "__main__":


    test="F30021表示过电流故障"


    print(
        "Text:",
        test
    )


    print(
        "Tokens:",
        count_tokens(test)
    )
