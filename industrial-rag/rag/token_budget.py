"""
Day18 Stage4 / Stage8
Token Budget 配置

重点：预算必须与“实际部署的 vLLM max_model_len”一致，
而不是只看模型理论支持的最大上下文。

可通过环境变量覆盖：
    export MODEL_MAX_CONTEXT=4096
    export OUTPUT_RESERVED=1000
"""

import os

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
MODEL_MAX_CONTEXT = int(os.getenv("MODEL_MAX_CONTEXT", "4096"))

SYSTEM_BUDGET = int(os.getenv("SYSTEM_BUDGET", "500"))
HISTORY_BUDGET = int(os.getenv("HISTORY_BUDGET", "800"))
QUESTION_BUDGET = int(os.getenv("QUESTION_BUDGET", "200"))
OUTPUT_RESERVED = int(os.getenv("OUTPUT_RESERVED", "1000"))

AVAILABLE_INPUT = MODEL_MAX_CONTEXT - OUTPUT_RESERVED

CONTEXT_BUDGET = (
    AVAILABLE_INPUT
    - SYSTEM_BUDGET
    - HISTORY_BUDGET
    - QUESTION_BUDGET
)

if CONTEXT_BUDGET <= 0:
    raise ValueError(
        "Token Budget 配置无效：CONTEXT_BUDGET <= 0。"
        f" max_context={MODEL_MAX_CONTEXT},"
        f" output_reserved={OUTPUT_RESERVED},"
        f" system={SYSTEM_BUDGET},"
        f" history={HISTORY_BUDGET},"
        f" question={QUESTION_BUDGET}"
    )


def get_budget():
    return {
        "model": MODEL_NAME,
        "max_context": MODEL_MAX_CONTEXT,
        "system": SYSTEM_BUDGET,
        "history": HISTORY_BUDGET,
        "question": QUESTION_BUDGET,
        "context": CONTEXT_BUDGET,
        "output_reserved": OUTPUT_RESERVED,
    }
