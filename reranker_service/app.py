"""
Qwen3-Reranker-0.6B 独立 Reranker 服务。

目标：
1. 保持现有 Gateway API 契约不变：
   POST /rerank
   {
       "query": "...",
       "documents": ["...", "..."],
       "top_k": 3
   }

2. 使用 Qwen3-Reranker 官方推荐的生成式打分方式：
   AutoTokenizer + AutoModelForCausalLM
   -> 构造 Instruct / Query / Document
   -> 读取最后位置的 "yes" / "no" logits
   -> softmax 得到 0~1 相关性分数

3. 不再使用传统 CrossEncoder / SequenceClassification，
   避免随机初始化 score.weight 的问题。

4. 默认针对 CPU 稳定运行：
   - 分批推理
   - 推理互斥锁
   - use_cache=False
   - torch.inference_mode()
   - 显式 Token 预算控制

要求：
- Python >= 3.10
- torch
- transformers >= 4.51.0
- fastapi
- pydantic >= 2
"""

from __future__ import annotations

import inspect
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import torch
import transformers
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)
from packaging.version import Version
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


# ============================================================
# 1. 日志
# ============================================================

LOG_LEVEL = os.getenv(
    "RERANKER_LOG_LEVEL",
    "INFO",
).upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("reranker_service")


# ============================================================
# 2. 环境变量读取
# ============================================================

def _get_non_empty_string(
    name: str,
    default: str,
) -> str:
    """读取非空字符串环境变量。"""

    value = os.getenv(name, default).strip()

    if not value:
        raise RuntimeError(
            f"环境变量 {name} 不能为空",
        )

    return value


def _get_positive_int(
    name: str,
    default: int,
) -> int:
    """读取正整数环境变量。"""

    raw_value = os.getenv(
        name,
        str(default),
    ).strip()

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"环境变量 {name} 必须是整数，"
            f"当前值为：{raw_value}",
        ) from exc

    if value <= 0:
        raise RuntimeError(
            f"环境变量 {name} 必须大于 0，"
            f"当前值为：{value}",
        )

    return value


RERANKER_MODEL_PATH = _get_non_empty_string(
    "RERANKER_MODEL_PATH",
    "Qwen/Qwen3-Reranker-0.6B",
)

RERANKER_DEVICE = _get_non_empty_string(
    "RERANKER_DEVICE",
    "cpu",
)

RERANKER_MAX_DOCUMENTS = _get_positive_int(
    "RERANKER_MAX_DOCUMENTS",
    50,
)

RERANKER_MAX_QUERY_CHARS = _get_positive_int(
    "RERANKER_MAX_QUERY_CHARS",
    4096,
)

RERANKER_MAX_DOCUMENT_CHARS = _get_positive_int(
    "RERANKER_MAX_DOCUMENT_CHARS",
    8192,
)

# 对工业 RAG 的 Chunk 来说 1024 通常已经足够。
# Qwen3-Reranker 支持更长上下文，但 CPU 服务不建议一开始开得过大。
RERANKER_MAX_LENGTH = _get_positive_int(
    "RERANKER_MAX_LENGTH",
    1024,
)

# CPU 默认保守使用 4。
RERANKER_BATCH_SIZE = _get_positive_int(
    "RERANKER_BATCH_SIZE",
    4,
)

# Qwen 官方建议根据业务场景定制 Instruct。
# 多语言检索场景中，官方也建议 instruction 优先使用英文。
RERANKER_INSTRUCTION = _get_non_empty_string(
    "RERANKER_INSTRUCTION",
    (
        "Given an industrial equipment troubleshooting query, "
        "retrieve relevant passages from technical manuals that "
        "can help answer the query."
    ),
)


# ============================================================
# 3. Qwen3-Reranker Prompt
# ============================================================

SYSTEM_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. Note that the answer "
    "can only be \"yes\" or \"no\"."
    "<|im_end|>\n"
    "<|im_start|>user\n"
)

ASSISTANT_SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)


# ============================================================
# 4. 并发控制
# ============================================================

# 当前服务默认 CPU 推理。
# 防止多个请求同时执行大模型前向计算造成 CPU 线程争抢和内存峰值。
inference_lock = threading.Lock()


# ============================================================
# 5. Pydantic 请求 / 响应
# ============================================================

class RerankerSchema(BaseModel):
    """Reranker 请求和响应模型公共基类。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class RerankRequest(RerankerSchema):
    """POST /rerank 请求体。"""

    query: str = Field(
        min_length=1,
        max_length=RERANKER_MAX_QUERY_CHARS,
        description="用户问题，不能为空",
    )

    documents: list[str] = Field(
        min_length=1,
        max_length=RERANKER_MAX_DOCUMENTS,
        description="候选文档列表",
    )

    top_k: int = Field(
        gt=0,
        description="需要返回的结果数量",
    )

    @field_validator(
        "query",
        mode="before",
    )
    @classmethod
    def validate_query(
        cls,
        value: object,
    ) -> object:
        """拒绝空字符串和纯空格 query。"""

        if not isinstance(value, str):
            return value

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "query 不能为空或只包含空格",
            )

        return cleaned_value

    @field_validator(
        "documents",
        mode="before",
    )
    @classmethod
    def validate_documents(
        cls,
        value: object,
    ) -> object:
        """检查候选文档并去除首尾空格。"""

        if not isinstance(value, list):
            return value

        cleaned_documents: list[str] = []

        for index, document in enumerate(value):
            if not isinstance(document, str):
                cleaned_documents.append(document)
                continue

            cleaned_document = document.strip()

            if not cleaned_document:
                raise ValueError(
                    f"documents[{index}] 不能为空"
                    "或只包含空格",
                )

            if (
                len(cleaned_document)
                > RERANKER_MAX_DOCUMENT_CHARS
            ):
                raise ValueError(
                    f"documents[{index}] 长度超过限制："
                    f"{RERANKER_MAX_DOCUMENT_CHARS}",
                )

            cleaned_documents.append(
                cleaned_document,
            )

        return cleaned_documents

    @model_validator(mode="after")
    def validate_top_k(
        self,
    ) -> "RerankRequest":
        """保证 top_k 不超过候选文档数量。"""

        document_count = len(self.documents)

        if self.top_k > document_count:
            raise ValueError(
                "top_k 不能超过 documents 数量："
                f"top_k={self.top_k}, "
                f"documents={document_count}",
            )

        return self


class RerankResult(RerankerSchema):
    """单条重排结果。"""

    index: int = Field(
        ge=0,
        description="文档在原始 documents 中的索引",
    )

    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Qwen3-Reranker 基于 yes/no logits "
            "归一化得到的相关性分数"
        ),
    )

    document: str = Field(
        min_length=1,
        description="原始文档内容",
    )


class RerankResponse(RerankerSchema):
    """Reranker 统一响应。"""

    results: list[RerankResult]


class HealthResponse(RerankerSchema):
    """保持与原服务兼容的健康检查响应。"""

    status: Literal["ok"]
    model: str
    device: str
    max_documents: int
    batch_size: int


# ============================================================
# 6. 环境和模型辅助函数
# ============================================================

def _validate_runtime() -> None:
    """检查 Qwen3 所需的 Transformers 版本。"""

    current_version = Version(
        transformers.__version__,
    )

    minimum_version = Version("4.51.0")

    if current_version < minimum_version:
        raise RuntimeError(
            "Qwen3-Reranker-0.6B 要求 "
            "transformers>=4.51.0，"
            f"当前版本为 {current_version}",
        )


def _validate_device(
    device: str,
) -> None:
    """检查配置的推理设备是否可用。"""

    normalized = device.lower()

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RERANKER_DEVICE 配置为 CUDA，"
                "但当前 PyTorch 检测不到可用 GPU",
            )

    elif normalized == "cpu":
        return

    elif normalized.startswith("mps"):
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "RERANKER_DEVICE 配置为 MPS，"
                "但当前环境不可用",
            )

    else:
        # torch 本身支持更多设备名称，
        # 这里不直接拒绝，后续 model.to(device) 会给出具体错误。
        logger.warning(
            "使用非标准设备配置：%s",
            device,
        )


def _resolve_dtype(
    device: str,
) -> torch.dtype:
    """
    选择稳定的推理 dtype。

    当前 AutoDL 配置中 Reranker 默认在 CPU：
    CPU 使用 float32，优先保证兼容性和稳定性。

    如果以后改为 CUDA：
    - 支持 BF16 时优先 bfloat16；
    - 否则使用 float16。
    """

    normalized = device.lower()

    if normalized == "cpu":
        return torch.float32

    if normalized.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    return torch.float32


def _get_actual_device(
    model: PreTrainedModel,
) -> str:
    """获取模型实际所在设备。"""

    try:
        return str(
            next(model.parameters()).device,
        )
    except StopIteration:
        return RERANKER_DEVICE


def _get_single_token_id(
    tokenizer: PreTrainedTokenizerBase,
    token_text: str,
) -> int:
    """
    获取 yes/no 的单 Token ID。

    Qwen3-Reranker 官方打分依赖这两个 Token。
    如果 tokenizer 行为异常则直接阻止服务启动，
    避免悄悄产生错误分数。
    """

    token_ids = tokenizer(
        token_text,
        add_special_tokens=False,
    )["input_ids"]

    if (
        not isinstance(token_ids, list)
        or len(token_ids) != 1
    ):
        raise RuntimeError(
            f"Token {token_text!r} 不是单 Token："
            f"{token_ids}",
        )

    return int(token_ids[0])


def _format_instruction(
    query: str,
    document: str,
) -> str:
    """按照 Qwen3-Reranker 官方格式构造待判断文本。"""

    return (
        f"<Instruct>: {RERANKER_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


def _prepare_inputs(
    *,
    tokenizer: PreTrainedTokenizerBase,
    formatted_pairs: list[str],
    prefix_tokens: list[int],
    suffix_tokens: list[int],
    device: str,
) -> dict[str, torch.Tensor]:
    """
    构造 Qwen3-Reranker 输入。

    总 Token 预算：
        prefix + pair + suffix <= RERANKER_MAX_LENGTH

    pair 超长时仅截断 pair 主体，
    永远保留 system prefix 和 assistant suffix，
    因为 yes/no 打分依赖固定结尾。
    """

    reserved_tokens = (
        len(prefix_tokens)
        + len(suffix_tokens)
    )

    pair_max_length = (
        RERANKER_MAX_LENGTH
        - reserved_tokens
    )

    if pair_max_length <= 0:
        raise RuntimeError(
            "RERANKER_MAX_LENGTH 太小，"
            "无法容纳 Qwen3-Reranker "
            "固定 prefix/suffix。"
            f"max_length={RERANKER_MAX_LENGTH}, "
            f"reserved={reserved_tokens}",
        )

    encoded = tokenizer(
        formatted_pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=pair_max_length,
        add_special_tokens=False,
    )

    input_ids: list[list[int]] = []

    for pair_token_ids in encoded["input_ids"]:
        full_token_ids = (
            prefix_tokens
            + pair_token_ids
            + suffix_tokens
        )

        if len(full_token_ids) > RERANKER_MAX_LENGTH:
            raise RuntimeError(
                "内部 Token 长度计算异常："
                f"{len(full_token_ids)} > "
                f"{RERANKER_MAX_LENGTH}",
            )

        input_ids.append(
            full_token_ids,
        )

    padded = tokenizer.pad(
        {
            "input_ids": input_ids,
        },
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return {
        key: value.to(device)
        for key, value in padded.items()
    }


@torch.inference_mode()
def _compute_relevance_scores(
    *,
    model: PreTrainedModel,
    inputs: dict[str, torch.Tensor],
    token_true_id: int,
    token_false_id: int,
    supports_logits_to_keep: bool,
) -> list[float]:
    """
    计算 Qwen3-Reranker 相关性分数。

    官方逻辑：
    取序列最后位置的 logits，
    只比较 "no" 和 "yes" 两个 Token，
    再归一化。

    score = P(yes | yes/no)
    """

    model_kwargs: dict[str, object] = {
        "use_cache": False,
    }

    # 新版 Qwen3ForCausalLM 支持 logits_to_keep。
    # Reranker 只需要最后一个位置的 logits，
    # 因此只计算最后位置可以显著减少 LM Head 中间张量。
    if supports_logits_to_keep:
        model_kwargs["logits_to_keep"] = 1

    outputs = model(
        **inputs,
        **model_kwargs,
    )

    # padding_side='left'，
    # 因此 batch 中每条序列的有效最后位置都位于 -1。
    last_token_logits = (
        outputs.logits[:, -1, :]
        .float()
    )

    yes_logits = last_token_logits[
        :,
        token_true_id,
    ]

    no_logits = last_token_logits[
        :,
        token_false_id,
    ]

    binary_logits = torch.stack(
        [
            no_logits,
            yes_logits,
        ],
        dim=1,
    )

    probabilities = torch.softmax(
        binary_logits,
        dim=1,
    )

    scores = probabilities[
        :,
        1,
    ]

    if not torch.isfinite(scores).all():
        raise RuntimeError(
            "Reranker 返回了 NaN 或无穷分数",
        )

    return [
        float(score)
        for score in scores.cpu().tolist()
    ]


def _score_documents(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    query: str,
    documents: list[str],
    prefix_tokens: list[int],
    suffix_tokens: list[int],
    token_true_id: int,
    token_false_id: int,
    device: str,
    supports_logits_to_keep: bool,
) -> list[float]:
    """按 batch 对全部候选文档进行打分。"""

    scores: list[float] = []

    for batch_start in range(
        0,
        len(documents),
        RERANKER_BATCH_SIZE,
    ):
        batch_documents = documents[
            batch_start:
            batch_start + RERANKER_BATCH_SIZE
        ]

        formatted_pairs = [
            _format_instruction(
                query=query,
                document=document,
            )
            for document in batch_documents
        ]

        inputs = _prepare_inputs(
            tokenizer=tokenizer,
            formatted_pairs=formatted_pairs,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            device=device,
        )

        batch_scores = _compute_relevance_scores(
            model=model,
            inputs=inputs,
            token_true_id=token_true_id,
            token_false_id=token_false_id,
            supports_logits_to_keep=supports_logits_to_keep,
        )

        scores.extend(
            batch_scores,
        )

        # 尽快释放本批中间 Tensor 引用。
        del inputs

    return scores


# ============================================================
# 7. 模型生命周期
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """服务启动时仅加载一次 Qwen3-Reranker。"""

    _validate_runtime()
    _validate_device(
        RERANKER_DEVICE,
    )

    torch_dtype = _resolve_dtype(
        RERANKER_DEVICE,
    )

    logger.info(
        "开始加载 Qwen3 Reranker："
        "model=%s, device=%s, dtype=%s, "
        "max_length=%s, batch_size=%s",
        RERANKER_MODEL_PATH,
        RERANKER_DEVICE,
        torch_dtype,
        RERANKER_MAX_LENGTH,
        RERANKER_BATCH_SIZE,
    )

    load_start_time = time.perf_counter()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL_PATH,
            padding_side="left",
        )

        # 对 batch 的最后位置取 logits，
        # 必须使用 left padding。
        tokenizer.padding_side = "left"

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError(
                    "Tokenizer 同时缺少 "
                    "pad_token_id 和 eos_token_id",
                )

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        model = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL_PATH,
            torch_dtype=torch_dtype,
        )

        model.to(
            RERANKER_DEVICE,
        )

        model.eval()

        forward_parameters = inspect.signature(
            model.forward,
        ).parameters

        supports_logits_to_keep = (
            "logits_to_keep"
            in forward_parameters
        )

        if supports_logits_to_keep:
            logger.info(
                "检测到 logits_to_keep 支持，"
                "将只计算最后位置 logits",
            )
        else:
            logger.warning(
                "当前 Transformers/Qwen3 实现"
                "不支持 logits_to_keep；"
                "功能仍可运行，但中间 logits "
                "内存占用会更高。"
            )

        token_true_id = _get_single_token_id(
            tokenizer,
            "yes",
        )

        token_false_id = _get_single_token_id(
            tokenizer,
            "no",
        )

        if token_true_id == token_false_id:
            raise RuntimeError(
                "yes/no Token ID 相同，"
                "Tokenizer 状态异常",
            )

        prefix_tokens = tokenizer.encode(
            SYSTEM_PREFIX,
            add_special_tokens=False,
        )

        suffix_tokens = tokenizer.encode(
            ASSISTANT_SUFFIX,
            add_special_tokens=False,
        )

        reserved_tokens = (
            len(prefix_tokens)
            + len(suffix_tokens)
        )

        if reserved_tokens >= RERANKER_MAX_LENGTH:
            raise RuntimeError(
                "RERANKER_MAX_LENGTH 太小："
                f"max_length={RERANKER_MAX_LENGTH}, "
                f"prefix+suffix={reserved_tokens}",
            )

        actual_device = _get_actual_device(
            model,
        )

        # 启动自检：
        # 使用一正一负两个简单样例，确认推理链路能正常得到有限分数。
        self_test_documents = [
            "The capital of China is Beijing.",
            "Bananas are usually yellow when ripe.",
        ]

        with inference_lock:
            self_test_scores = _score_documents(
                model=model,
                tokenizer=tokenizer,
                query="What is the capital of China?",
                documents=self_test_documents,
                prefix_tokens=prefix_tokens,
                suffix_tokens=suffix_tokens,
                token_true_id=token_true_id,
                token_false_id=token_false_id,
                device=actual_device,
                supports_logits_to_keep=(
                    supports_logits_to_keep
                ),
            )

        if len(self_test_scores) != 2:
            raise RuntimeError(
                "Reranker 启动自检返回数量异常",
            )

        if not all(
            0.0 <= score <= 1.0
            for score in self_test_scores
        ):
            raise RuntimeError(
                "Reranker 启动自检分数越界："
                f"{self_test_scores}",
            )

        if (
            self_test_scores[0]
            <= self_test_scores[1]
        ):
            raise RuntimeError(
                "Reranker 启动语义自检失败："
                "明显相关文档得分没有高于无关文档。"
                f"scores={self_test_scores}"
            )

    except Exception:
        logger.exception(
            "Qwen3 Reranker 模型加载或自检失败："
            "model=%s",
            RERANKER_MODEL_PATH,
        )
        raise

    load_latency = (
        time.perf_counter()
        - load_start_time
    )

    app.state.reranker_model = model
    app.state.tokenizer = tokenizer
    app.state.actual_device = actual_device
    app.state.token_true_id = token_true_id
    app.state.token_false_id = token_false_id
    app.state.prefix_tokens = prefix_tokens
    app.state.suffix_tokens = suffix_tokens
    app.state.supports_logits_to_keep = (
        supports_logits_to_keep
    )

    logger.info(
        "Qwen3 Reranker 模型加载成功："
        "model=%s, actual_device=%s, "
        "yes_token_id=%s, no_token_id=%s, "
        "prefix_tokens=%s, suffix_tokens=%s, "
        "self_test_scores=%s, latency=%.3fs",
        RERANKER_MODEL_PATH,
        actual_device,
        token_true_id,
        token_false_id,
        len(prefix_tokens),
        len(suffix_tokens),
        [
            round(score, 6)
            for score in self_test_scores
        ],
        load_latency,
    )

    try:
        yield

    finally:
        app.state.reranker_model = None
        app.state.tokenizer = None

        try:
            del model
            del tokenizer
        except UnboundLocalError:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "Qwen3 Reranker 服务已关闭",
        )


# ============================================================
# 8. FastAPI 应用
# ============================================================

app = FastAPI(
    title="Qwen3 Reranker Service",
    description=(
        "使用 Qwen3-Reranker-0.6B "
        "对 query 和候选 documents "
        "执行相关性打分与重新排序"
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="检查 Reranker 服务状态",
)
def health(
    http_request: Request,
) -> HealthResponse:
    """检查模型是否已正确加载。"""

    reranker_model = getattr(
        http_request.app.state,
        "reranker_model",
        None,
    )

    tokenizer = getattr(
        http_request.app.state,
        "tokenizer",
        None,
    )

    if (
        reranker_model is None
        or tokenizer is None
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="Reranker 模型尚未加载",
        )

    return HealthResponse(
        status="ok",
        model=RERANKER_MODEL_PATH,
        device=http_request.app.state.actual_device,
        max_documents=RERANKER_MAX_DOCUMENTS,
        batch_size=RERANKER_BATCH_SIZE,
    )


@app.post(
    "/rerank",
    response_model=RerankResponse,
    tags=["Reranker"],
    summary="对候选文档重新排序",
)
def rerank(
    rerank_request: RerankRequest,
    http_request: Request,
) -> RerankResponse:
    """计算 query 与每个候选文档的相关性并返回 Top-K。"""

    model: PreTrainedModel | None = getattr(
        http_request.app.state,
        "reranker_model",
        None,
    )

    tokenizer: PreTrainedTokenizerBase | None = getattr(
        http_request.app.state,
        "tokenizer",
        None,
    )

    if (
        model is None
        or tokenizer is None
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="Reranker 模型尚未加载",
        )

    request_start_time = time.perf_counter()

    documents = rerank_request.documents

    logger.info(
        "开始执行 Qwen3 Reranker："
        "document_count=%s, top_k=%s, "
        "batch_size=%s, max_length=%s",
        len(documents),
        rerank_request.top_k,
        min(
            RERANKER_BATCH_SIZE,
            len(documents),
        ),
        RERANKER_MAX_LENGTH,
    )

    try:
        with inference_lock:
            scores = _score_documents(
                model=model,
                tokenizer=tokenizer,
                query=rerank_request.query,
                documents=documents,
                prefix_tokens=(
                    http_request.app.state
                    .prefix_tokens
                ),
                suffix_tokens=(
                    http_request.app.state
                    .suffix_tokens
                ),
                token_true_id=(
                    http_request.app.state
                    .token_true_id
                ),
                token_false_id=(
                    http_request.app.state
                    .token_false_id
                ),
                device=(
                    http_request.app.state
                    .actual_device
                ),
                supports_logits_to_keep=(
                    http_request.app.state
                    .supports_logits_to_keep
                ),
            )

    except Exception as exc:
        logger.exception(
            "Qwen3 Reranker 推理失败："
            "document_count=%s",
            len(documents),
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Reranker 推理失败",
        ) from exc

    if len(scores) != len(documents):
        logger.error(
            "Reranker 返回分数数量异常："
            "expected=%s, actual=%s",
            len(documents),
            len(scores),
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Reranker 返回结果数量异常",
        )

    indexed_results = [
        (
            index,
            score,
            document,
        )
        for index, (
            score,
            document,
        ) in enumerate(
            zip(
                scores,
                documents,
                strict=True,
            )
        )
    ]

    # score 降序；
    # 同分时保持原始文档索引顺序。
    ranked_results = sorted(
        indexed_results,
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    top_results = [
        RerankResult(
            index=index,
            score=round(
                float(score),
                6,
            ),
            document=document,
        )
        for (
            index,
            score,
            document,
        ) in ranked_results[
            :rerank_request.top_k
        ]
    ]

    latency = (
        time.perf_counter()
        - request_start_time
    )

    logger.info(
        "Qwen3 Reranker 执行完成："
        "document_count=%s, returned=%s, "
        "score_max=%.6f, score_min=%.6f, "
        "latency=%.3fs",
        len(documents),
        len(top_results),
        max(scores),
        min(scores),
        latency,
    )

    return RerankResponse(
        results=top_results,
    )
