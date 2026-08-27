from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from app.config import (
    EMBEDDING_MAX_BATCH_SIZE,
    EMBEDDING_MAX_TEXT_LENGTH,
    RERANKER_MAX_DOCUMENTS,
)


class GatewaySchema(BaseModel):
    """Gateway 所有请求和响应模型的公共基类。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class Message(GatewaySchema):
    """单条对话消息。"""

    role: Literal["system", "user", "assistant"]
    content: str = Field(
        ...,
        min_length=1,
        description="消息内容，不能为空或只有空格",
    )


class ChatRequest(GatewaySchema):
    """POST /chat 的请求体。"""

    model: str = Field(
        default="Qwen/Qwen3-8B",
        min_length=1,
        description="调用的模型名称",
    )
    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="对话消息列表，至少包含一条消息",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="生成随机性，范围为 0 到 2",
    )
    max_tokens: int = Field(
        default=1536,
        gt=0,
        le=4096,
        description="最大生成 Token 数，必须大于 0",
    )
    stream: StrictBool = Field(
        default=False,
        description="是否使用真实流式输出，必须是真正的布尔值",
    )


ServiceType = Literal[
    "gateway",
    "llm",
    "embedding",
    "reranker",
]

ServiceAvailability = Literal[
    "available",
    "unavailable",
]


class HealthServices(GatewaySchema):
    """三个下游模型服务的可用状态。"""

    llm: ServiceAvailability
    embedding: ServiceAvailability
    reranker: ServiceAvailability


class HealthResponse(GatewaySchema):
    """GET /health 的统一响应体。"""

    status: Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]
    services: HealthServices


class ModelInfo(GatewaySchema):
    """单个模型的信息。"""

    id: str = Field(..., min_length=1)
    status: Literal["available"] = "available"


class ModelsResponse(GatewaySchema):
    """GET /models 的响应体。"""

    models: list[ModelInfo]


class TokenUsage(GatewaySchema):
    """模型调用的 Token 使用情况。"""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ChatResponse(GatewaySchema):
    """Gateway 普通聊天响应。"""

    request_id: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    usage: TokenUsage
    latency: float = Field(..., ge=0)


class ErrorDetail(GatewaySchema):
    """统一错误对象。"""

    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ErrorResponse(GatewaySchema):
    """Gateway 的统一错误响应。"""

    request_id: str = Field(..., min_length=1)
    error: ErrorDetail
    service: ServiceType = Field(
        ...,
        description=(
            "产生错误的组件。下游调用失败时为具体模型服务，"
            "Gateway 自身错误时为 gateway"
        ),
    )


# ==================================================
# Embedding
# ==================================================

EmbeddingText = Annotated[
    StrictStr,
    Field(
        min_length=1,
        max_length=EMBEDDING_MAX_TEXT_LENGTH,
    ),
]


class EmbeddingRequest(GatewaySchema):
    """POST /embed 的请求体。"""

    texts: list[EmbeddingText] = Field(
        ...,
        min_length=1,
        max_length=EMBEDDING_MAX_BATCH_SIZE,
        description=(
            "待编码文本。支持单个字符串或字符串列表；"
            f"最终统一为列表，最多 {EMBEDDING_MAX_BATCH_SIZE} 条，"
            f"单条最多 {EMBEDDING_MAX_TEXT_LENGTH} 个字符"
        ),
    )
    normalize: StrictBool = Field(
        default=True,
        description="是否对向量执行归一化，必须是真正的布尔值",
    )

    @field_validator("texts", mode="before")
    @classmethod
    def normalize_texts(cls, value: object) -> object:
        """兼容单字符串输入，并统一转换成字符串列表。"""

        if isinstance(value, str):
            return [value]
        return value


class EmbeddingResponse(GatewaySchema):
    """POST /embed 的统一响应体。"""

    request_id: str = Field(..., min_length=1)
    embeddings: list[list[float]] = Field(..., min_length=1)
    dimension: int = Field(..., gt=0)
    count: int = Field(..., gt=0)
    latency: float = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_embedding_shape(self) -> "EmbeddingResponse":
        """保证响应数量、维度和向量形状一致。"""

        if self.count != len(self.embeddings):
            raise ValueError("count 必须等于 embeddings 数量")

        for index, vector in enumerate(self.embeddings):
            if len(vector) != self.dimension:
                raise ValueError(
                    f"embeddings[{index}] 的维度与 dimension 不一致"
                )

        return self


# ==================================================
# Reranker
# ==================================================

NonEmptyDocument = Annotated[
    StrictStr,
    Field(min_length=1),
]


class RerankRequest(GatewaySchema):
    """POST /rerank 的请求体。"""

    query: StrictStr = Field(
        ...,
        min_length=1,
        description="检索问题，不能为空或只有空格",
    )
    documents: list[NonEmptyDocument] = Field(
        ...,
        min_length=1,
        max_length=RERANKER_MAX_DOCUMENTS,
        description=(
            "候选文档列表，至少一条，"
            f"最多 {RERANKER_MAX_DOCUMENTS} 条"
        ),
    )
    top_k: StrictInt = Field(
        default=5,
        gt=0,
        description="返回结果数量，必须大于 0 且不能超过文档数量",
    )

    @model_validator(mode="after")
    def validate_top_k(self) -> "RerankRequest":
        """执行 top_k 与 documents 数量之间的跨字段校验。"""

        if self.top_k > len(self.documents):
            raise ValueError("top_k 不能大于 documents 数量")
        return self


class RerankResult(GatewaySchema):
    """单条精排结果。"""

    index: int = Field(..., ge=0)
    score: float
    document: str = Field(..., min_length=1)


class RerankResponse(GatewaySchema):
    """POST /rerank 的统一响应体。"""

    request_id: str = Field(..., min_length=1)
    results: list[RerankResult] = Field(..., min_length=1)
    latency: float = Field(..., ge=0)
