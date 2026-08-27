# API 说明

## 1. Agent API

默认地址：`http://127.0.0.1:8010`

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | Agent、Graph 和 RAG 健康状态 |
| GET | `/v1/graph` | Graph 节点与支持的意图 |
| POST | `/v1/diagnose` | 执行设备诊断 |

诊断请求示例：

```json
{
  "query": "请综合诊断 DEVICE-001 当前运行状态，并判断是否存在过热风险。",
  "device_id": "DEVICE-001",
  "device_model": "G120C",
  "knowledge_base_id": "kb_xxxxxxxxxxxx",
  "create_report": true
}
```

主要响应字段：

| 字段 | 含义 |
|---|---|
| `status` | completed、needs_input、insufficient_evidence、human_review_required 或 failed |
| `device_status` | 设备当前状态 |
| `maintenance_history` | 维修记录聚合结果 |
| `rag_decision` | 证据是否充分、是否允许调用 LLM |
| `sources` | 文档、页码、章节和 Chunk |
| `risk_level` | unknown、low、medium、high 或 critical |
| `execution_trace` | Graph 节点执行轨迹 |

推荐传入 `X-Request-ID`；服务会在响应头和响应体中返回它。

## 2. Industrial RAG API

默认地址：`http://127.0.0.1:8000`

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | RAG 及依赖组件健康状态 |
| GET | `/models` | 当前模型信息 |
| POST | `/documents/upload` | 上传 PDF、Markdown 或 TXT |
| POST | `/knowledge-bases` | 构建知识库 |
| POST | `/chat` | 基于指定知识库问答 |

### 上传文档

```bash
curl -sS -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@/你的路径/G120C_manual.pdf" \
  -F "device_model=G120C" \
  -F "document_type=technical_manual" \
  -F "version=2026"
```

记录返回的 `document_id`。

### 构建知识库

```bash
curl -sS -X POST http://127.0.0.1:8000/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_name": "G120C 技术知识库",
    "device_model": "G120C",
    "document_ids": ["doc_xxxxxxxx"]
  }'
```

记录返回的 `knowledge_base_id`，写入本机 `.env`：

```text
DEFAULT_KNOWLEDGE_BASE_ID=kb_xxxxxxxxxxxx
E2E_KNOWLEDGE_BASE_ID=kb_xxxxxxxxxxxx
```

### RAG 问答

```json
{
  "question": "变频器温度过高时应该检查哪些方面？",
  "knowledge_base_id": "kb_xxxxxxxxxxxx",
  "device_model": "G120C",
  "history": [],
  "history_summary": null
}
```

## 3. HTTP 状态约定

| 情况 | HTTP 状态 |
|---|---:|
| 请求 Schema 不合法 | 422 |
| 设备或知识库不存在 | 404 |
| RAG/LLM 不可用 | 503 |
| 上游超时 | 504 |
| 证据不足或等待人工复核 | 200，使用业务状态表达 |
