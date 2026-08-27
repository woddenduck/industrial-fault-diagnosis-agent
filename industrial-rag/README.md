# Industrial RAG V1

面向工业设备故障诊断场景的 Retrieval-Augmented Generation（RAG）V1 系统。

当前版本以 **Siemens SINAMICS G120C** 为主要演示设备，完成了从工业 PDF 上传、知识库构建、BM25 + Vector 混合检索、Reranker 重排、上下文证据判断，到 LLM 生成带来源引用故障诊断回答的完整业务闭环。

---

## 1. 项目介绍

Industrial RAG V1 的目标不是单独验证某一个 RAG 算法，而是将文档处理、检索、重排、上下文控制、LLM 和 API 服务整合为一个可实际演示的工业故障诊断系统。

当前 V1 支持：

- 上传工业设备 PDF；
- 基于 SHA256 检测重复文档；
- 生成 `document_id`；
- 创建独立 Knowledge Base；
- PDF 解析与 Chunking；
- Embedding 向量化；
- Chroma Vector Store；
- Knowledge Base 专属 BM25 索引；
- BM25 + Vector Hybrid Retrieval；
- RRF 融合；
- Reranker 重排；
- Knowledge Base 隔离；
- 设备型号过滤；
- Query Rewrite；
- Context 去重与 Token Budget；
- Prompt Injection 基础检测；
- Evidence Decision；
- 无可靠证据时拒答；
- LLM 结构化故障诊断；
- PDF / Page / Section / Chunk ID 来源引用；
- Reranker 失败时 fallback；
- Health / Models / Request ID；
- 端到端 Demo、Fault Injection 与基础评测。

当前主要 Demo 设备：

```text
Siemens SINAMICS G120C
```

---

## 2. 系统架构

### 2.1 文档建库链路

```text
Industrial PDF
    ↓
POST /documents/upload
    ↓
Document Service
    ↓
File Validation
    ↓
SHA256 Duplicate Detection
    ↓
document_id
    ↓
POST /knowledge-bases
    ↓
Knowledge Builder
    ↓
Document Parser
    ↓
Chunking
    ↓
┌───────────────────────┐
│                       │
↓                       ↓
BM25 Index          Embedding
                        ↓
                   Vector Store
                        ↓
                Knowledge Base READY
                        ↓
               knowledge_base_id
```

### 2.2 在线问答链路

```text
User Question
    ↓
POST /chat
    ↓
QA Service
    ↓
Retrieval Service
    ↓
Query Rewriter
    ↓
Knowledge Base Filter
    ↓
Device Model Filter
    ↓
┌─────────────────────────────┐
│                             │
↓                             ↓
BM25                        Vector
│                             │
└──────────────┬──────────────┘
               ↓
          Hybrid / RRF
               ↓
           Reranker
               ↓
        Context Builder
               ↓
       Evidence Decision
        ┌──────┴──────┐
        ↓             ↓
Evidence Enough   Evidence Weak
        ↓             ↓
       LLM           Reject
        ↓
Structured Answer
        ↓
Sources
        ↓
PDF / Page / Section / Chunk ID
```

---

## 3. 项目目录结构

```text
industrial-rag/
│
├── app/
│   ├── config.py
│   ├── main.py
│   ├── schemas.py
│   │
│   └── services/
│       ├── document_service.py
│       ├── knowledge_builder.py
│       ├── knowledge_service.py
│       ├── qa_service.py
│       └── retrieval_service.py
│
├── data/
│   ├── bm25/
│   ├── chunks/
│   ├── uploads/
│   ├── vector_store/
│   ├── documents.json
│   └── knowledge_base.json
│
├── rag/
│   ├── bm25_retriever.py
│   ├── chunking.py
│   ├── config.py
│   ├── context_builder.py
│   ├── data_contracts.py
│   ├── document_parser.py
│   ├── history_manager.py
│   ├── hybrid_retriever.py
│   ├── injection_detector.py
│   ├── prompt_builder.py
│   ├── query_rewriter.py
│   ├── rerank_pipeline.py
│   ├── token_budget.py
│   ├── tokenizer.py
│   └── vector_store.py
│
├── reports/
│   └── rag_v1_evaluation.json
│
├── tests/
│   ├── rag_v1_test_questions.json
│   ├── test_document_service.py
│   ├── test_knowledge_build.py
│   ├── test_qa_service.py
│   ├── test_rag_v1_demo.py
│   ├── test_rag_v1_evaluation.py
│   ├── test_rag_v1_faults.py
│   ├── test_retrieval_service.py
│   └── test_stage1.py
│
├── rag_v1_report.md
└── README.md
```

### 3.1 `app/`

业务 API 与 Service 层。

- `main.py`
  - FastAPI 入口；
  - `/documents/upload`；
  - `/knowledge-bases`；
  - `/chat`；
  - `/health`；
  - `/models`。

- `schemas.py`
  - HTTP Request / Response 数据契约。

- `services/document_service.py`
  - 文件校验；
  - SHA256；
  - Duplicate 检测；
  - 文档保存；
  - `document_id` 管理。

- `services/knowledge_builder.py`
  - Knowledge Base 完整构建流程。

- `services/knowledge_service.py`
  - Knowledge Base 元数据；
  - 状态管理；
  - `knowledge_base_id` 查询。

- `services/retrieval_service.py`
  - 检索业务编排；
  - KB / Device Model 校验；
  - Hybrid Retrieval；
  - Reranker；
  - Context Builder。

- `services/qa_service.py`
  - Evidence Decision；
  - Prompt；
  - LLM 调用；
  - 最终结构化回答。

### 3.2 `rag/`

核心 RAG 算法与上下文处理模块。

### 3.3 `data/`

运行时数据目录。

```text
data/uploads/
```

保存原始上传文档。

```text
data/chunks/
```

保存 Knowledge Base 对应的 Chunk JSON。

```text
data/bm25/
```

保存 Knowledge Base 专属 BM25 Index。

```text
data/vector_store/
```

保存 Chroma Vector Store。

```text
data/documents.json
```

保存 Document Metadata。

```text
data/knowledge_base.json
```

保存 Knowledge Base Metadata。

### 3.4 `tests/`

包含模块测试、故障注入、评测和 Stage 9 端到端 Demo。

---

## 4. 环境要求

当前项目在 Linux / AutoDL 环境中完成验证。

推荐环境：

```text
Python 3.10
Linux
NVIDIA GPU
CUDA 环境
```

当前 RAG 业务环境：

```text
Conda Environment: rag_system
```

系统还依赖独立运行的：

- LLM Service；
- Embedding Service；
- Reranker Service；
- Gateway。

Python 依赖以当前 `rag_system` 环境为准。

建议在冻结 V1 后导出：

```bash
pip freeze > requirements.txt
```

后续即可使用：

```bash
pip install -r requirements.txt
```

恢复 Python 环境。

---

## 5. 模型与服务

V1 最终验收时实际运行模型：

| 模块 | 模型 |
|---|---|
| LLM | `Qwen/Qwen3-8B` |
| Embedding | `BAAI/bge-m3` |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` |

当前验收环境中的服务地址：

| 服务 | 地址 |
|---|---|
| LLM | `http://127.0.0.1:6006` |
| Gateway | `http://127.0.0.1:6008` |
| Reranker | `http://127.0.0.1:6009` |
| Embedding | `http://127.0.0.1:6010` |
| Industrial RAG API | `http://127.0.0.1:8000` |

实际地址由项目配置决定。

运行后建议以以下接口为准确认真实状态：

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl http://127.0.0.1:8000/models
```

---

## 6. 启动顺序

完整 V1 Demo 必须按照依赖顺序启动服务。

```text
① LLM
   ↓
② Embedding
   ↓
③ Reranker
   ↓
④ Gateway
   ↓
⑤ Industrial RAG API
```

LLM、Embedding、Reranker 与 Gateway 使用当前上层项目已有的部署方式启动。

进入 `industrial-rag` 项目目录后启动业务 API：

```bash
conda activate rag_system
```

```bash
python -m app.main
```

也可以使用 Uvicorn：

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000
```

启动成功后：

```bash
curl http://127.0.0.1:8000/health
```

完整 Demo 推荐看到：

```json
{
  "status": "healthy"
}
```

如果 Reranker 不可用而其它关键组件正常，系统可能进入：

```text
degraded
```

此时 Retrieval Pipeline 可以 fallback，但完整 Stage 9 Demo 建议所有服务均为 `healthy`。

---

## 7. API 说明

FastAPI 文档：

```text
http://127.0.0.1:8000/docs
```

### 7.1 Health

```http
GET /health
```

示例：

```bash
curl http://127.0.0.1:8000/health
```

用于检查：

- Gateway；
- LLM；
- Embedding；
- Reranker；
- Vector Store。

---

### 7.2 Models

```http
GET /models
```

示例：

```bash
curl http://127.0.0.1:8000/models
```

用于确认当前实际运行的：

- LLM；
- Embedding；
- Reranker。

---

### 7.3 上传文档

```http
POST /documents/upload
```

请求类型：

```text
multipart/form-data
```

示例：

```bash
curl -X POST \
  http://127.0.0.1:8000/documents/upload \
  -F "file=@data/uploads/G120C_list_man_0223_zh-CHS.pdf" \
  -F "device_model=G120C" \
  -F "document_type=operation_manual" \
  -F "version=V1"
```

可能返回：

```json
{
  "status": "created",
  "document_id": "doc_xxxxxxxx"
}
```

重复文件可能返回：

```json
{
  "status": "duplicate",
  "document_id": "doc_43f7223d"
}
```

V1 使用文件内容哈希进行重复文档识别，而不是只依赖文件名。

---

### 7.4 创建 Knowledge Base

```http
POST /knowledge-bases
```

请求示例：

```bash
curl -X POST \
  http://127.0.0.1:8000/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_name": "G120C故障知识库",
    "device_model": "G120C",
    "document_ids": [
      "doc_43f7223d"
    ]
  }'
```

构建流程：

```text
document_id
    ↓
PDF Parser
    ↓
Chunking
    ↓
Embedding
    ↓
Chroma
    ↓
BM25 Index
    ↓
Consistency Check
    ↓
status = ready
```

成功后返回：

```text
knowledge_base_id = kb_xxxxxxxxxxxx
```

只有 `status=ready` 的 Knowledge Base 才应该进入正常 Retrieval。

---

### 7.5 Chat

```http
POST /chat
```

示例：

```bash
curl -X POST \
  http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "F30021怎么处理？",
    "knowledge_base_id": "kb_e630cd8a79f3",
    "device_model": "G120C",
    "history": []
  }'
```

典型响应：

```json
{
  "status": "answered",
  "answer": "...",
  "llm_called": true,
  "sources": [
    {
      "document": "G120C_list_man_0223_zh-CHS.pdf",
      "page": "707",
      "section": "4.2 故障和报警列表",
      "chunk_id": "doc_43f7223d_section_0973_00"
    }
  ],
  "knowledge_base_id": "kb_e630cd8a79f3",
  "device_model": "G120C",
  "degraded": false,
  "retrieval_mode": "rerank",
  "rerank_executed": true,
  "decision": {
    "evidence_sufficient": true,
    "allow_llm": true,
    "context_status": "ANSWERABLE"
  },
  "request_id": "..."
}
```

---

## 8. Knowledge Base 创建流程

### 8.1 `document_id`

代表：

```text
系统中保存的一份原始文档
```

例如：

```text
doc_43f7223d
```

### 8.2 `knowledge_base_id`

代表：

```text
已经完成解析、Chunk、Embedding、Vector 和 BM25 构建，
可以参与 RAG 检索的知识集合
```

例如：

```text
kb_e630cd8a79f3
```

两者关系：

```text
PDF
 ↓
document_id
 ↓
Knowledge Base Build
 ↓
knowledge_base_id
```

不要直接把 `document_id` 当成 Retrieval Scope。

---

## 9. Chat 示例

测试问题：

```text
F30021怎么处理？
```

系统最终回答结构固定为：

```text
可能原因
证据来源
排查步骤
风险提示
仍需确认的信息
```

V1 正常回答要求：

```text
Evidence Sufficient
        ↓
allow_llm = true
        ↓
LLM Called
        ↓
Structured Answer
        ↓
Sources
```

来源必须能够追溯到：

```text
PDF
↓
Page
↓
Section
↓
Chunk ID
```

例如：

```text
Document : G120C_list_man_0223_zh-CHS.pdf
Page     : 707
Section  : 4.2 故障和报警列表
Chunk ID : doc_43f7223d_section_0973_00
```

---

## 10. 拒答与错误处理

### 10.1 无可靠证据

当 Context Builder 判断：

```text
evidence_sufficient = false
```

系统应：

```text
allow_llm = false
```

避免在缺乏可靠工业资料时让 LLM 自由生成答案。

### 10.2 Knowledge Base 不存在

例如：

```text
kb_not_exist
```

系统返回：

```http
HTTP 404
```

而不是继续进入正常 LLM 调用链路。

### 10.3 Reranker 不可用

Reranker 服务不可用时，系统允许：

```text
Hybrid Retrieval
    ↓
Reranker Failed
    ↓
Fallback
    ↓
Hybrid Top-K
    ↓
Context Builder
```

并在响应中暴露 degraded / rerank 状态。

### 10.4 LLM Timeout

LLM Timeout 使用统一错误响应，并返回：

```text
request_id
```

便于日志追踪。

---

## 11. 测试方法

### 11.1 Document Service

```bash
python tests/test_document_service.py
```

### 11.2 Knowledge Base Build

```bash
python tests/test_knowledge_build.py
```

### 11.3 Retrieval Service

```bash
python tests/test_retrieval_service.py
```

### 11.4 QA Service

```bash
python tests/test_qa_service.py
```

### 11.5 Fault Injection

```bash
python tests/test_rag_v1_faults.py
```

主要验证：

```text
Unsupported File
Duplicate PDF
Device Model Conflict
No Evidence
Reranker Fallback
LLM Timeout
```

### 11.6 V1 Evaluation

```bash
python tests/test_rag_v1_evaluation.py
```

测试问题：

```text
tests/rag_v1_test_questions.json
```

评测输出：

```text
reports/rag_v1_evaluation.json
```

### 11.7 Stage 9 End-to-End Demo

推荐最终交付前运行：

```bash
python tests/test_rag_v1_demo.py \
  --pdf data/uploads/G120C_list_man_0223_zh-CHS.pdf
```

该测试真实执行：

```text
Health
→ Models
→ Upload
→ Knowledge Base Build
→ Chat
→ Citation
→ Missing KB Reject
```

V1 最终验收结果：

```text
Passed: 36/36

FINAL RESULT: PASS
```

详细结果见：

```text
rag_v1_report.md
```

---

## 12. 当前 Demo 数据

Stage 9 最终端到端测试使用：

```text
PDF:
G120C_list_man_0223_zh-CHS.pdf
```

```text
Document ID:
doc_43f7223d
```

```text
Knowledge Base ID:
kb_e630cd8a79f3
```

该 Knowledge Base：

```text
Chunk Count         : 1541
Vector Count        : 1541
BM25 Document Count : 1541
Status              : ready
```

测试问题：

```text
F30021怎么处理？
```

最终：

```text
HTTP 200
status = answered
llm_called = true
retrieval_mode = rerank
rerank_executed = true
```

并成功返回 3 个完整来源。

---

## 13. 已知限制

Industrial RAG V1 当前主要用于验证完整工程闭环，还不是最终生产系统。

当前限制包括：

1. 主要针对 Siemens SINAMICS G120C 资料完成验证。
2. Knowledge Base 当前以完整构建流程为主，尚未完善增量索引。
3. 大型工业 PDF 首次建库耗时较长。
4. 在线 Retrieval 与 LLM 延迟仍有进一步优化空间。
5. 当前来源已经支持 PDF / Page / Section / Chunk ID，但尚未实现前端 PDF 页面直接跳转。
6. Evidence Decision 仍属于 V1 规则，可在后续正式评测集中继续优化。
7. 尚未正式接入设备实时状态 API。
8. 尚未正式接入历史维修数据库。
9. 尚未加入完整用户权限体系。
10. 尚未加入生产级数据库、任务队列、监控、告警与分布式部署。
11. 当前主要为单机 / AutoDL 工程验证环境。
12. LLM 输出仍需遵守工业现场安全流程，不能替代设备厂商正式维修规范及合格工程人员判断。

---

## 14. V1 验收状态

最终 Stage 9：

```text
Health
→ Models
→ Upload
→ KB Build
→ Chat
→ Citation
→ Reject
```

完整通过。

```text
36 / 36 PASS
```

因此当前 V1 主链建议冻结，不再继续在 Day20 修改：

- Chunk 算法；
- BM25；
- Vector Retrieval；
- RRF；
- Reranker；
- Prompt 总体结构；
- Token Budget 总体策略。

后续优化进入：

```text
Day21 / Industrial RAG V2
```

---

## 15. 后续方向

V2 可以重点考虑：

```text
设备实时状态 API
+
历史维修记录数据库
+
文档 RAG
```

形成：

```text
用户问题
    ↓
设备型号 / 故障码
    ↓
实时运行状态
    ↓
历史维修记录
    ↓
官方说明书
    ↓
Hybrid Retrieval
    ↓
Reranker
    ↓
Evidence Fusion
    ↓
工业故障诊断 Agent
```

同时可以进一步加入：

- 增量 Knowledge Base；
- 多设备、多型号知识库管理；
- Retrieval / Reranker 缓存；
- 更系统的离线评测；
- RAGAS / 自定义工业评测指标；
- 故障诊断工作流；
- Tool Calling；
- Agent；
- 维修记录回写；
- 可观测性；
- 前端 Demo。

---

## 16. 项目状态

```text
Industrial RAG V1
Status: Completed
Stage 9 E2E: 36/36 PASS
```

当前版本已经完成：

> Industrial PDF → Knowledge Base → Hybrid Retrieval → Reranker → Evidence Control → LLM → Citation → Controlled Rejection

的完整工业故障诊断 RAG V1 闭环。
