# Industrial RAG V1 项目验收报告

## 1. 项目概述

本项目为工业设备故障诊断 RAG V1，当前以 **Siemens SINAMICS G120C** 为主要演示设备。

系统完成了从工业说明书上传、知识库构建、混合检索、Reranker 重排、上下文决策，到 LLM 生成带来源引用诊断回答的完整业务闭环。

V1 的核心目标不是继续优化单一算法指标，而是验证以下能力能够以完整工程形式稳定运行：

- 上传工业设备 PDF；
- 自动解析并切分文档；
- 构建 Knowledge Base；
- 构建 Embedding / Vector Store；
- 构建 Knowledge Base 专属 BM25 索引；
- BM25 + Vector 混合召回；
- Reranker 重排；
- 按 Knowledge Base 与设备型号隔离检索；
- Evidence Decision 控制是否允许调用 LLM；
- 生成结构化工业故障诊断回答；
- 输出可追溯的 PDF / Page / Section / Chunk ID；
- 对不存在的 Knowledge Base 进行受控拒绝；
- 服务异常时保持明确的 HTTP 错误语义。

---

## 2. V1 系统链路

```text
Industrial PDF
    ↓
Document Upload
    ↓
Document Service
    ↓
document_id
    ↓
Knowledge Base Builder
    ↓
Document Parser
    ↓
Chunking
    ↓
┌─────────────────────────────┐
│                             │
↓                             ↓
BM25 Index              Embedding
                              ↓
                         Vector Store
                              ↓
                       Knowledge Base
                              ↓
                      knowledge_base_id
                              ↓
                         User Question
                              ↓
                       Query Rewriter
                              ↓
                  Knowledge Base Filter
                              ↓
               ┌──────────────┴──────────────┐
               ↓                             ↓
             BM25                          Vector
               └──────────────┬──────────────┘
                              ↓
                         Hybrid / RRF
                              ↓
                          Reranker
                              ↓
                       Context Builder
                              ↓
                    Evidence Decision
                    ┌─────────┴─────────┐
                    ↓                   ↓
              Evidence Enough      Evidence Weak
                    ↓                   ↓
                   LLM                Reject
                    ↓
              Structured Answer
                    ↓
                  Sources
                    ↓
        PDF / Page / Section / Chunk ID
```

---

## 3. 测试环境

### 3.1 Industrial RAG API

```text
http://127.0.0.1:8000
```

### 3.2 模型与服务

| 模块 | 实际运行模型 / 服务 | 状态 |
|---|---|---|
| LLM | Qwen/Qwen3-8B | Healthy |
| Embedding | BAAI/bge-m3 | Healthy |
| Reranker | Qwen/Qwen3-Reranker-0.6B | Healthy |
| Gateway | http://127.0.0.1:6008 | Healthy |
| Vector Store | Chroma | Healthy |
| Industrial RAG API | http://127.0.0.1:8000 | Healthy |

端到端测试时 `/health` 返回：

```text
status = healthy
```

Gateway、LLM、Embedding、Reranker 和 Vector Store 均正常。

---

## 4. Demo 输入数据

### 4.1 测试 PDF

```text
G120C_list_man_0223_zh-CHS.pdf
```

文件大小：

```text
4.84 MB
```

设备型号：

```text
G120C
```

### 4.2 测试问题

```text
F30021怎么处理？
```

---

## 5. 文档上传验收

调用：

```text
POST /documents/upload
```

测试结果：

```text
HTTP 200
status      = duplicate
document_id = doc_43f7223d
```

由于该 PDF 此前已经进入系统，Document Service 通过重复文档检测识别出已有文档，并返回原有 `document_id`。

这证明：

- PDF 文件校验正常；
- 文档 SHA256 / Duplicate 检测链路正常；
- 重复上传不会产生新的重复文档；
- 系统可以复用已有 `document_id` 继续后续 Demo。

验收结果：

```text
PASS
```

---

## 6. Knowledge Base 构建验收

请求：

```json
{
  "knowledge_name": "G120C故障知识库",
  "device_model": "G120C",
  "document_ids": [
    "doc_43f7223d"
  ]
}
```

返回：

```text
knowledge_base_id = kb_e630cd8a79f3
status            = ready
```

构建统计：

| 指标 | 结果 |
|---|---:|
| Chunk Count | 1541 |
| Vector Count | 1541 |
| BM25 Document Count | 1541 |
| Knowledge Base Status | ready |
| Build Report | passed=true |
| Errors | [] |

生成文件：

```text
data/chunks/kb_e630cd8a79f3.json
data/bm25/kb_e630cd8a79f3.pkl
```

一致性结果：

```text
chunk_count = vector_count = bm25_document_count = 1541
```

说明本次 Knowledge Base 的 Chunk、Vector 和 BM25 三个数据层数量一致。

Knowledge Base 完整构建耗时：

```text
406.293 s
```

验收结果：

```text
PASS
```

---

## 7. RAG Chat 端到端验收

请求：

```json
{
  "question": "F30021怎么处理？",
  "knowledge_base_id": "kb_e630cd8a79f3",
  "device_model": "G120C",
  "history": [],
  "history_summary": null
}
```

返回：

```text
HTTP 200
status             = answered
llm_called         = true
retrieval_mode     = rerank
rerank_executed    = true
degraded           = false
evidence_sufficient= true
allow_llm          = true
context_status     = ANSWERABLE
```

本次请求成功经过：

```text
Question
→ Query Rewrite
→ KB Filter
→ Hybrid Retrieval
→ Reranker
→ Context Builder
→ Evidence Decision
→ LLM
→ Structured Answer
→ Citation
```

验收结果：

```text
PASS
```

---

## 8. 结构化回答验收

系统成功生成包含以下结构的工业故障诊断结果：

```text
可能原因
证据来源
排查步骤
风险提示
仍需确认的信息
```

针对 F30021，回答中给出了可能原因，例如：

- 功率电缆接地；
- 电机接地；
- 变流器损坏；
- 立即制动引起硬件直流监控响应；
- 制动电阻短路。

同时给出了对应排查步骤以及仍需确认的故障值信息。

回答长度：

```text
331 字符
```

验收结果：

```text
PASS
```

---

## 9. Evidence Decision 验收

本次正常问题的 Context Decision：

```json
{
  "evidence_sufficient": true,
  "allow_llm": true,
  "context_status": "ANSWERABLE",
  "evidence_mode": "rerank",
  "reason": "SAFE_CONTEXT_READY"
}
```

说明：

```text
检索到可靠证据
        ↓
Context Builder 判断上下文可回答
        ↓
allow_llm = true
        ↓
LLM 才被调用
```

该机制避免系统在没有可靠资料时直接让 LLM 自由生成答案。

验收结果：

```text
PASS
```

---

## 10. 来源引用验收

本次回答返回 3 个 Sources。

### Source 1

```text
Document : G120C_list_man_0223_zh-CHS.pdf
Page     : 707
Section  : 4.2 故障和报警列表
Chunk ID : doc_43f7223d_section_0973_00
```

### Source 2

```text
Document : G120C_list_man_0223_zh-CHS.pdf
Page     : 715-716
Section  : 4.2 故障和报警列表
Chunk ID : doc_43f7223d_section_0981_01
```

### Source 3

```text
Document : G120C_list_man_0223_zh-CHS.pdf
Page     : 715
Section  : 4.2 故障和报警列表
Chunk ID : doc_43f7223d_section_0981_00
```

引用完整性检查：

| 检查项 | 结果 |
|---|---|
| PDF / Document | PASS |
| Page | PASS |
| Section | PASS |
| Chunk ID | PASS |
| 单个 Source 可完整追溯 | PASS |

三个 Source 均可以独立完成：

```text
Answer
→ Source
→ PDF
→ Page
→ Section
→ Chunk ID
```

验收结果：

```text
PASS
```

---

## 11. 异常场景验收

测试不存在的 Knowledge Base：

```text
knowledge_base_id =
kb_demo_not_exist_a386ce0c3cf0
```

返回：

```text
HTTP 404
```

响应：

```json
{
  "detail": "Knowledge Base 不存在：kb_demo_not_exist_a386ce0c3cf0"
}
```

说明系统在 Retrieval 之前已经完成 KB 存在性校验，没有继续进入正常 LLM 回答链路。

错误语义：

```text
Knowledge Base Not Found
        ↓
HTTP 404
```

验收结果：

```text
PASS
```

---

## 12. 性能数据

本次端到端运行中的关键耗时：

| 阶段 | 耗时 |
|---|---:|
| Health Check | 0.373 s |
| Models Check | 0.033 s |
| Document Upload | 0.052 s |
| Knowledge Base Build | 406.293 s |
| Chat HTTP Response | 8.152 s |
| Retrieval | 3994.59 ms |
| LLM | 4145.99 ms |
| QA Service Total | 8140.64 ms |
| Missing KB Reject | 0.006 s |

### 性能观察

Knowledge Base 构建是一次性离线操作，本次 1541 个 Chunk 的完整构建约耗时 406 秒。

在线 Chat 请求约 8.15 秒，其中主要时间来自：

```text
Retrieval ≈ 3.99 s
LLM       ≈ 4.15 s
```

对于 V1 Demo，该性能已经能够完成完整业务演示。

后续版本可进一步优化：

- Retrieval latency；
- Reranker latency 统计；
- Embedding / Vector 检索缓存；
- LLM 推理吞吐；
- Knowledge Base 增量构建。

---

## 13. Stage 9 最终端到端验收

最终测试结果：

```text
Passed: 36/36

FINAL RESULT: PASS
```

完整链路：

```text
Health
→ Models
→ Upload
→ Knowledge Base Build
→ Chat
→ Citation
→ Missing KB Reject
```

全部通过。

---

## 14. V1 项目能力总结

| 项目能力 | 验收结果 |
|---|---|
| PDF 上传 | ✅ |
| 文件类型与非空校验 | ✅ |
| 重复文档检测 | ✅ |
| document_id | ✅ |
| Knowledge Base 创建 | ✅ |
| 文档解析 | ✅ |
| Chunking | ✅ |
| Embedding | ✅ |
| Vector Store | ✅ |
| BM25 | ✅ |
| Hybrid Retrieval | ✅ |
| RRF Fusion | ✅ |
| Reranker | ✅ |
| Knowledge Base 隔离 | ✅ |
| 设备型号过滤 | ✅ |
| Context Builder | ✅ |
| Evidence Decision | ✅ |
| LLM 调用控制 | ✅ |
| 结构化故障回答 | ✅ |
| PDF 来源引用 | ✅ |
| Page 来源引用 | ✅ |
| Section 来源引用 | ✅ |
| Chunk ID 来源引用 | ✅ |
| 不存在 KB 拒绝 | ✅ |
| HTTP 错误语义 | ✅ |
| Request ID | ✅ |
| 健康检查 | ✅ |
| 模型状态查询 | ✅ |
| Stage 9 E2E | **36/36 PASS** |

> 注：Day20 其它独立测试（例如基础问题集评测、Fault Injection）属于此前阶段的独立验收。
> 本报告中的具体数字以本次 Stage 9 `test_rag_v1_demo.py` 端到端运行日志为准。

---

## 15. V1 已知限制

当前版本主要用于验证工业故障诊断 RAG 的完整工程闭环，因此仍存在以下限制：

1. 当前 Demo 主要围绕 Siemens SINAMICS G120C 文档进行验证。
2. Knowledge Base 构建仍属于全量构建流程，尚未实现完整增量索引。
3. Knowledge Base 构建时间较长，大文档首次构建需要数分钟。
4. 在线问答延迟仍有优化空间。
5. 当前引用能够追溯到 PDF、页码、章节和 Chunk，但尚未提供 PDF 页面可视化跳转。
6. 当前 Evidence Decision 使用 V1 规则，后续仍可通过正式评测集优化阈值。
7. 当前系统主要处理文档知识，尚未正式接入设备实时状态 API、历史维修数据库等工业数据源。
8. 当前为 V1 单机工程验证版本，尚未加入完整的用户权限、数据库、任务队列、监控告警等生产基础设施。

这些限制不影响 V1 的核心目标：

> 验证工业文档从上传、建库、检索、证据控制到带引用诊断回答的完整业务链路。

---

## 16. 项目阶段结论

Industrial RAG V1 已达到 Day20 阶段性目标。

系统已经从前期的单独实验脚本：

```text
Chunk Test
Vector Test
BM25 Test
Reranker Test
Context Test
```

演进为可以真实运行的业务系统：

```text
Upload PDF
→ Create Knowledge Base
→ Retrieval
→ Reranker
→ Evidence Control
→ LLM
→ Citation
→ Controlled Rejection
```

Stage 9 最终端到端测试结果：

```text
36 / 36 PASS
```

因此：

# Industrial RAG V1：验收通过

后续工作应进入 V2 / Day21+，而不是继续修改已经通过验收的 V1 主链。
