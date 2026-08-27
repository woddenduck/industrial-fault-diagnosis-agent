# 测试说明

项目把默认离线测试与真实服务测试分开。执行 `python -m pytest` 时不会访问模型、RAG 或本机端口；只有显式选择 `e2e` 标记时才会调用真实服务。

## 测试分层

| 层级 | 真实服务 | 主要覆盖 |
|---|---|---|
| 仓库封装 | 否 | 目录、配置、脚本、依赖、文档和 Git 排除策略 |
| 单元与契约 | 否 | Schema、State、工具、RAG Client、API 和 Runner |
| Graph 集成 | 否 | Node、Routing、完整条件路由和失败收口 |
| 诊断质量 | 否 | 温度、报警、维修异常、受控操作和人工复核 |
| HTTP E2E | 是 | 六服务健康状态与真实诊断链路 |

## 离线测试

安装依赖后，在仓库根目录执行：

```bash
python -m pip install -r requirements/agent.txt -r requirements/dev.txt
python -m pytest
```

需要查看各验收程序的详细输出时，可以直接运行：

```bash
python tests/test_stage8_packaging.py
python tests/test_stage8_repository.py
python tests/test_diagnosis_quality.py
```

正式离线测试集中包含：

- `test_agent_api.py`：FastAPI 接口及 HTTP 状态映射；
- `test_agent_runner_runtime.py`：Runner 输入、终态和字段过滤；
- `test_api_contract.py`：请求与响应 Schema；
- `test_tools.py`：设备、维修和报告工具；
- `test_nodes.py`、`test_routing.py`、`test_graph_workflow.py`：生产 Graph；
- `test_rag_client.py`：RAG HTTP 客户端重试和错误映射；
- `test_diagnosis_quality.py`：风险分级质量；
- `test_stage8_packaging.py`、`test_stage8_repository.py`：发布仓库契约。

## 真实 HTTP E2E

先启动六项服务，并确认 `.env` 中的 `E2E_KNOWLEDGE_BASE_ID` 指向已构建的真实知识库：

```bash
set -a
source .env
set +a

python -m pytest -m e2e tests/test_agent_http_e2e.py -s
```

也可以直接运行测试文件：

```bash
python tests/test_agent_http_e2e.py
```

E2E 会验证 RAG 健康检查、Agent 健康检查、Graph 契约、缺少设备编号的条件路由和完整真实诊断链路。它不会使用 Mock，运行前必须启动模型与业务服务。

## 测试数据原则

- 离线测试使用仓库中的示例设备数据或测试替身；
- E2E 的知识库 ID 只通过环境变量提供；
- 模型、PDF、向量库、索引、日志和本机 `.env` 不进入 Git；
- 新增生产模块时，必须由默认 `python -m pytest` 收集对应离线测试。
