# 测试说明

## 1. 测试分层

| 层级 | 是否需要真实服务 | 作用 |
|---|---|---|
| 封装验收 | 否 | 检查目录、配置、脚本、依赖和文档 |
| 离线业务测试 | 否 | 验证 State、Node、Graph、Schema 和安全规则 |
| 质量验收 | 否 | 验证温度、报警、历史异常和受控操作的风险分级 |
| HTTP E2E | 是 | 验证六服务真实调用链 |

## 2. 封装验收

```bash
python tests/test_stage8_packaging.py
python tests/test_stage8_repository.py
```

两项程序都可以直接运行，并打印每项检查过程。

## 3. 离线测试

安装 Agent 和测试依赖后执行：

```bash
conda run -n agent_system \
  python -m pytest
```

`pyproject.toml` 默认排除 `e2e` 标记，因此不会意外请求本机服务。

`test_structured_output.py` 中前三项会调用真实 vLLM，已归入
`e2e`；其余 JSON 解析与 Schema 校验仍属于离线测试。

风险分级专项验收：

```bash
conda run -n agent_system \
  python tests/test_diagnosis_quality.py
```

## 4. 真实 HTTP E2E

先确认：

- 六项服务已启动；
- RAG `/health` 为 healthy；
- `.env` 中填写了真实知识库 ID。

然后执行：

```bash
set -a
source .env
set +a

conda run -n agent_system \
  python tests/test_agent_http_e2e.py
```

也可以通过 Pytest 显式执行：

```bash
conda run -n agent_system \
  python -m pytest -m e2e tests/test_agent_http_e2e.py -s
```

只验证真实 LLM 的结构化输出：

```bash
conda run -n agent_system \
  python -m pytest -m e2e tests/test_structured_output.py -s
```

## 5. 测试数据原则

- 离线测试使用示例设备数据或测试替身。
- E2E 测试不使用 Mock。
- 知识库 ID 必须通过环境变量提供。
- PDF、向量库和索引不进入 Git 仓库。
