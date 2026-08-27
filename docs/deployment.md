# 部署说明

## 1. 环境要求

- Linux
- Python 3.10
- Conda
- `curl`、`setsid`
- NVIDIA GPU；Qwen3-8B 建议至少 24GB 显存
- 可访问或已经缓存所需模型

项目使用多个 Conda 环境，避免 vLLM、RAG 和 CPU 模型服务之间发生依赖冲突。

## 2. 创建环境

如果环境已经存在，只需要在对应环境中安装或核对依赖。

```bash
conda create -n agent_system python=3.10 -y
conda run -n agent_system python -m pip install -r requirements/agent.txt -r requirements/dev.txt

conda create -n llm_gateway python=3.10 -y
conda run -n llm_gateway python -m pip install -r requirements/gateway.txt

conda create -n rag_system python=3.10 -y
conda run -n rag_system python -m pip install -r requirements/rag.txt

conda create -n embedding_service python=3.10 -y
conda run -n embedding_service python -m pip install -r requirements/embedding.txt

conda create -n reranker_service python=3.10 -y
conda run -n reranker_service python -m pip install -r requirements/reranker.txt
```

vLLM 与 CUDA、驱动和 PyTorch 版本关系紧密。创建环境后，应按照当前 GPU 环境安装兼容版本：

```bash
conda create -n vllm_qwen3 python=3.10 -y
conda run -n vllm_qwen3 python -m pip install -r requirements/vllm.txt
```

安装后执行：

```bash
conda run -n vllm_qwen3 python -c "import vllm; print(vllm.__version__)"
```

不要为了统一依赖而把所有服务安装到同一个 Conda 环境。

## 3. 配置

```bash
cp .env.example .env
```

至少确认：

- 六个 `*_CONDA_ENV`
- `HF_HOME`
- 三个模型名称或本地路径
- 六项服务端口
- `DEFAULT_KNOWLEDGE_BASE_ID`
- `E2E_KNOWLEDGE_BASE_ID`

`.env` 是本机配置，不提交 GitHub。相对路径按照仓库根目录解析。

## 4. 准备 RAG 运行数据

公开仓库不保存 PDF、向量库和索引。首次部署需要先启动服务，再通过 API 上传文档并构建知识库，方法见 [API 说明](api.md)。

如果只是把当前已经验收的 AutoDL 项目迁移到新发布目录，可以在旧服务全部停止后复制运行数据：

```bash
rsync -a \
  ~/autodl-tmp/enterprise-llm-agent-platform/industrial-rag/data/ \
  ./industrial-rag/data/
```

这些运行数据只保留在部署机器，不提交 Git。

## 5. 启动

先做静态检查：

```bash
bash deployment/start_all.sh --validate
```

确认端口未被旧服务占用后启动：

```bash
bash deployment/start_all.sh
```

启动顺序：

```text
LLM → Embedding → Reranker → Gateway → RAG → Agent
```

## 6. 健康检查与停止

```bash
bash deployment/health_check.sh
bash deployment/stop_all.sh
```

Reranker 不可用时可以显示为降级；LLM、Embedding、Gateway、RAG 和 Agent 属于必要服务。

## 7. 日志与 PID

- 启动日志：`logs/`
- RAG 日志：`industrial-rag/logs/`
- PID：`deployment/run/`

上述内容均被 `.gitignore` 排除。

## 8. 常见问题

### 端口已被占用

先运行旧项目对应的停止脚本，再检查：

```bash
ss -lntp | grep -E ':6006|:6008|:6009|:6010|:8000|:8010'
```

### Knowledge Base 不存在

确认 `.env` 中的 ID 属于当前 `industrial-rag/data`，或者重新上传文档并构建知识库。

### 模型被重复下载

将 `.env` 中的 `HF_HOME` 指向当前机器已有的 Hugging Face 缓存目录。
