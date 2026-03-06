# LLM API 自动化测试

不依赖任何官方 SDK，纯 HTTP 层面对 LLM API 进行多协议、多层级、全参数的自动化测试。

## 支持的协议

| 协议 | 端点 | 说明 |
|------|------|------|
| **OpenAI** | `/v1/chat/completions` | Chat Completions API |
| **Anthropic** | `/v1/messages` | Messages API |
| **Vertex AI** | `/{model}:generateContent` | Gemini API |
| **Response** | `/v1/responses` | OpenAI Response API |

每个协议均支持配置多个模型，并区分思考模型 (thinking) 和非思考模型 (non_thinking)。

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| HTTP 客户端 | httpx | AsyncClient + asyncio 异步并发 |
| 测试引擎 | pytest + pytest-asyncio | 异步测试、参数化、自动收集 |
| JSON 解析 | json5 | 支持 JSONC 注释语法 |
| 重试 | tenacity | 指数退避重试 |
| 控制台 | rich | 进度条 + 彩色输出 |
| 组合测试 | allpairspy | 正交实验法 (pairwise) |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yaml`，填入你的 API 接入点和密钥：

```yaml
protocols:
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "sk-your-key"
    models:
      thinking: []
      non_thinking:
        - "gpt-4o"

  anthropic:
    base_url: "https://api.anthropic.com"
    api_key: "sk-ant-your-key"
    auth_header: "x-api-key"    # 或 "authorization"
    models:
      thinking:
        - "claude-sonnet-4-20250514"
      non_thinking:
        - "claude-sonnet-4-20250514"
```

不需要测试的协议，把 `models` 留空即可跳过。

### 3. 运行

```bash
# 通过入口脚本运行（带 Rich 面板）
python run.py

# 或直接用 pytest
python -m pytest tests/ -v

# 只测某个协议
python run.py --protocol openai

# 只测某一层
python -m pytest tests/test_level1.py -v
```

## 测试层级

通过 `config.yaml` 中的 `test_levels` 控制开关：

```yaml
test_levels:
  level1: true    # 基础连通性
  level2: false   # 功能可用性
  level3: false   # 组合覆盖率
```

### L1 - 基础连通性

遍历每个协议的每个模型（含 thinking/non_thinking），执行三项基础测试：

| 测试 | 内容 |
|------|------|
| `test_connectivity_minimal` | 只发送必须字段，验证连通性 |
| `test_non_stream_usage` | 非流式请求，验证响应结构和 usage |
| `test_stream_events_and_usage` | 流式请求，验证 SSE 事件完整性和 usage |

### L2 - 功能可用性

自动读取 `protocols/{name}_request.jsonc` 中值为 `true` 的每个非必须字段，逐个发送请求测试。每个参数执行两层断言：

```
通用断言：响应结构完整吗？                ← 所有参数都跑
              +
专属断言：这个参数的效果体现了吗？          ← 有注册就跑
```

例如 `tools` 参数不仅检查响应 200，还检查是否返回了 `tool_calls`；`response_format: json_object` 检查返回的 content 是否为合法 JSON。

此外还包含 Anthropic 专项测试：
- **提示词缓存测试** - 发送足量 token + 随机前缀，验证 `cache_creation_input_tokens`
- **Beta 头测试** - 逐个测试 `anthropic-beta` 头的可用性

### L3 - 组合覆盖率

使用 [allpairspy](https://github.com/thombashi/allpairspy) 正交实验法，对所有可选参数生成 pairwise 组合。自动处理参数互斥（如 `temperature` 与 `top_p`、`max_tokens` 与 `max_completion_tokens`）。组合中每个启用的参数都会触发其专属断言。

## 项目结构

```
auto-llm-api-test/
├── config.yaml                     # 主配置文件
├── protocols/                      # 协议配置
│   ├── openai_request.jsonc        #   请求体字段开关 (true=测试)
│   ├── openai_response.json        #   响应体模板 (参考用)
│   ├── anthropic_request.jsonc
│   ├── anthropic_response.json
│   ├── vertex_request.jsonc
│   ├── vertex_response.json
│   ├── response_request.jsonc
│   └── response_response.json
├── prompts/                        # 提示词
│   ├── short.txt                   #   短提示词 (连通性/功能测试)
│   ├── long.txt                    #   长提示词 (缓存测试)
│   └── images/
│       └── test.png                #   测试图片 (多模态)
├── src/
│   ├── config.py                   # 配置加载
│   ├── client.py                   # httpx 异步客户端
│   ├── middleware.py               # 重试 + AI 检验
│   ├── logger.py                   # curl/response 日志
│   ├── console.py                  # Rich 控制台
│   └── protocols/
│       ├── base.py                 #   ProtocolBuilder 基类
│       ├── openai.py               #   请求构建 + 响应断言
│       ├── anthropic.py
│       ├── vertex.py
│       └── response.py
├── tests/
│   ├── test_level1.py              # L1 基础连通性
│   ├── test_level2.py              # L2 逐参数可用性
│   └── test_level3.py              # L3 正交组合覆盖
├── logs/                           # 运行时生成
│   ├── success.log                 #   成功请求的 curl + response
│   └── failure.log                 #   失败请求的 curl + response + 原因
├── conftest.py                     # pytest fixtures
├── pytest.ini
├── run.py                          # 入口脚本
└── requirements.txt
```

## 请求体配置

每个协议的可测试字段在 `protocols/{name}_request.jsonc` 中用 bool 值声明：

```jsonc
{
  "model": true,              // [必须] 模型名称
  "messages": true,           // [必须] 消息列表

  "temperature": true,        // 需要测试
  "top_p": true,              // 需要测试
  "logprobs": false,          // 暂不测试
  "seed": true,               // 需要测试
  "tools": true,              // 需要测试
  "response_format": true,    // 需要测试
  // ...
}
```

改为 `true` 即自动加入 L2 和 L3 测试，无需修改测试代码。

## 断言机制

每次 API 请求返回后执行两层断言：

**通用断言** (`assert_non_stream_response`) — 检查响应基本结构：

- OpenAI: 有 `id`、`choices[0].message.content`、`finish_reason`
- Anthropic: 有 `id`、`type=message`、`content[0].type=text`、`stop_reason`
- Vertex: 有 `candidates[0].content.parts[0].text`
- Response: 有 `id`、`status=completed`、`output[0].content[0].output_text`

**专属断言** (`assert_option_response`) — 检查参数效果是否体现：

| 参数 | 断言内容 |
|------|----------|
| `tools` / `tool_choice` | `finish_reason=tool_calls` 时检查 tool_calls 结构 |
| `response_format` | content 是合法 JSON |
| `logprobs` | 响应中有 logprobs 数据 |
| `thinking` | content 中有 thinking block |
| `messages_image_*` | 响应有足够长度的文本描述 |
| `responseMimeType` | 响应文本是合法 JSON |
| `seed` | 响应中有 system_fingerprint / modelVersion |

未注册专属断言的参数只走通用断言，不会影响测试。

## 后置中间件

### 重试

失败的请求自动进行 tenacity 指数退避重试：

```yaml
retry:
  enabled: true
  max_attempts: 3
  multiplier: 1        # wait = multiplier * 2^attempt
  max_wait: 30         # 最大等待秒数
```

### AI 检验

重试耗尽后仍失败的请求，可以发送给 AI 判断是否为**预期行为**（如不支持的参数被正确拒绝）。判定为预期的测试标记为 UNSTABLE（黄色），独立于 PASS 和 FAIL：

```yaml
ai_validation:
  enabled: false
  base_url: "https://api.openai.com/v1"
  api_key: "sk-xxx"
  model: "gpt-4o"
```

## 日志

每次运行自动记录到 `logs/` 目录：

- `success.log` — 所有成功请求的 curl 命令 + 响应体
- `failure.log` — 所有失败请求的 curl 命令 + 响应体 + 失败原因

curl 中的 API Key 自动脱敏。

## 贡献

参见 [CONTRIBUTING_CN.md](CONTRIBUTING_CN.md)，涵盖：

- 给现有协议添加新的请求参数
- 添加参数专属断言
- 添加全新协议
