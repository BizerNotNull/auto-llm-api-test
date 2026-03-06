# 贡献指南：如何添加测试样例与断言

本文档面向需要为项目扩展测试能力的开发者，涵盖三类典型操作：

1. 给现有协议添加新的请求参数测试
2. 添加新的响应断言规则
3. 添加一个全新的协议

---

## 项目结构速览

```
protocols/
  {name}_request.jsonc   ← 请求体字段开关（jsonc，哪些字段参与测试）
  {name}_response.json   ← 响应体模板（参考用，断言逻辑在代码中）
src/
  config.py              ← 配置加载 + 必须字段映射
  client.py              ← httpx 异步客户端（URL 构建、Header 构建）
  protocols/
    base.py              ← ProtocolBuilder 抽象基类
    openai.py            ← 各协议的 Builder 实现
    anthropic.py
    vertex.py
    response.py
  middleware.py           ← 重试 + AI 检验
  logger.py              ← curl/response 日志
conftest.py              ← pytest fixtures + BUILDERS 注册表
tests/
  test_level1.py          ← L1 基础连通性（自动遍历所有协议×模型）
  test_level2.py          ← L2 逐个可选参数（自动读取 jsonc 开关）
  test_level3.py          ← L3 正交组合（自动生成 pairwise 组合）
```

**核心机制**：L2 和 L3 的测试用例是**自动生成**的——它们读取 `protocols/{name}_request.jsonc` 中值为 `true` 的非必须字段，然后调用 Builder 的 `build_with_option(model, prompt, option_name)` 构造请求体。你只需要在两个地方登记即可让新参数自动参与测试。

---

## 一、给现有协议添加新的请求参数

以给 OpenAI 添加 `logit_bias` 参数为例，完整步骤如下。

### 第 1 步：在 jsonc 中登记字段开关

编辑 `protocols/openai_request.jsonc`，添加一行：

```jsonc
  "logit_bias": true,          // Token 偏置 (token_id -> bias, -100~100)
```

- `true` 表示该字段需要参与 L2/L3 测试
- `false` 表示暂不测试（仍需要在 Builder 中写好构造逻辑，以备后续开启）
- 在 `// [必须]` 注释只用于标记必须字段，普通可选项不需要加

### 第 2 步：在 Builder 的 `build_with_option` 中添加构造逻辑

编辑 `src/protocols/openai.py`，在 `OpenAIBuilder.build_with_option` 方法的 `option_values` 字典中添加：

```python
option_values = {
    # ... 已有的 ...
    "logit_bias": {"logit_bias": {
        "15339": -100,   # token id for "hello" → 禁止
        "1820": 5,       # token id for "hi" → 轻微偏好
    }},
}
```

**关键规则**：

- 字典的 key（`"logit_bias"`）必须与 jsonc 中的字段名**完全一致**
- value 是一个 dict，会被 `body.update(value)` 合并到请求体中
- 所以 value 的 key 就是你要发给 API 的实际字段名
- 如果该参数依赖另一个参数（如 `tool_choice` 依赖 `tools`），value 中需要同时包含依赖项：

```python
# tool_choice 依赖 tools，所以 value 里也要带上 tools
"tool_choice": {"tools": [_TOOL_DEF], "tool_choice": "auto"},
```

- 如果该参数需要覆盖 `messages`（如多模态传图），直接在 value 中包含完整的 `"messages": [...]`：

```python
"messages_image_url": {
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}},
        ]},
    ],
},
```

- 如果某参数在非流式下无意义（如 `stream_options`），value 设为空 dict `{}`

### 第 3 步（推荐）：添加参数专属断言

编辑 `src/protocols/openai.py`，在 `assert_option_response` 方法中添加一个 `elif` 分支：

```python
def assert_option_response(self, option_name: str, data: dict) -> list[str]:
    errors = []
    # ...已有的 if/elif...

    elif option_name == "logit_bias":
        content = message.get("content", "")
        if not content:
            errors.append("[logit_bias] Response content is empty")

    return errors
```

这一步可以先跳过——没有专属断言的参数只走通用断言，不会报错。但尽量补上，否则测试"通过"只代表服务端没返回 400，不代表参数真的生效了。

### 第 4 步（可选）：处理互斥关系

如果新参数与已有参数互斥，编辑 `tests/test_level3.py` 的 `MUTEX_GROUPS`：

```python
MUTEX_GROUPS = {
    "openai": [
        # ...已有的...
        {"logit_bias", "response_format"},  # 假设这两者互斥
    ],
}
```

互斥的含义：一个 set 内的参数**不会同时出现**在同一个 L3 组合中。

如果参数存在**依赖关系**（如 `top_logprobs` 必须配合 `logprobs`），在 `_is_valid_combination` 函数中添加检查：

```python
if protocol_name == "openai":
    if "top_logprobs" in enabled and "logprobs" not in enabled:
        return False
```

### 完成

不需要改 `test_level2.py` 或 `test_level3.py` 的测试函数——它们会自动从 jsonc 中读取 `true` 的字段并生成测试，自动调用通用断言 + 专属断言。运行 `pytest --collect-only -q` 可以验证新测试已被收集。

---

## 二、添加新的响应断言

### 断言的两层结构

每次测试请求返回后，框架执行**两层断言**：

```
1. 通用断言  assert_non_stream_response(data)     ← 所有请求都跑
                              +
2. 专属断言  assert_option_response(option_name, data) ← 按参数名分发
```

- **通用断言**：检查响应基本结构完整性（有 id、有 choices、有 content 等）
- **专属断言**：检查**这个参数的效果是否体现在响应中**（tools 请求是否返回了 tool_calls、logprobs 请求是否返回了 logprobs 数据、response_format=json 是否返回了合法 JSON 等）

L2（逐参数测试）和 L3（组合测试）都会自动调用这两层。L3 中会对组合里**每个启用的参数**逐一调用其专属断言。

### 断言方法清单

| 方法 | 用途 | 修改场景 |
|------|------|----------|
| `assert_non_stream_response(data)` | 所有非流式响应的通用结构断言 | 协议响应格式有变化时 |
| `assert_stream_events(events)` | 所有流式响应的通用事件断言 | 协议流式格式有变化时 |
| `assert_option_response(option_name, data)` | **某个参数的专属断言** | 添加/完善参数行为验证时 |
| `extract_usage(data)` | 从非流式响应提取 usage | usage 结构变化时 |
| `extract_stream_usage(events)` | 从流式事件提取 usage | usage 结构变化时 |

### 返回值约定

所有断言方法返回 `list[str]`——**失败原因列表**。空列表表示通过。

```python
def assert_non_stream_response(self, data: dict) -> list[str]:
    errors = []
    if "id" not in data:
        errors.append("Missing 'id' in response")
    return errors
```

测试框架这样使用（以 L2 为例）：

```python
# 通用断言
errors = builder.assert_non_stream_response(data)
# 专属断言（追加到同一个 errors 列表）
errors += builder.assert_option_response(option_name, data)

if errors:
    log_failure(test_name, ..., reason="; ".join(errors))
    raise RequestFailed(status, text, "; ".join(errors))
```

### 添加参数专属断言

编辑 `src/protocols/{name}.py` 中的 `assert_option_response` 方法。以 OpenAI 为例，当前已有的专属断言：

```python
def assert_option_response(self, option_name: str, data: dict) -> list[str]:
    errors = []
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})

    if option_name in ("tools", "tool_choice", "parallel_tool_calls"):
        # finish_reason=tool_calls 时，必须有 tool_calls
        if choice.get("finish_reason") == "tool_calls":
            tc = message.get("tool_calls")
            if not tc:
                errors.append(f"[{option_name}] finish_reason='tool_calls' but no tool_calls")
            # ... 检查每个 tool_call 的 name/arguments ...

    elif option_name == "response_format":
        # content 必须是合法 JSON
        content = message.get("content", "")
        if content:
            try:
                json5.loads(content)
            except Exception:
                errors.append(f"[response_format] content is not valid JSON")

    elif option_name == "logprobs":
        # 响应中必须有 logprobs 数据
        lp = choice.get("logprobs")
        if lp is None:
            errors.append("[logprobs] logprobs is null in response")

    # ... 更多参数 ...
    return errors
```

**添加新参数的断言**只需要在 `assert_option_response` 中加一个 `elif` 分支：

```python
    elif option_name == "logit_bias":
        # logit_bias 不改变响应结构，但内容应该正常
        content = message.get("content", "")
        if not content:
            errors.append("[logit_bias] Response content is empty")
```

没有注册专属断言的参数（即没有对应的 `elif` 分支），`assert_option_response` 返回空列表，只走通用断言。这样可以**渐进式补充**——先让参数跑起来，后续再补断言。

### 各协议已有的专属断言覆盖

**OpenAI**：tools/tool_choice/parallel_tool_calls（tool_calls 结构）、response_format（JSON 合法性）、logprobs/top_logprobs（logprobs 数据）、n（choices 数量）、seed（system_fingerprint）、messages_image_*（描述文本）、service_tier

**Anthropic**：tools/tool_choice（tool_use block）、thinking（thinking block）、system（文本输出）、messages_image_*（描述文本）、stop_sequences（stop_sequence 字段）

**Vertex**：tools/toolConfig（functionCall）、responseMimeType/responseSchema（JSON 合法性）、contents_image_*（描述文本）、seed（modelVersion）、responseLogprobs/logprobs（logprobs 数据）

**Response API**：tools/tool_choice/parallel_tool_calls（function_call 输出）、text（output_text）、input_image_*（描述文本）、reasoning（reasoning 输出项）、service_tier

### 修改通用断言

如果需要改变**所有请求**的断言行为（如新增检查 `system_fingerprint`），编辑 `assert_non_stream_response`：

```python
def assert_non_stream_response(self, data: dict) -> list[str]:
    errors = []
    # ...已有的检查...

    # 新增
    if "system_fingerprint" not in data:
        errors.append("Missing 'system_fingerprint' in response")

    return errors
```

流式断言同理，编辑 `assert_stream_events`。

---

## 三、添加一个全新的协议

以添加 `cohere` 协议为例。

### 第 1 步：创建请求体和响应体配置

创建 `protocols/cohere_request.jsonc`：

```jsonc
{
  // Cohere Chat 请求体配置
  "model": true,           // [必须]
  "message": true,         // [必须]

  "stream": true,
  "temperature": true,
  "max_tokens": true,
  "p": true,               // top_p
  "k": true,               // top_k
  "preamble": true,        // system prompt
  "tools": true,
  "seed": false
}
```

创建 `protocols/cohere_response.json`（参考用模板）：

```json
{
  "id": "xxx",
  "finish_reason": "COMPLETE",
  "message": {
    "role": "assistant",
    "content": [{"type": "text", "text": "string"}]
  },
  "usage": {
    "billed_units": {"input_tokens": 0, "output_tokens": 0},
    "tokens": {"input_tokens": 0, "output_tokens": 0}
  }
}
```

### 第 2 步：实现 Builder

创建 `src/protocols/cohere.py`：

```python
"""Cohere Chat 协议实现"""
import json5
from src.protocols.base import ProtocolBuilder


class CohereBuilder(ProtocolBuilder):

    def build_minimal(self, model: str, prompt: str) -> dict:
        return {
            "model": model,
            "message": prompt,
        }

    def build_non_stream(self, model: str, prompt: str, **kwargs) -> dict:
        body = self.build_minimal(model, prompt)
        body["stream"] = False
        body.update(kwargs)
        return body

    def build_stream(self, model: str, prompt: str, **kwargs) -> dict:
        body = self.build_minimal(model, prompt)
        body["stream"] = True
        body.update(kwargs)
        return body

    def build_with_option(self, model: str, prompt: str,
                          option_name: str, **kwargs) -> dict:
        option_values = {
            "temperature": {"temperature": 0.7},
            "max_tokens": {"max_tokens": 200},
            "p": {"p": 0.9},
            "k": {"k": 40},
            "preamble": {"preamble": "You are a helpful assistant."},
            "tools": {"tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"],
                    },
                },
            }]},
            "seed": {"seed": 42},
        }
        body = self.build_non_stream(model, prompt, **kwargs)
        if option_name in option_values:
            body.update(option_values[option_name])
        return body

    def assert_non_stream_response(self, data: dict) -> list[str]:
        errors = []
        if "id" not in data:
            errors.append("Missing 'id'")
        if "message" not in data:
            errors.append("Missing 'message'")
        elif data["message"].get("role") != "assistant":
            errors.append("Expected role='assistant'")
        if "finish_reason" not in data:
            errors.append("Missing 'finish_reason'")
        return errors

    def assert_stream_events(self, events: list[str]) -> list[str]:
        errors = []
        if not events:
            errors.append("No stream events")
            return errors
        has_text = False
        for ev in events:
            if ev == "[DONE]":
                continue
            try:
                data = json5.loads(ev)
                if data.get("type") == "content-delta":
                    has_text = True
            except Exception:
                pass
        if not has_text:
            errors.append("No text in stream")
        return errors

    def extract_usage(self, data: dict) -> dict | None:
        return data.get("usage")

    def extract_stream_usage(self, events: list[str]) -> dict | None:
        for ev in reversed(events):
            if ev == "[DONE]":
                continue
            try:
                data = json5.loads(ev)
                if data.get("usage"):
                    return data["usage"]
            except Exception:
                pass
        return None
```

### 第 3 步：注册 Builder + Client URL

在 `conftest.py` 的 `BUILDERS` 字典中注册：

```python
from src.protocols.cohere import CohereBuilder

BUILDERS = {
    # ...已有的...
    "cohere": CohereBuilder(),
}
```

在 `src/client.py` 的 `_build_url` 中添加 URL 规则：

```python
elif self.protocol.name == "cohere":
    return f"{base}/v2/chat"
```

在 `src/client.py` 的 `_build_headers` 中添加认证方式（如果不是标准 Bearer token）：

```python
elif self.protocol.name == "cohere":
    headers["Authorization"] = f"Bearer {self.protocol.api_key}"
```

### 第 4 步：添加必须字段映射

编辑 `src/config.py` 的 `get_required_fields` 函数：

```python
REQUIRED = {
    # ...已有的...
    "cohere": ["model", "message"],
}
```

### 第 5 步：在 config.yaml 中添加协议配置

```yaml
protocols:
  # ...已有的...
  cohere:
    base_url: "https://api.cohere.com"
    api_key: "xxx"
    models:
      thinking: []
      non_thinking:
        - "command-r-plus"
```

### 第 6 步（可选）：添加互斥组

编辑 `tests/test_level3.py` 的 `MUTEX_GROUPS`：

```python
MUTEX_GROUPS = {
    # ...已有的...
    "cohere": [
        {"temperature", "p"},   # 不建议同时用
    ],
}
```

### 第 7 步（可选）：添加协议特有测试

如果协议有特殊功能（类似 Anthropic 的缓存/beta头），在 `tests/test_level2.py` 中添加专用测试函数。参照 `test_anthropic_prompt_caching` 的模式即可。

### 完成

运行验证：

```bash
python -m pytest tests/ --collect-only -q
```

---

## 四、完整操作清单

### 添加请求参数（checklist）

- [ ] `protocols/{name}_request.jsonc` — 添加字段，设为 `true`
- [ ] `src/protocols/{name}.py` — 在 `build_with_option` 的 `option_values` 中添加构造
- [ ] `src/protocols/{name}.py` — 在 `assert_option_response` 中添加专属断言（可后补）
- [ ] `tests/test_level3.py` — 如有互斥，更新 `MUTEX_GROUPS`
- [ ] `protocols/{name}_response.json` — 如响应结构有变化，更新模板（参考用）
- [ ] `pytest --collect-only -q` — 验证新测试已收集

### 添加响应断言（checklist）

- [ ] `src/protocols/{name}.py` — 参数级断言加在 `assert_option_response`，全局断言加在 `assert_non_stream_response` / `assert_stream_events`

### 添加新协议（checklist）

- [ ] `protocols/{name}_request.jsonc` — 创建请求体开关
- [ ] `protocols/{name}_response.json` — 创建响应模板
- [ ] `src/protocols/{name}.py` — 实现 `ProtocolBuilder` 子类（8个方法 + `assert_option_response`）
- [ ] `conftest.py` — 在 `BUILDERS` 中注册
- [ ] `src/client.py` — `_build_url` 和 `_build_headers` 中添加分支
- [ ] `src/config.py` — `get_required_fields` 中添加必须字段
- [ ] `config.yaml` — 添加协议的 base_url、api_key、models
- [ ] `tests/test_level3.py` — 按需添加 `MUTEX_GROUPS`
- [ ] `pytest --collect-only -q` — 验证

---

## 五、关键约定

### option_values 的 key 命名

jsonc 中的字段名 = `build_with_option` 的 `option_name` 参数 = `option_values` 字典的 key。三者必须一致。

对于多模态等需要覆盖顶层字段的参数，命名建议使用 `{顶层字段}_{子类型}` 的格式：

```
messages_image_url       ← OpenAI 通过 messages 传图片 URL
messages_image_base64    ← OpenAI 通过 messages 传图片 base64
contents_image_base64    ← Vertex 通过 contents 传图片
input_image_url          ← Response API 通过 input 传图片
```

### 断言不要抛异常

断言方法（`assert_*`）只收集错误字符串并返回 list，不要在里面 `raise` 或 `assert`。异常由测试函数统一处理。

### Vertex 的 generationConfig 合并

Vertex 的多个参数（temperature、topP、maxOutputTokens 等）都嵌套在 `generationConfig` 中。`build_with_option` 中每个参数的 value 写成：

```python
"temperature": {
    "generationConfig": {"temperature": 0.7}
},
```

Builder 和 L3 的 `_build_combo_body` 会自动做 `generationConfig` 的合并（dict.update），不会互相覆盖。

### 空 dict 表示"此参数不构造请求体"

```python
"stream_options": {},       # 非流式下无意义，不添加任何字段
"cachedContent": {},        # 需要外部资源，暂不构建
```

空 dict 不会修改请求体，但该参数仍会出现在 jsonc 中（设为 `false` 则不参与测试）。

### 测试图片

多模态测试使用 `prompts/images/test.png`（1x1 红色 PNG）。如需更复杂的图片测试，在该目录下添加文件，然后在 Builder 中引用。
