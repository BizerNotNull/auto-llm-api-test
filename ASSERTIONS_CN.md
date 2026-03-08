# 特殊断言参考

本文档列出了各协议中实现的所有**特殊断言**（`assert_option_response`）。这些断言在通用响应结构检查之后运行，用于验证特定参数是否真正生效。

未列出的参数仅运行通用断言（`assert_non_stream_response` / `assert_stream_events`），只验证响应结构正确，不确认参数是否产生了实际效果。

---

## OpenAI Chat Completions

源码：`src/protocols/openai.py`

| 参数 | 断言内容 |
|------|---------|
| `tools` / `tool_choice` / `parallel_tool_calls` | 当 `finish_reason` = `"tool_calls"` 时：`message.tool_calls` 必须存在；每项必须有 `type` = `"function"`、非空的 `function.name`、以及 `function.arguments` |
| `response_format` | `message.content` 必须是合法 JSON（当 `type` = `"json_object"` 时） |
| `logprobs` | `choices[0].logprobs` 不能为 null，且必须包含 `content` 字段 |
| `top_logprobs` | `logprobs.content[0].top_logprobs` 必须非空 |
| `n` | `choices` 数量必须 >= 请求的 `n` 值 |
| `seed` | 响应中必须包含 `system_fingerprint`（用于可复现性） |
| `messages_image_url` / `messages_image_base64` | `message.content` 非空且长度至少 2 个字符（图片描述） |
| `service_tier` | 响应中必须包含 `service_tier` 字段 |

### 无特殊断言（仅通用检查）

`temperature`、`top_p`、`frequency_penalty`、`presence_penalty`、`max_tokens`、`max_completion_tokens`、`stop`、`modalities`、`audio`、`reasoning_effort`、`prediction`、`web_search_options`、`stream_options`、`user`、`store`、`metadata`

---

## Anthropic Messages

源码：`src/protocols/anthropic.py`

| 参数 | 断言内容 |
|------|---------|
| `tools` / `tool_choice` | 当 `stop_reason` = `"tool_use"` 时：`content` 中必须包含 `tool_use` block；每个 `tool_use` block 必须有 `id`、`name`、`input` |
| `thinking` | `content` 中必须包含 `thinking` block，且 `thinking` 文本非空 |
| `system` | 响应中必须包含至少一个 `text` 类型的 content block，且文本非空 |
| `messages_image_base64` / `messages_image_url` | 必须有 `text` 类型的 content，总文本长度 >= 2 个字符 |
| `stop_sequences` | 当 `stop_reason` = `"stop_sequence"` 时：`stop_sequence` 字段不能为 null |

| `cache_control` | 双请求测试：第 1 次请求 `cache_creation_input_tokens > 0`（或 `cache_read_input_tokens > 0`）；第 2 次请求 `cache_read_input_tokens > 0`（缓存命中） |

### 无特殊断言（仅通用检查）

`temperature`、`top_p`、`top_k`、`metadata`、`messages_pdf_base64`、`service_tier`

---

## Vertex AI (Gemini)

源码：`src/protocols/vertex.py`

| 参数 | 断言内容 |
|------|---------|
| `tools` / `toolConfig` | `parts` 中必须包含 `functionCall` 或 `text`；若有 `functionCall`，每项必须有 `name` 和 `args` |
| `responseMimeType` / `responseSchema` | 响应文本必须是合法 JSON |
| `contents_image_base64` / `contents_image_url` | 必须有文本响应且长度 >= 2 个字符 |
| `seed` | 响应中必须包含 `modelVersion` |
| `responseLogprobs` / `logprobs` | candidate 中必须包含 `avgLogprobs` 或 `logprobsResult` |

### 无特殊断言（仅通用检查）

`systemInstruction`、`generationConfig`、`temperature`、`topP`、`topK`、`presencePenalty`、`frequencyPenalty`、`maxOutputTokens`、`stopSequences`、`candidateCount`、`safetySettings`、`cachedContent`

---

## OpenAI Response API

源码：`src/protocols/response.py`

| 参数 | 断言内容 |
|------|---------|
| `tools` / `tool_choice` / `parallel_tool_calls` | `output` 中必须包含 `function_call` 或 `message`；若有 `function_call`，必须有 `name` 和 `arguments` |
| `text` | message content 中必须包含 `output_text` 项 |
| `input_image_url` / `input_image_base64` | 必须有描述图片的文本响应 |
| `reasoning` | `output` 中必须包含 `type` = `"reasoning"` 的项 |
| `service_tier` | 响应中必须包含 `service_tier` 字段 |

### 无特殊断言（仅通用检查）

`instructions`、`temperature`、`top_p`、`max_output_tokens`、`previous_response_id`、`include`、`truncation`、`user`、`store`、`metadata`
