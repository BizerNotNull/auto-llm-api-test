# Option-Specific Assertions Reference

This document lists all **option-specific assertions** (`assert_option_response`) implemented in each protocol. These assertions run *after* the generic response structure check and validate that a specific parameter actually took effect.

Parameters not listed here only run the generic assertion (`assert_non_stream_response` / `assert_stream_events`), which verifies the response structure is correct but does not confirm the parameter had any effect.

---

## OpenAI Chat Completions

Source: `src/protocols/openai.py`

| Option | Assertion |
|--------|-----------|
| `tools` / `tool_choice` / `parallel_tool_calls` | When `finish_reason` = `"tool_calls"`: `message.tool_calls` must exist; each entry must have `type` = `"function"`, non-empty `function.name`, and `function.arguments` present |
| `response_format` | `message.content` must be valid JSON (when `type` = `"json_object"`) |
| `logprobs` | `choices[0].logprobs` must not be null and must contain a `content` field |
| `top_logprobs` | `logprobs.content[0].top_logprobs` must be non-empty |
| `n` | Number of `choices` must be >= the requested `n` value |
| `seed` | Response must contain `system_fingerprint` (needed for reproducibility) |
| `messages_image_url` / `messages_image_base64` | `message.content` must be non-empty and at least 2 characters (image description) |
| `service_tier` | Response must contain a `service_tier` field |

### No specific assertion (generic only)

`temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `max_tokens`, `max_completion_tokens`, `stop`, `modalities`, `audio`, `reasoning_effort`, `prediction`, `web_search_options`, `stream_options`, `user`, `store`, `metadata`

---

## Anthropic Messages

Source: `src/protocols/anthropic.py`

| Option | Assertion |
|--------|-----------|
| `tools` / `tool_choice` | When `stop_reason` = `"tool_use"`: `content` must contain a `tool_use` block; each `tool_use` block must have `id`, `name`, and `input` |
| `thinking` | `content` must contain a `thinking` block with non-empty `thinking` text |
| `system` | Response must contain at least one `text`-type content block with non-empty text |
| `messages_image_base64` / `messages_image_url` | Must have `text`-type content with total text length >= 2 characters |
| `stop_sequences` | When `stop_reason` = `"stop_sequence"`: `stop_sequence` field must not be null |

| `cache_control` | Two-request test: Request 1 must have `cache_creation_input_tokens > 0` (or `cache_read_input_tokens > 0`); Request 2 must have `cache_read_input_tokens > 0` (cache hit) |

### No specific assertion (generic only)

`temperature`, `top_p`, `top_k`, `metadata`, `messages_pdf_base64`, `service_tier`

---

## Vertex AI (Gemini)

Source: `src/protocols/vertex.py`

| Option | Assertion |
|--------|-----------|
| `tools` / `toolConfig` | `parts` must contain `functionCall` or `text`; if `functionCall` is present, each must have `name` and `args` |
| `responseMimeType` / `responseSchema` | Response text must be valid JSON |
| `contents_image_base64` / `contents_image_url` | Must have text response with length >= 2 characters |
| `seed` | Response must contain `modelVersion` |
| `responseLogprobs` / `logprobs` | Candidate must contain `avgLogprobs` or `logprobsResult` |

### No specific assertion (generic only)

`systemInstruction`, `generationConfig`, `temperature`, `topP`, `topK`, `presencePenalty`, `frequencyPenalty`, `maxOutputTokens`, `stopSequences`, `candidateCount`, `safetySettings`, `cachedContent`

---

## OpenAI Response API

Source: `src/protocols/response.py`

| Option | Assertion |
|--------|-----------|
| `tools` / `tool_choice` / `parallel_tool_calls` | `output` must contain `function_call` or `message`; if `function_call` is present, must have `name` and `arguments` |
| `text` | Message content must contain an `output_text` item |
| `input_image_url` / `input_image_base64` | Must have text response describing the image |
| `reasoning` | `output` must contain an item with `type` = `"reasoning"` |
| `service_tier` | Response must contain a `service_tier` field |

### No specific assertion (generic only)

`instructions`, `temperature`, `top_p`, `max_output_tokens`, `previous_response_id`, `include`, `truncation`, `user`, `store`, `metadata`
