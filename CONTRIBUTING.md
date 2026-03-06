# Contributing Guide: Adding Test Cases and Assertions

This document is for developers who need to extend the project's testing capabilities, covering three typical operations:

1. Adding new request parameter tests to existing protocols
2. Adding new response assertion rules
3. Adding an entirely new protocol

---

## Project Structure Overview

```
protocols/
  {name}_request.jsonc   ← Request body field toggles (jsonc, which fields are tested)
  {name}_response.json   ← Response body template (reference only, assertions are in code)
src/
  config.py              ← Config loading + required field mappings
  client.py              ← httpx async client (URL building, header building)
  protocols/
    base.py              ← ProtocolBuilder abstract base class
    openai.py            ← Each protocol's Builder implementation
    anthropic.py
    vertex.py
    response.py
  middleware.py           ← Retry + AI validation
  logger.py              ← curl/response logging
conftest.py              ← pytest fixtures + BUILDERS registry
tests/
  test_level1.py          ← L1 basic connectivity (auto-iterates all protocols × models)
  test_level2.py          ← L2 per-optional-parameter (auto-reads jsonc toggles)
  test_level3.py          ← L3 orthogonal combinations (auto-generates pairwise combos)
```

**Core mechanism**: L2 and L3 test cases are **auto-generated** — they read non-required fields set to `true` from `protocols/{name}_request.jsonc`, then call the Builder's `build_with_option(model, prompt, option_name)` to construct the request body. You only need to register in two places to have a new parameter automatically participate in testing.

---

## I. Adding a New Request Parameter to an Existing Protocol

Using adding `logit_bias` to OpenAI as an example, here are the complete steps.

### Step 1: Register the field toggle in jsonc

Edit `protocols/openai_request.jsonc` and add a line:

```jsonc
  "logit_bias": true,          // Token bias (token_id -> bias, -100~100)
```

- `true` means the field participates in L2/L3 testing
- `false` means skip for now (but still write the build logic in the Builder for future use)
- The `// [required]` comment is only for marking required fields; regular optional fields don't need it

### Step 2: Add build logic in the Builder's `build_with_option`

Edit `src/protocols/openai.py` and add to the `option_values` dict in `OpenAIBuilder.build_with_option`:

```python
option_values = {
    # ... existing entries ...
    "logit_bias": {"logit_bias": {
        "15339": -100,   # token id for "hello" → forbid
        "1820": 5,       # token id for "hi" → slight preference
    }},
}
```

**Key rules**:

- The dict key (`"logit_bias"`) must **exactly match** the field name in jsonc
- The value is a dict that gets merged into the request body via `body.update(value)`
- So the value's keys are the actual field names sent to the API
- If the parameter depends on another parameter (e.g., `tool_choice` depends on `tools`), include the dependency in the value:

```python
# tool_choice depends on tools, so the value must also include tools
"tool_choice": {"tools": [_TOOL_DEF], "tool_choice": "auto"},
```

- If the parameter needs to override `messages` (e.g., multimodal image input), include the full `"messages": [...]` in the value:

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

- If a parameter is meaningless in non-streaming mode (e.g., `stream_options`), set the value to an empty dict `{}`

### Step 3 (Recommended): Add a parameter-specific assertion

Edit `src/protocols/openai.py` and add an `elif` branch in the `assert_option_response` method:

```python
def assert_option_response(self, option_name: str, data: dict) -> list[str]:
    errors = []
    # ...existing if/elif...

    elif option_name == "logit_bias":
        content = message.get("content", "")
        if not content:
            errors.append("[logit_bias] Response content is empty")

    return errors
```

This step can be deferred — parameters without specific assertions only run the generic assertion and won't cause errors. But do add them eventually, otherwise a "passing" test only means the server didn't return 400, not that the parameter actually took effect.

### Step 4 (Optional): Handle mutual exclusions

If the new parameter is mutually exclusive with existing parameters, edit `MUTEX_GROUPS` in `tests/test_level3.py`:

```python
MUTEX_GROUPS = {
    "openai": [
        # ...existing...
        {"logit_bias", "response_format"},  # assuming these are mutually exclusive
    ],
}
```

Mutual exclusion means parameters within a set **will never appear together** in the same L3 combination.

If there are **dependency relationships** (e.g., `top_logprobs` requires `logprobs`), add a check in `_is_valid_combination`:

```python
if protocol_name == "openai":
    if "top_logprobs" in enabled and "logprobs" not in enabled:
        return False
```

### Done

No changes needed to `test_level2.py` or `test_level3.py` — they automatically read `true` fields from jsonc and generate tests, automatically calling generic + specific assertions. Run `pytest --collect-only -q` to verify the new tests are collected.

---

## II. Adding New Response Assertions

### Two-Layer Assertion Structure

After each test request returns, the framework executes **two assertion layers**:

```
1. Generic assertion  assert_non_stream_response(data)         ← runs for all requests
                              +
2. Specific assertion assert_option_response(option_name, data) ← dispatched by parameter name
```

- **Generic assertion**: Checks basic response structure integrity (has id, has choices, has content, etc.)
- **Specific assertion**: Checks **whether this parameter's effect is reflected in the response** (tools request returned tool_calls, logprobs request returned logprobs data, response_format=json returned valid JSON, etc.)

Both L2 (per-parameter tests) and L3 (combination tests) automatically call both layers. In L3, each **enabled parameter** in a combination triggers its specific assertion individually.

### Assertion Method Reference

| Method | Purpose | When to Modify |
|--------|---------|----------------|
| `assert_non_stream_response(data)` | Generic structure assertion for all non-streaming responses | When protocol response format changes |
| `assert_stream_events(events)` | Generic event assertion for all streaming responses | When protocol streaming format changes |
| `assert_option_response(option_name, data)` | **Parameter-specific assertion** | When adding/improving parameter behavior verification |
| `extract_usage(data)` | Extract usage from non-streaming response | When usage structure changes |
| `extract_stream_usage(events)` | Extract usage from streaming events | When usage structure changes |

### Return Value Convention

All assertion methods return `list[str]` — a **list of failure reasons**. An empty list means pass.

```python
def assert_non_stream_response(self, data: dict) -> list[str]:
    errors = []
    if "id" not in data:
        errors.append("Missing 'id' in response")
    return errors
```

The test framework uses them like this (L2 example):

```python
# Generic assertion
errors = builder.assert_non_stream_response(data)
# Specific assertion (appended to same errors list)
errors += builder.assert_option_response(option_name, data)

if errors:
    log_failure(test_name, ..., reason="; ".join(errors))
    raise RequestFailed(status, text, "; ".join(errors))
```

### Adding Parameter-Specific Assertions

Edit `assert_option_response` in `src/protocols/{name}.py`. Using OpenAI as an example, existing specific assertions:

```python
def assert_option_response(self, option_name: str, data: dict) -> list[str]:
    errors = []
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {})

    if option_name in ("tools", "tool_choice", "parallel_tool_calls"):
        # When finish_reason=tool_calls, must have tool_calls
        if choice.get("finish_reason") == "tool_calls":
            tc = message.get("tool_calls")
            if not tc:
                errors.append(f"[{option_name}] finish_reason='tool_calls' but no tool_calls")
            # ... check each tool_call's name/arguments ...

    elif option_name == "response_format":
        # Content must be valid JSON
        content = message.get("content", "")
        if content:
            try:
                json5.loads(content)
            except Exception:
                errors.append(f"[response_format] content is not valid JSON")

    elif option_name == "logprobs":
        # Response must have logprobs data
        lp = choice.get("logprobs")
        if lp is None:
            errors.append("[logprobs] logprobs is null in response")

    # ... more parameters ...
    return errors
```

**Adding a new parameter's assertion** just requires a new `elif` branch in `assert_option_response`:

```python
    elif option_name == "logit_bias":
        # logit_bias doesn't change response structure, but content should be normal
        content = message.get("content", "")
        if not content:
            errors.append("[logit_bias] Response content is empty")
```

Parameters without a registered specific assertion (no matching `elif` branch) return an empty list from `assert_option_response`, running only the generic assertion. This allows **incremental improvement** — get the parameter running first, add assertions later.

### Existing Specific Assertion Coverage by Protocol

**OpenAI**: tools/tool_choice/parallel_tool_calls (tool_calls structure), response_format (JSON validity), logprobs/top_logprobs (logprobs data), n (choices count), seed (system_fingerprint), messages_image_* (description text), service_tier

**Anthropic**: tools/tool_choice (tool_use block), thinking (thinking block), system (text output), messages_image_* (description text), stop_sequences (stop_sequence field)

**Vertex**: tools/toolConfig (functionCall), responseMimeType/responseSchema (JSON validity), contents_image_* (description text), seed (modelVersion), responseLogprobs/logprobs (logprobs data)

**Response API**: tools/tool_choice/parallel_tool_calls (function_call output), text (output_text), input_image_* (description text), reasoning (reasoning output), service_tier

### Modifying Generic Assertions

To change the assertion behavior for **all requests** (e.g., add a `system_fingerprint` check), edit `assert_non_stream_response`:

```python
def assert_non_stream_response(self, data: dict) -> list[str]:
    errors = []
    # ...existing checks...

    # New
    if "system_fingerprint" not in data:
        errors.append("Missing 'system_fingerprint' in response")

    return errors
```

Same approach for streaming assertions — edit `assert_stream_events`.

---

## III. Adding an Entirely New Protocol

Using adding `cohere` as an example.

### Step 1: Create request body and response body configs

Create `protocols/cohere_request.jsonc`:

```jsonc
{
  // Cohere Chat request body config
  "model": true,           // [required]
  "message": true,         // [required]

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

Create `protocols/cohere_response.json` (reference template):

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

### Step 2: Implement the Builder

Create `src/protocols/cohere.py`:

```python
"""Cohere Chat protocol implementation"""
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

### Step 3: Register the Builder + Client URL

Register in the `BUILDERS` dict in `conftest.py`:

```python
from src.protocols.cohere import CohereBuilder

BUILDERS = {
    # ...existing...
    "cohere": CohereBuilder(),
}
```

Add URL rules in `src/client.py`'s `_build_url`:

```python
elif self.protocol.name == "cohere":
    return f"{base}/v2/chat"
```

Add auth method in `src/client.py`'s `_build_headers` (if not standard Bearer token):

```python
elif self.protocol.name == "cohere":
    headers["Authorization"] = f"Bearer {self.protocol.api_key}"
```

### Step 4: Add required field mappings

Edit the `get_required_fields` function in `src/config.py`:

```python
REQUIRED = {
    # ...existing...
    "cohere": ["model", "message"],
}
```

### Step 5: Add protocol config in config.yaml

```yaml
protocols:
  # ...existing...
  cohere:
    base_url: "https://api.cohere.com"
    api_key: "xxx"
    models:
      thinking: []
      non_thinking:
        - "command-r-plus"
```

### Step 6 (Optional): Add mutex groups

Edit `MUTEX_GROUPS` in `tests/test_level3.py`:

```python
MUTEX_GROUPS = {
    # ...existing...
    "cohere": [
        {"temperature", "p"},   # not recommended together
    ],
}
```

### Step 7 (Optional): Add protocol-specific tests

If the protocol has special features (like Anthropic's caching/beta headers), add dedicated test functions in `tests/test_level2.py`. Follow the pattern of `test_anthropic_prompt_caching`.

### Done

Run to verify:

```bash
python -m pytest tests/ --collect-only -q
```

---

## IV. Complete Checklists

### Adding a Request Parameter (checklist)

- [ ] `protocols/{name}_request.jsonc` — Add field, set to `true`
- [ ] `src/protocols/{name}.py` — Add build logic in `build_with_option`'s `option_values`
- [ ] `src/protocols/{name}.py` — Add specific assertion in `assert_option_response` (can defer)
- [ ] `tests/test_level3.py` — Update `MUTEX_GROUPS` if mutually exclusive
- [ ] `protocols/{name}_response.json` — Update template if response structure changed (reference only)
- [ ] `pytest --collect-only -q` — Verify new tests are collected

### Adding a Response Assertion (checklist)

- [ ] `src/protocols/{name}.py` — Parameter-level assertions go in `assert_option_response`; global assertions go in `assert_non_stream_response` / `assert_stream_events`

### Adding a New Protocol (checklist)

- [ ] `protocols/{name}_request.jsonc` — Create request body toggles
- [ ] `protocols/{name}_response.json` — Create response template
- [ ] `src/protocols/{name}.py` — Implement `ProtocolBuilder` subclass (8 methods + `assert_option_response`)
- [ ] `conftest.py` — Register in `BUILDERS`
- [ ] `src/client.py` — Add branches in `_build_url` and `_build_headers`
- [ ] `src/config.py` — Add required fields in `get_required_fields`
- [ ] `config.yaml` — Add protocol's base_url, api_key, models
- [ ] `tests/test_level3.py` — Add `MUTEX_GROUPS` as needed
- [ ] `pytest --collect-only -q` — Verify

---

## V. Key Conventions

### option_values Key Naming

jsonc field name = `build_with_option`'s `option_name` parameter = `option_values` dict key. All three must be identical.

For parameters that override top-level fields (like multimodal), use the `{top_level_field}_{subtype}` naming convention:

```
messages_image_url       ← OpenAI: pass image URL via messages
messages_image_base64    ← OpenAI: pass image base64 via messages
contents_image_base64    ← Vertex: pass image via contents
input_image_url          ← Response API: pass image via input
```

### Assertions Must Not Raise Exceptions

Assertion methods (`assert_*`) only collect error strings and return a list — never `raise` or `assert` inside them. Exceptions are handled uniformly by the test functions.

### Vertex's generationConfig Merging

Multiple Vertex parameters (temperature, topP, maxOutputTokens, etc.) are nested inside `generationConfig`. In `build_with_option`, each parameter's value is written as:

```python
"temperature": {
    "generationConfig": {"temperature": 0.7}
},
```

Both the Builder and L3's `_build_combo_body` automatically merge `generationConfig` (via dict.update) without overwriting each other.

### Empty Dict Means "Don't Construct Request Body"

```python
"stream_options": {},       # Meaningless in non-streaming mode, adds nothing
"cachedContent": {},        # Requires external resources, skip for now
```

An empty dict won't modify the request body, but the parameter still appears in jsonc (set to `false` to exclude from testing).

### Test Image

Multimodal tests use `prompts/images/test.png` (1x1 red PNG). For more complex image testing, add files to that directory and reference them in the Builder.
