# LLM API Automated Testing

Pure HTTP-layer automated testing for LLM APIs — multi-protocol, multi-level, full-parameter coverage with zero SDK dependencies.

## Supported Protocols

| Protocol | Endpoint | Description |
|----------|----------|-------------|
| **OpenAI** | `/v1/chat/completions` | Chat Completions API |
| **Anthropic** | `/v1/messages` | Messages API |
| **Vertex AI** | `/{model}:generateContent` | Gemini API |
| **Response** | `/v1/responses` | OpenAI Response API |

Each protocol supports multiple models, distinguishing between thinking and non-thinking models.

## Tech Stack

| Component | Choice | Description |
|-----------|--------|-------------|
| HTTP Client | httpx | AsyncClient + asyncio concurrency |
| Test Engine | pytest + pytest-asyncio | Async tests, parametrize, auto-collection |
| JSON Parser | json5 | JSONC comment syntax support |
| Retry | tenacity | Exponential backoff retry |
| Console | rich | Progress bars + colored output |
| Combinatorial | allpairspy | Pairwise (orthogonal array) testing |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `config.yaml` with your API endpoints and keys:

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
    auth_header: "x-api-key"    # or "authorization"
    models:
      thinking:
        - "claude-sonnet-4-20250514"
      non_thinking:
        - "claude-sonnet-4-20250514"
```

Leave `models` empty for any protocol you don't need to test.

### 3. Run

```bash
# Run via entry script (with Rich panel)
python run.py

# Or run directly with pytest
python -m pytest tests/ -v

# Test a specific protocol
python run.py --protocol openai

# Test a specific level
python -m pytest tests/test_level1.py -v
```

## Test Levels

Control which levels run via `test_levels` in `config.yaml`:

```yaml
test_levels:
  level1: true    # Basic connectivity
  level2: false   # Feature availability
  level3: false   # Combination coverage
```

### L1 — Basic Connectivity

Iterates over every model (thinking/non-thinking) of each protocol, running three basic tests:

| Test | Description |
|------|-------------|
| `test_connectivity_minimal` | Send only required fields, verify connectivity |
| `test_non_stream_usage` | Non-streaming request, verify response structure and usage |
| `test_stream_events_and_usage` | Streaming request, verify SSE event completeness and usage |

### L2 — Feature Availability

Automatically reads each non-required field set to `true` in `protocols/{name}_request.jsonc` and sends individual requests. Each parameter goes through two assertion layers:

```
Generic assertion: Is the response structure valid?          ← runs for all parameters
              +
Specific assertion: Is this parameter's effect reflected?    ← runs if registered
```

For example, the `tools` parameter not only checks for HTTP 200 but also verifies that `tool_calls` are returned; `response_format: json_object` checks that the content is valid JSON.

Also includes Anthropic-specific tests:
- **Prompt caching test** — Sends sufficient tokens + random prefix, verifies `cache_creation_input_tokens`
- **Beta header test** — Tests each `anthropic-beta` header individually

### L3 — Combination Coverage

Uses [allpairspy](https://github.com/thombashi/allpairspy) pairwise testing to generate orthogonal combinations of all optional parameters. Automatically handles parameter mutual exclusions (e.g., `temperature` vs `top_p`, `max_tokens` vs `max_completion_tokens`). Each enabled parameter in a combination triggers its specific assertion.

## Project Structure

```
auto-llm-api-test/
├── config.yaml                     # Main configuration
├── protocols/                      # Protocol configs
│   ├── openai_request.jsonc        #   Request body field toggles (true=test)
│   ├── openai_response.json        #   Response body template (reference)
│   ├── anthropic_request.jsonc
│   ├── anthropic_response.json
│   ├── vertex_request.jsonc
│   ├── vertex_response.json
│   ├── response_request.jsonc
│   └── response_response.json
├── prompts/                        # Prompts
│   ├── short.txt                   #   Short prompt (connectivity/feature tests)
│   ├── long.txt                    #   Long prompt (cache tests)
│   └── images/
│       └── test.png                #   Test image (multimodal)
├── src/
│   ├── config.py                   # Config loader
│   ├── client.py                   # httpx async client
│   ├── middleware.py               # Retry + AI validation
│   ├── logger.py                   # curl/response logging
│   ├── console.py                  # Rich console
│   └── protocols/
│       ├── base.py                 #   ProtocolBuilder base class
│       ├── openai.py               #   Request building + response assertions
│       ├── anthropic.py
│       ├── vertex.py
│       └── response.py
├── tests/
│   ├── test_level1.py              # L1 basic connectivity
│   ├── test_level2.py              # L2 per-parameter availability
│   └── test_level3.py              # L3 orthogonal combinations
├── logs/                           # Generated at runtime
│   ├── success.log                 #   Successful request curl + response
│   └── failure.log                 #   Failed request curl + response + reason
├── conftest.py                     # pytest fixtures
├── pytest.ini
├── run.py                          # Entry script
└── requirements.txt
```

## Request Body Configuration

Testable fields for each protocol are declared with boolean values in `protocols/{name}_request.jsonc`:

```jsonc
{
  "model": true,              // [required] Model name
  "messages": true,           // [required] Message list

  "temperature": true,        // Test this
  "top_p": true,              // Test this
  "logprobs": false,          // Skip for now
  "seed": true,               // Test this
  "tools": true,              // Test this
  "response_format": true,    // Test this
  // ...
}
```

Setting a field to `true` automatically includes it in L2 and L3 tests — no test code changes needed.

## Assertion Mechanism

Two assertion layers execute after each API request:

**Generic assertion** (`assert_non_stream_response`) — checks basic response structure:

- OpenAI: has `id`, `choices[0].message.content`, `finish_reason`
- Anthropic: has `id`, `type=message`, `content[0].type=text`, `stop_reason`
- Vertex: has `candidates[0].content.parts[0].text`
- Response: has `id`, `status=completed`, `output[0].content[0].output_text`

**Specific assertion** (`assert_option_response`) — checks whether the parameter's effect is reflected:

| Parameter | Assertion |
|-----------|-----------|
| `tools` / `tool_choice` | Check tool_calls structure when `finish_reason=tool_calls` |
| `response_format` | Content is valid JSON |
| `logprobs` | Response contains logprobs data |
| `thinking` | Content includes a thinking block |
| `messages_image_*` | Response has sufficiently long text description |
| `responseMimeType` | Response text is valid JSON |
| `seed` | Response contains system_fingerprint / modelVersion |

Parameters without registered specific assertions only run the generic assertion.

## Post-Request Middleware

### Retry

Failed requests are automatically retried with tenacity exponential backoff:

```yaml
retry:
  enabled: true
  max_attempts: 3
  multiplier: 1        # wait = multiplier * 2^attempt
  max_wait: 30         # Maximum wait in seconds
```

### AI Validation

Requests that still fail after all retries can be sent to an AI to determine if the failure is **expected behavior** (e.g., an unsupported parameter being correctly rejected). Tests deemed expected are marked as UNSTABLE (yellow), separate from PASS and FAIL:

```yaml
ai_validation:
  enabled: false
  base_url: "https://api.openai.com/v1"
  api_key: "sk-xxx"
  model: "gpt-4o"
```

## Logging

Each run automatically logs to the `logs/` directory:

- `success.log` — curl commands + response bodies for all successful requests
- `failure.log` — curl commands + response bodies + failure reasons for all failed requests

API keys in curl commands are automatically redacted.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Adding new request parameters to existing protocols
- Adding parameter-specific assertions
- Adding entirely new protocols
