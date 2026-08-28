# Agent Lab — Phase 1

A small, framework-free LLM application layer. FastAPI business code calls one
`ModelRouter`; DeepSeek, Qwen, and Kimi are interchangeable OpenAI-compatible
providers selected by configuration.

## What is implemented

- `POST /api/chat`: raw and multi-turn chat with a unified response.
- `POST /api/chat/structured`: JSON mode plus strict `ShoppingIntent` validation.
- `POST /api/chat/stream`: Server-Sent Events (`token`, `usage`, `done`, `error`).
- Config-based provider selection and V1 task routing.
- Bounded exponential retry, ordered provider fallback, timeout and typed errors.
- Per-attempt JSON logs with request/provider/model/token/latency/status fields.
- Repeatable context-window and system-prompt experiments.
- A 30-prompt automated evaluation corpus plus isolated unit/API tests.

There is intentionally no database, Redis, Kafka, vector store, LangChain, or
other agent framework.

## Run locally

Python 3.11+ is supported.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Put real keys in .env, then:
uvicorn app.main:app --reload
```

OpenAPI is at `http://localhost:8000/docs`.

Changing only this setting switches the default provider; endpoint/business
code stays unchanged:

```dotenv
LLM_PROVIDER=qwen
```

Provider keys are independent. A provider without a key is skipped safely.
Fallback order is configured with `LLM_FALLBACK_PROVIDERS=qwen,kimi`.

## Examples

Raw chat:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/api/chat `
  -ContentType application/json `
  -Body '{"message":"Tại sao trời có mưa?"}'
```

Multi-turn chat is stateless: the application sends the necessary history on
every request. This demonstrates that context is not durable memory.

```json
{
  "message": "Tên tôi là gì?",
  "history": [
    {"role": "user", "content": "Tôi tên An."},
    {"role": "assistant", "content": "Chào An!"}
  ]
}
```

Structured output:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/api/chat/structured `
  -ContentType application/json `
  -Body '{"message":"Tôi cần balo đi Nhật khoảng 1 triệu, ưu tiên nhẹ và mang cabin."}'
```

Streaming event shape:

```text
event: token
data: {"content":"Xin","provider":"deepseek","model":"deepseek-chat"}

event: usage
data: {"content":"","provider":"deepseek","model":"deepseek-chat","input_tokens":12,"output_tokens":5}

event: done
data: [DONE]
```

If a provider fails before the first token, the router can retry/fallback. Once
a token has reached the client, it emits `event: error` rather than splice a
second model's answer into the first.

## Tests and experiments

```powershell
pytest
python -m experiments.context_window
python -m experiments.system_prompts
python -m experiments.prompt_suite
```

The experiments require the API server and at least one real provider key. They
write ignored result files under `experiment-results/`. `prompt_suite` exits
non-zero when any raw response is empty or any structured response fails schema
validation.

## API error behavior

Authentication errors and invalid requests are not retried. Timeouts, 429,
connection resets, malformed/empty responses, model unavailability and 5xx are
contained and eligible for bounded retry/fallback. Context-window errors are
reported as a client error and never crash the application.

