# Agent Lab — Phase 2

Đây là lớp ứng dụng LLM viết bằng FastAPI và không dùng agent framework.
Business code chỉ làm việc với `ModelRouter`, vì vậy có thể đổi giữa DeepSeek,
Qwen, Kimi và OpenAI bằng cấu hình mà không phải sửa endpoint.

Phase 2 bổ sung agent loop: model được quyền yêu cầu một tool trong allowlist,
ứng dụng kiểm tra đầu vào, thực thi tool, gửi kết quả về model và tiếp tục cho
đến khi có câu trả lời cuối hoặc chạm giới hạn an toàn.

## Chức năng đã có

- `POST /api/chat`: chat thông thường và hội thoại nhiều lượt.
- `POST /api/chat/structured`: yêu cầu JSON và kiểm tra schema `ShoppingIntent`.
- `POST /api/chat/stream`: stream bằng Server-Sent Events.
- `POST /api/agent/chat`: agent có khả năng tự chọn và gọi tool nhiều bước.
- Chọn provider/model theo cấu hình và loại task.
- Retry có giới hạn, fallback provider, timeout và hệ thống lỗi có kiểu rõ ràng.
- Log JSON cho từng lần gọi model/tool, gồm request ID, model, token và latency.
- `ToolRegistry` có JSON Schema, allowlist, permission, validation và timeout.
- Calculator an toàn, catalog sản phẩm giả lập và thời tiết giả lập.
- Chống lặp tool call, giới hạn tổng số bước, số tool call và tổng thời gian.
- Bộ test tự động và corpus đánh giá quyết định chọn tool.

Dự án hiện chưa dùng database, Redis, Kafka, vector store, LangChain hoặc agent
framework khác. Dữ liệu sản phẩm và thời tiết đều là dữ liệu giả lập trong code.

## Yêu cầu

- Python 3.11 trở lên.
- API key của ít nhất một provider.

## Cài đặt và chạy

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Sau khi server chạy:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Nếu đang dùng Command Prompt (`cmd`) thay vì PowerShell, kích hoạt môi trường bằng:

```bat
.venv\Scripts\activate.bat
```

## Cấu hình provider

Ví dụ dùng DeepSeek:

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-key
DEEPSEEK_MODEL=deepseek-chat
```

Ví dụ dùng OpenAI GPT-4.1 Mini snapshot:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4.1-mini-2025-04-14
```

Không đưa API key thật lên Git. Nếu key đã có trong biến môi trường Windows thì
không cần ghi lại vào `.env`. Provider không có key sẽ được bỏ qua an toàn.

Thứ tự fallback được cấu hình như sau:

```dotenv
LLM_FALLBACK_PROVIDERS=qwen,kimi,openai
```

## Sử dụng Agent API

Endpoint:

```text
POST http://localhost:8000/api/agent/chat
Content-Type: application/json
```

Ví dụ yêu cầu nhiều bước:

```json
{
  "message": "Cuối tuần Đà Lạt có mưa không? Tìm áo khoác dưới 1 triệu và chọn mẫu nhẹ nhất.",
  "history": [],
  "system_prompt": "Trả lời bằng tiếng Việt, ngắn gọn.",
  "task": "complex_reasoning",
  "options": {
    "temperature": 0,
    "max_tokens": 500,
    "response_format": "text"
  },
  "include_trace": true
}
```

Agent có thể tự thực hiện chuỗi hành động:

```text
get_weather → search_products → get_product → trả lời
```

Các trường riêng của Agent API:

- `allowed_tools`: danh sách tool được phép dùng trong request hiện tại. Nếu bỏ
  trống, agent được dùng toàn bộ tool đã đăng ký.
- `include_trace`: `true` để trả lịch sử model/tool step phục vụ học tập và debug;
  đặt `false` nếu phía client chỉ cần câu trả lời cuối.

Ví dụ chỉ cho phép calculator:

```json
{
  "message": "38 * 27 + 17 bằng bao nhiêu?",
  "allowed_tools": ["calculate"],
  "include_trace": true
}
```

Các tool hiện có:

- `calculate`: tính số học bằng AST allowlist, không dùng `eval`.
- `search_products`: tìm sản phẩm còn hàng theo category, giá và cân nặng.
- `get_product`: lấy thông tin đầy đủ theo ID sản phẩm.
- `get_weather`: lấy thời tiết giả lập theo thành phố.

Response của Agent API gồm:

- `content`: câu trả lời cuối dành cho người dùng.
- `provider`, `model`: provider và model tạo câu trả lời cuối.
- `totalSteps`: tổng số lượt gọi model.
- `totalToolCalls`: tổng số tool call model đã yêu cầu.
- `inputTokens`, `outputTokens`, `latencyMs`: số liệu sử dụng và thời gian.
- `trace`: từng model step và tool step; kết quả tool trong đây là dữ liệu trung gian.

## Sử dụng Chat API thông thường

```powershell
$body = @{
  message = "Tại sao trời có mưa?"
} | ConvertTo-Json

Invoke-RestMethod -Method Post http://localhost:8000/api/chat `
  -ContentType "application/json" `
  -Body $body
```

Ứng dụng không tự lưu lịch sử. Client phải gửi lại `history` ở mỗi request:

```json
{
  "message": "Tên tôi là gì?",
  "history": [
    {"role": "user", "content": "Tôi tên An."},
    {"role": "assistant", "content": "Chào An!"}
  ]
}
```

## Structured API

Endpoint này chỉ phù hợp với bài toán trích xuất ý định mua sắm theo schema
`ShoppingIntent`, không phải endpoint chat JSON tổng quát.

```powershell
$body = @{
  message = "Tôi cần balo đi Nhật khoảng 1 triệu, ưu tiên nhẹ và mang cabin."
} | ConvertTo-Json

Invoke-RestMethod -Method Post http://localhost:8000/api/chat/structured `
  -ContentType "application/json" `
  -Body $body
```

## Stream API

`POST /api/chat/stream` trả về các SSE event. Mỗi event `token` chỉ chứa một
phần nhỏ của câu trả lời; frontend nối trường `content` theo đúng thứ tự để tạo
hiệu ứng chữ xuất hiện dần.

```text
event: token
data: {"content":"Xin","provider":"deepseek","model":"deepseek-chat"}

event: usage
data: {"content":"","provider":"deepseek","model":"deepseek-chat","input_tokens":12,"output_tokens":5}

event: done
data: [DONE]
```

Nếu provider lỗi trước token đầu tiên, router có thể retry hoặc fallback. Sau
khi đã gửi token về client, stream trả event `error` thay vì ghép câu trả lời từ
provider khác vào giữa nội dung đang chạy.

## Giới hạn an toàn của agent

Có thể thay đổi trong `.env`:

```dotenv
AGENT_MAX_STEPS=8
AGENT_MAX_TOOL_CALLS=12
AGENT_MAX_DUPLICATE_CALLS=1
AGENT_TIMEOUT_SECONDS=60
TOOL_TIMEOUT_SECONDS=2
```

- Arguments sai schema không được chuyển đến handler.
- Tool ngoài allowlist không được thực thi.
- Tool không có quyền phù hợp trả lỗi `PERMISSION_DENIED`.
- Một tool call giống hệt không được thực thi lặp vô hạn.
- Calculator không cho chạy tên biến, function hoặc mã hệ thống.
- Agent không được tự bịa sản phẩm, giá, tồn kho hoặc thời tiết.

## Chạy test và evaluation

```powershell
python -m pytest -q
python -m experiments.context_window
python -m experiments.system_prompts
python -m experiments.prompt_suite
python -m experiments.agent_evaluation
```

Các experiment cần server đang chạy và ít nhất một API key hợp lệ. Kết quả được
ghi vào thư mục `experiment-results/`. `agent_evaluation` kiểm tra chính xác
chuỗi tool của sáu tình huống: không cần tool, calculator, tìm sản phẩm, lấy chi
tiết, so sánh sản phẩm và thời tiết kết hợp tư vấn mua sắm.

## Cách API xử lý lỗi

- Lỗi authentication và request không hợp lệ sẽ không retry.
- Timeout, HTTP 429, lỗi kết nối, response rỗng/sai định dạng, model unavailable
  và HTTP 5xx có thể retry hoặc fallback trong giới hạn cấu hình.
- Lỗi context window được trả về như lỗi phía client và không làm server crash.
- Lỗi tool được chuẩn hóa thành `ToolResult` rồi gửi lại model để model sửa
  arguments hoặc giải thích giới hạn, thay vì làm agent loop crash ngay lập tức.
