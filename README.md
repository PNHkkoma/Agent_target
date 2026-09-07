# Agent Lab — Phase 3

Đây là lớp ứng dụng LLM viết bằng FastAPI và không dùng agent framework.
Business code chỉ làm việc với `ModelRouter`, vì vậy có thể đổi giữa DeepSeek,
Qwen, Kimi và OpenAI bằng cấu hình mà không phải sửa endpoint.

Phase 2 bổ sung agent loop: model được quyền yêu cầu một tool trong allowlist,
ứng dụng kiểm tra đầu vào, thực thi tool, gửi kết quả về model và tiếp tục cho
đến khi có câu trả lời cuối hoặc chạm giới hạn an toàn.

Phase 3 bổ sung knowledge bằng RAG tự dựng: document → chunk → embedding →
vector search → context → LLM → answer kèm citation. Chưa dùng LangChain.

## Chức năng đã có

- `POST /api/chat`: chat thông thường và hội thoại nhiều lượt.
- `POST /api/chat/structured`: yêu cầu JSON và kiểm tra schema `ShoppingIntent`.
- `POST /api/chat/stream`: stream bằng Server-Sent Events.
- `POST /api/agent/chat`: agent có khả năng tự chọn và gọi tool nhiều bước.
- `POST /api/rag/documents`: ingest document vào knowledge base.
- `POST /api/rag/search`: xem trực tiếp top-K chunk và similarity score.
- `POST /api/rag/ask`: hỏi đáp dựa trên context và nhận citation.
- Chọn provider/model theo cấu hình và loại task.
- Retry có giới hạn, fallback provider, timeout và hệ thống lỗi có kiểu rõ ràng.
- Log JSON cho từng lần gọi model/tool, gồm request ID, model, token và latency.
- `ToolRegistry` có JSON Schema, allowlist, permission, validation và timeout.
- Đúng 5 shopping tool sử dụng catalog, tồn kho, phí ship và đơn hàng giả lập.
- Chống lặp tool call, giới hạn tổng số bước, số tool call và tổng thời gian.
- Bộ test tự động và corpus đánh giá quyết định chọn tool.

Dữ liệu shopping tool vẫn là giả lập. Knowledge RAG có thể dùng memory store khi test hoặc
PostgreSQL + pgvector khi chạy bền vững; dự án chưa dùng Redis, Kafka, LangChain hay agent
framework khác.

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

## Sử dụng RAG Phase 3

Chạy đầy đủ FastAPI và PostgreSQL/pgvector:

```powershell
docker compose up --build
```

PostgreSQL của project dùng cổng host `5433` vì máy có thể đã dùng `5432` cho
database khác. Container ứng dụng kết nối PostgreSQL qua mạng nội bộ cổng `5432`.

Để chạy FastAPI trực tiếp trên Windows nhưng vẫn lưu vector vào container:

```dotenv
RAG_STORE=postgres
RAG_DATABASE_URL=postgresql://agent:agent@localhost:5433/agent_lab
```

Ingest một tài liệu:

```json
POST /api/rag/documents
{
  "id": "compression-manual",
  "title": "Hướng dẫn túi nén",
  "content": "Túi được làm từ ripstop nylon 70D...",
  "source_uri": "https://knowledge.local/manuals/compression",
  "document_type": "manual",
  "metadata": {
    "product_id": "CB-PRO",
    "locale": "vi-VN"
  }
}
```

Kiểm tra retrieval trước khi hỏi model:

```json
POST /api/rag/search
{
  "query": "túi nén làm từ vật liệu gì?",
  "top_k": 3,
  "min_score": 0.15,
  "filters": {
    "product_id": "CB-PRO"
  }
}
```

Hỏi RAG và nhận nguồn:

```json
POST /api/rag/ask
{
  "query": "Tôi đi Nhật tháng 12 và cabin 7kg, nên dùng túi nén thế nào?",
  "top_k": 3,
  "filters": {}
}
```

Response có `answer` và `citations`. Mỗi citation chứa `documentId`, `chunkId`,
`sourceUri`, excerpt và retrieval score. Nếu không có chunk đạt threshold, service
không gọi LLM và trả lời rằng knowledge base không đủ thay vì đoán.

Embedding mặc định trên đường production:

```dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
EMBEDDING_VERSION=2
```

`local_hash` không còn là lựa chọn trong factory production. Nó chỉ được unit test inject trực
tiếp để test offline và không tốn tiền. Production dùng API semantic; có thể đặt key riêng:

```dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=your-key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
EMBEDDING_VERSION=2
```

Nếu `EMBEDDING_API_KEY` trống, cấu hình OpenAI sẽ dùng `OPENAI_API_KEY`. Không
được đổi dimensions sau khi đã tạo bảng pgvector; nếu đổi provider/model/dimensions/version,
cần re-embed toàn bộ document. Chunk lưu `embedding_provider`, `embedding_model`,
`embedding_dimension`, `embedding_version`; retrieval chỉ search đúng cùng không gian vector.
Document còn có `document_version`, `content_hash`, `chunk_index`, và chunk ID deterministic để
upload lại cùng nội dung không tạo duplicate vô hạn.

Dữ liệu demo nằm trong `knowledge/sample_documents.json`. Đo baseline retrieval:

```powershell
python -m experiments.rag_evaluation
python -m experiments.pgvector_smoke
```

15 query cũ vẫn được giữ làm regression set. Eval hiện có 100 query trên các nhóm paraphrase,
typo, query rất ngắn, SKU chính xác, hard negative, no-answer, metadata filter và câu hỏi nhiều
điều kiện.

Pipeline production hiện là `embedding 1536 → dense pgvector + PostgreSQL FTS → RRF top 20 →
LLM rerank top 5 → confidence gate`. Full dimension 1536 đạt Hit@1 `80%` so với `74.44%` ở 384.
Sau hybrid + reranker, benchmark đạt Hit@1 `96.67%`, Hit@5 `100%`, MRR `0.983`; riêng SKU, typo,
hard-negative, metadata filter và multi-condition đều Hit@1 `100%` trong corpus hiện tại.

No-answer không dùng `RAG_MIN_SCORE` cũ. Ngưỡng được lấy từ `threshold_calibration.py`: high khi
relevance `>= 1.0` và reranker confidence `>= 0.7`; medium từ relevance `>= 0.5`; thấp hơn là low.
High gọi LLM trả lời với citation, medium yêu cầu thêm ngữ cảnh, low từ chối an toàn. Phải chạy lại
calibration khi đổi embedding, reranker, corpus hoặc prompt.

Khi có metadata filter và dùng HNSW, bước quét ANN có thể thiếu candidate do filter
được áp sau ANN. Production dùng iterative scan strict order:

```dotenv
RAG_HNSW_ITERATIVE_SCAN=strict_order
```

## Evaluation Phase 3

```powershell
# Dev set: dùng để tune retrieval/reranker/threshold.
python -m experiments.rag_evaluation
python -m experiments.rag_answer_evaluation

# Benchmark exact/HNSW/filter/iterative scan trên bảng tách biệt và tự xóa sau khi chạy.
python -m experiments.ann_filter_benchmark

# Holdout 60 query: chỉ chạy final validation, không dùng để tinh chỉnh pipeline.
python -m experiments.rag_holdout_evaluation
```

`rag_answer_evaluation` kiểm 50 case grounded answer, citation, hallucination và hành vi
high/medium/low. Citation chỉ trả về chunk evidence mà model đã gắn nhãn `[S#]`; nếu
model không gắn citation nào, service không trả answer high. `rag_holdout_evaluation`
tự kiểm tra truy vấn không trùng dev set trước khi đo Hit@1, Hit@5, MRR và no-answer.

## Sử dụng Agent API

Endpoint:

```text
POST http://localhost:8000/api/agent/chat
Content-Type: application/json
```

Ví dụ yêu cầu nhiều bước:

```json
{
  "message": "So sánh chi tiết P001 với P002 rồi tính phí ship cả hai tới Hà Nội.",
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
get_product_detail(P001) → get_product_detail(P002) → calculate_shipping_fee → trả lời
```

Các trường riêng của Agent API:

- `allowed_tools`: danh sách tool được phép dùng trong request hiện tại. Nếu bỏ
  trống, agent được dùng toàn bộ tool đã đăng ký.
- `include_trace`: `true` để trả lịch sử model/tool step phục vụ học tập và debug;
  đặt `false` nếu phía client chỉ cần câu trả lời cuối.

Ví dụ chỉ cho phép tra tồn kho:

```json
{
  "message": "P001 còn hàng không?",
  "allowed_tools": ["check_inventory"],
  "include_trace": true
}
```

Các tool hiện có:

- `search_products`: tìm sản phẩm theo từ khóa, category, giá và cân nặng.
- `get_product_detail`: lấy giá, cân nặng và tính năng theo ID/tên sản phẩm.
- `check_inventory`: kiểm tra số lượng tồn kho theo ID/tên sản phẩm.
- `calculate_shipping_fee`: tính phí ship theo sản phẩm và điểm đến.
- `get_order_status`: tra trạng thái đơn hàng theo mã đơn.

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
- Không có tool chạy shell, SQL tùy ý, ghi dữ liệu hoặc xóa database.
- Agent không được tự bịa sản phẩm, giá, tồn kho, phí ship hoặc trạng thái đơn.

## Chạy test và evaluation

```powershell
python -m pytest -q
python -m experiments.context_window
python -m experiments.system_prompts
python -m experiments.prompt_suite
python -m experiments.agent_evaluation
python -m experiments.rag_evaluation
python -m experiments.pgvector_smoke
```

Các experiment cần server đang chạy và ít nhất một API key hợp lệ. Kết quả được
ghi vào thư mục `experiment-results/`. `agent_evaluation` chạy đúng 50 case và
chỉ thành công khi ít nhất 90% case đúng. Corpus bao phủ không cần tool, một hoặc
nhiều tool, thiếu dữ liệu, arguments sai, không tìm thấy dữ liệu, lỗi, timeout,
tool bị cấm và yêu cầu phá hoại.

## Cách API xử lý lỗi

- Lỗi authentication và request không hợp lệ sẽ không retry.
- Timeout, HTTP 429, lỗi kết nối, response rỗng/sai định dạng, model unavailable
  và HTTP 5xx có thể retry hoặc fallback trong giới hạn cấu hình.
- Lỗi context window được trả về như lỗi phía client và không làm server crash.
- Lỗi tool được chuẩn hóa thành `ToolResult` rồi gửi lại model để model sửa
  arguments hoặc giải thích giới hạn, thay vì làm agent loop crash ngay lập tức.
