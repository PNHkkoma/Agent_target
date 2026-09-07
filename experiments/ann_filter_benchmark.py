from __future__ import annotations

import math
import random
import statistics
import time
from typing import Any

from experiments.common import save


DATABASE_URL = "postgresql://agent:agent@localhost:5433/agent_lab"
TABLE = "rag_ann_filter_benchmark"
DIMENSIONS = 64
ROWS = 10_000
GROUPS = 100
QUERY_COUNT = 40
TOP_K = 5
# Chọn ef_search đủ lớn cho filter 5%; đây là cấu hình ứng viên dùng trong production,
# không phải chỉ số mặc định 40 vốn chỉ phù hợp khi không lọc mạnh.
HNSW_EF_SEARCH = 40
HNSW_MAX_SCAN_TUPLES = 20_000


# Nhận seed và số chiều; trả vector chuẩn hóa deterministic để benchmark không gọi embedding API.
def make_vector(seed: int) -> list[float]:
    generator = random.Random(seed)
    vector = [generator.uniform(-1, 1) for _ in range(DIMENSIONS)]
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


# Nhận latency; trả P50/P95 milliseconds để so sánh tốc độ giữa các chiến lược ANN.
def latency_summary(latencies: list[float]) -> dict[str, float]:
    ordered = sorted(latencies)
    return {
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[math.ceil(len(ordered) * 0.95) - 1], 3),
    }


# Nhận PostgreSQL connection; tạo dữ liệu benchmark, HNSW index và trả query vector/filter mẫu.
def prepare(connection: Any) -> list[tuple[list[float], int]]:
    from pgvector import Vector

    connection.execute(f"DROP TABLE IF EXISTS {TABLE}")
    connection.execute(f"CREATE TABLE {TABLE} (id INTEGER PRIMARY KEY, category INTEGER NOT NULL, embedding vector({DIMENSIONS}) NOT NULL)")
    rows = [(index, index % GROUPS, Vector(make_vector(index))) for index in range(ROWS)]
    with connection.cursor() as cursor:
        cursor.executemany(
            f"INSERT INTO {TABLE} (id, category, embedding) VALUES (%s, %s, %s)", rows
        )
    connection.execute(
        f"CREATE INDEX {TABLE}_hnsw_idx ON {TABLE} USING hnsw (embedding vector_cosine_ops)"
    )
    # Query không trùng vector đã index để tránh "self-match" làm benchmark quá dễ.
    return [
        (make_vector(100_000 + index * 37), index % GROUPS)
        for index in range(QUERY_COUNT)
    ]


# Nhận query, filter và chiến lược; trả ID result cùng latency của một lần query PostgreSQL.
def query_once(
    connection: Any,
    vector: list[float],
    category: int,
    *,
    mode: str,
) -> tuple[list[int], float]:
    from pgvector import Vector

    with connection.transaction():
        if mode.startswith("exact"):
            connection.execute("SET LOCAL enable_indexscan = off")
            connection.execute("SET LOCAL enable_bitmapscan = off")
        else:
            connection.execute("SET LOCAL enable_seqscan = off")
            connection.execute(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}")
            connection.execute(f"SET LOCAL hnsw.max_scan_tuples = {HNSW_MAX_SCAN_TUPLES}")
            connection.execute(
                "SET LOCAL hnsw.iterative_scan = "
                + ("strict_order" if mode == "hnsw_filter_iterative" else "off")
            )
        where = "WHERE category = %s" if "filter" in mode else ""
        started = time.perf_counter()
        rows = connection.execute(
            f"SELECT id FROM {TABLE} {where} ORDER BY embedding <=> %s LIMIT %s",
            ([category] if where else []) + [Vector(vector), TOP_K],
        ).fetchall()
        return [row[0] for row in rows], (time.perf_counter() - started) * 1000


# Nhận query set và mode; so exact reference để trả recall, latency và số result trung bình.
def benchmark_mode(
    connection: Any,
    queries: list[tuple[list[float], int]],
    exact_reference: list[list[int]],
    mode: str,
) -> dict[str, Any]:
    recalls: list[float] = []
    counts: list[int] = []
    latencies: list[float] = []
    for (vector, category), expected in zip(queries, exact_reference):
        actual, latency = query_once(connection, vector, category, mode=mode)
        recalls.append(len(set(actual) & set(expected)) / len(expected))
        counts.append(len(actual))
        latencies.append(latency)
    return {
        "mode": mode,
        "recall_at_k": round(statistics.mean(recalls), 4),
        "returned_result_count": round(statistics.mean(counts), 3),
        **latency_summary(latencies),
    }


# Không nhận đầu vào; chạy exact/HNSW/filter/iterative scan và luôn xóa bảng benchmark riêng.
def main() -> None:
    from pgvector.psycopg import register_vector
    from psycopg import connect

    with connect(DATABASE_URL) as connection:
        register_vector(connection)
        try:
            queries = prepare(connection)
            exact = [
                query_once(connection, vector, category, mode="exact")[0]
                for vector, category in queries
            ]
            exact_filtered = [
                query_once(connection, vector, category, mode="exact_filter")[0]
                for vector, category in queries
            ]
            report = {
                "rows": ROWS,
                "groups": GROUPS,
                "top_k": TOP_K,
                "filter_selectivity": 1 / GROUPS,
                "hnsw_ef_search": HNSW_EF_SEARCH,
                "hnsw_max_scan_tuples": HNSW_MAX_SCAN_TUPLES,
                "results": [
                    benchmark_mode(connection, queries, exact, "exact"),
                    benchmark_mode(connection, queries, exact, "hnsw"),
                    benchmark_mode(
                        connection, queries, exact_filtered, "hnsw_filter"
                    ),
                    benchmark_mode(
                        connection,
                        queries,
                        exact_filtered,
                        "hnsw_filter_iterative",
                    ),
                ],
            }
            destination = save("ann-filter-benchmark.json", report)
            print(f"ANN benchmark: {report['results']}; details={destination}")
        finally:
            connection.execute(f"DROP TABLE IF EXISTS {TABLE}")


if __name__ == "__main__":
    main()
