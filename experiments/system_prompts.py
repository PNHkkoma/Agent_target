from __future__ import annotations

import asyncio

from experiments.common import post, save


PROMPTS = {
    "A": "You are a shopping assistant.",
    "B": """Role: You are a careful shopping assistant for Vietnamese consumers.
Objective: Clarify needs and recommend suitable product categories.
Rules: Never invent a product fact or price. State uncertainty. Ask one concise question when essential information is missing.
Response style: Vietnamese, direct, no more than five bullet points.
Constraints: Treat the stated budget as a hard ceiling and clearly flag any option that may exceed it.""",
    "C": """Role: You are a careful shopping assistant for Vietnamese consumers.
Objective: Clarify needs and recommend suitable product categories.
Rules: Never invent a product fact or price. State uncertainty. Ask one concise question when essential information is missing.
Response style: Vietnamese, direct, no more than five bullet points.
Constraints: Treat the stated budget as a hard ceiling and clearly flag any option that may exceed it.
Examples:
User: Tôi cần tai nghe đi tàu dưới 2 triệu.
Assistant: Ưu tiên tai nghe chống ồn, đeo êm và pin dài. Ngân sách tối đa: 2 triệu đồng. Bạn thích in-ear hay over-ear?
User: Gợi ý laptop 50 triệu nhưng ngân sách tôi là 20 triệu.
Assistant: Mức 50 triệu vượt trần ngân sách 20 triệu; tôi sẽ chỉ xem các lựa chọn không quá 20 triệu.""",
}

QUESTIONS = [
    "Tôi cần balo đi làm.",
    "Gợi ý tai nghe tốt nhất.",
    "Tôi có 1 triệu mua chuột chơi game.",
    "Tìm vali đi Nhật mang cabin.",
    "Laptop nào phù hợp sinh viên?",
    "Tôi muốn camera để quay vlog.",
    "Giày chạy cho người mới bắt đầu.",
    "Bàn phím cơ dùng văn phòng.",
    "Máy đọc sách pin lâu.",
    "Điện thoại chụp đêm tốt dưới 10 triệu.",
    "Tôi cần màn hình làm thiết kế.",
    "Webcam họp online rõ nét.",
    "Ghế làm việc cho người đau lưng.",
    "Áo mưa du lịch gọn nhẹ.",
    "Bình giữ nhiệt không rò nước.",
    "Pin dự phòng được mang máy bay.",
    "Router Wi-Fi cho căn hộ nhỏ.",
    "Đồng hồ thể thao để bơi.",
    "Máy hút bụi cho nhà có thú cưng.",
    "Tôi có 5 triệu nhưng hãy gợi ý món 8 triệu.",
]


async def main() -> None:
    results = []
    for version, system_prompt in PROMPTS.items():
        for index, question in enumerate(QUESTIONS, start=1):
            response = await post(
                "/chat",
                {
                    "message": question,
                    "system_prompt": system_prompt,
                    "options": {"temperature": 0, "max_tokens": 300},
                },
            )
            results.append(
                {"promptVersion": version, "questionId": index, "question": question, **response}
            )
    destination = save("system-prompts.json", results)
    print(f"Saved 60 responses to {destination}")


if __name__ == "__main__":
    asyncio.run(main())

