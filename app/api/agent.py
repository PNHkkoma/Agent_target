from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.agent.runner import AgentRunner
from app.schemas.agent import AgentRequest, AgentResponse

router = APIRouter(prefix="/api/agent", tags=["agent"])


# Nhận FastAPI request; trả AgentRunner dùng chung đã được khởi tạo trong application state.
def get_agent_runner(request: Request) -> AgentRunner:
    return request.app.state.agent_runner


AgentDependency = Annotated[AgentRunner, Depends(get_agent_runner)]


# Nhận body AgentRequest, request context và runner; trả câu trả lời cùng trace agent.
@router.post("/chat", response_model=AgentResponse, response_model_by_alias=True)
async def agent_chat(
    payload: AgentRequest,
    request: Request,
    agent_runner: AgentDependency,
) -> AgentResponse:
    return await agent_runner.run(payload, request_id=request.state.request_id)
