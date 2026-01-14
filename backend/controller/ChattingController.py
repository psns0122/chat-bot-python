from fastapi import APIRouter
from backend.model.ChatModel import ChatModel
from backend.service.ChattingService import ChattingService, DEFAULT_URLS

router = APIRouter()
service = ChattingService()

@router.post("/api/ingest")
async def ingest():
    await service.ingest_from_urls(DEFAULT_URLS)
    return "데이터 적재 성공!"

@router.post("/api/get-gemini")
async def get_gemini(req: ChatModel):
    text = await service.ask(req.sessionId, req.message)

    # 너가 현재 프론트가 기대하는 응답 형태를 그대로 유지
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": text}
                    ]
                }
            }
        ]
    }