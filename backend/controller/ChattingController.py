from fastapi import APIRouter
from backend.model.ChatModel import ChatModel
from backend.service.ChattingService import ChattingService, FILE_DIR

router = APIRouter()
service = ChattingService()

@router.post("/api/ingest")
async def ingest():
    await service.ingest(FILE_DIR, force_rebuild=False)
    return "데이터 적재 성공!"

@router.post("/api/ingest/rebuild")
async def ingest_rebuild():
    await service.ingest(FILE_DIR, force_rebuild=True)
    return "데이터 재생성 성공!"

@router.post("/api/get-gemini")
async def get_gemini(req: ChatModel):
    result = await service.ask_with_sources(req.sessionId, req.message)
    return {
        "candidates": [
            {"content": {"parts": [{"text": result["answer"]}]}}
        ],
        "sources": result["sources"],
    }

@router.get("/api/sources")
async def sources():
    return service.list_sources()
