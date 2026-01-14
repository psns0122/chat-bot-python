# import os
#
# import uvicorn
# from dotenv import load_dotenv
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from google import genai
#
# # .env 에서 API key 로드
# load_dotenv("../.env")
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     raise RuntimeError("GOOGLE_API_KEY가 .env에 없습니다.")
#
# # Gemini 클라이언트 초기화
# client = genai.Client(api_key=api_key)
#
#
# # FastAPI 관련
# backend = FastAPI()
#
# origins = [
#     "http://localhost",
#     "http://localhost:8080",
# ]
#
# backend.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# class ChatRequest(BaseModel):
#     sessionId: str
#     message: str
#
# @backend.post("/api/get-gemini")
# def get_gemini(req: ChatRequest):
#     # Gemini 호출
#     response = client.models.generate_content(
#         model="gemini-3-flash-preview",
#         contents=req.message
#     )
#
#     return {
#         "candidates": [
#             {
#                 "content": {
#                     "parts": [
#                         {"text": response.text}
#                     ]
#                 }
#             }
#         ]
#     }
#
# if __name__ == "__main__":
#
#     # result = client.models.embed_content(
#     #     model="gemini-embedding-001",
#     #     contents="What is the meaning of life?"
#     # )
#     #
#     # print(result.embeddings)
#
#     uvicorn.run(backend, host="0.0.0.0", port=8080)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.controller.ChattingController import router as chatting_router

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatting_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
