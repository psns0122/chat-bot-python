from __future__ import annotations
from typing import Deque, Dict, Any, List
from collections import deque
import asyncio

from google import genai

from backend.core.config import GOOGLE_API_KEY
from backend.repository.IngestRepository import IngestRepository

SYSTEM_PROMPT = (
    "너는 주식회사 '브릭(BRIQUE)'의 공식 챗봇이다.\n"
    "- 처음 1회만 짧게 인사하고, 이후에는 자기소개를 반복하지 않는다. 저희 브릭(X) 브릭(O)\n"
    "- 사용자의 질문에 바로 답하고, 필요하면 이전 대화를 이어서 답한다.\n"
    "- 제공된 [브릭 회사 정보]와 통상적인 상식을 더한 범위 내에서 답변한다."
    "- 단어의 의미를 물어봤을 경우, 브릭에 특화된 단어가 아니라면 상식 범위에서 답변한다."
    "- 답변을 찾아내는데 실패했을 경우에만, 홈페이지 (http://www.brique.co.kr/) 참고를 안내한다. 링크 괄호 앞뒤로 띄어쓰기를 넣는다.\n"
    "- 답변은 한국어로, 불필요한 서론 없이 간결하게. 하지만 공손하게.\n"
)

DEFAULT_URLS = [
    "https://www.brique.co.kr/company/",
    "https://www.brique.co.kr/products/",
    "https://www.brique.co.kr/products/adc/",
    "https://www.brique.co.kr/products/fdc/",
    "https://www.brique.co.kr/company/location/",
    "https://www.brique.co.kr/services/",
]

class ChattingService:
    def __init__(self):
        self._client = genai.Client(api_key=GOOGLE_API_KEY)
        self._repo = IngestRepository()

        self._cached_web_data: str = ""
        self._lock = asyncio.Lock()

        # sessionId -> deque(messages)
        self._memory: Dict[str, Deque[Dict[str, Any]]] = {}

    def _content(self, role: str, text: str) -> Dict[str, Any]:
        return {
            "role": role,
            "parts": [{"text": text}]
        }

    def _trim_history(self, history: Deque[Dict[str, Any]], max_messages: int = 12) -> None:
        # Java 코드 로직 그대로: SYSTEM_PROMPT(첫 번째)는 유지하고, 그 다음(가장 오래된 대화)을 제거
        while len(history) > max_messages:
            if len(history) <= 1:
                break
            first = history.popleft()
            _second = history.popleft()
            history.appendleft(first)

    async def ingest_from_urls(self, urls: List[str]) -> None:
        if self._cached_web_data:
            return

        # 동시 호출 방지 (여러 요청이 동시에 들어오면 중복 파싱 방지)
        async with self._lock:
            if self._cached_web_data:
                return

            web_data = await asyncio.to_thread(self._repo.fetch_and_parse_urls, urls)
            self._cached_web_data = web_data

    async def ask(self, session_id: str, user_input: str) -> str:
        # 1) 웹 데이터 파싱 (필요 시 1회만)
        await self.ingest_from_urls(DEFAULT_URLS)

        # 2) 세션 히스토리 준비 (세션당 1회 system prompt)
        history = self._memory.get(session_id)
        if history is None:
            history = deque()
            history.append(self._content("user", SYSTEM_PROMPT))
            self._memory[session_id] = history

        # 3) 프롬프트 재조합
        user_message = (
            "[브릭 회사 정보]\n"
            f"{self._cached_web_data}\n\n"
            f"사용자 질문: {user_input}"
        )
        history.append(self._content("user", user_message))
        self._trim_history(history, 12)

        # 4) Gemini 호출
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            # model="gemini-3-flash-preview",
            model="gemini-2.5-flash-lite",
            contents=list(history),
        )

        model_text = (response.text or "").strip()
        if model_text:
            history.append(self._content("model", model_text))
            self._trim_history(history, 12)

        return model_text
