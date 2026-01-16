from __future__ import annotations
from typing import Deque, Dict, Any, List
from collections import deque
import asyncio
import os
import shutil
import hashlib
from google import genai
from google.genai import types

from backend.core.config import GOOGLE_API_KEY
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

SYSTEM_PROMPT = (
    "너는 주식회사 '브릭(BRIQUE)'의 공식 챗봇이다.\n"
    "- 처음 1회만 짧게 인사하고, 이후에는 자기소개를 반복하지 않는다. 저희 브릭(X) 브릭(O)\n"
    "- 사용자의 질문에 필요하면 이전 대화를 이어서 답한다.\n"
    "- 제공된 [참고자료] 범위 내에서 답변한다.\n"
    "- 단어의 의미를 물어봤을 경우, 브릭에 특화된 단어가 아니라면 상식 범위에서 답변한다.\n"
    "- 답변을 찾아내는데 실패했을 경우에만, 홈페이지 (http://www.brique.co.kr/) 참고를 안내한다. 링크 괄호 앞뒤로 띄어쓰기를 넣는다.\n"
    "- 답변은 한국어로, 공손하게. 글자수는 가급적 250~300자로 맞춘다.\n"
)

CHROMA_DIR = "./data/chroma_db"
COLLECTION_NAME = "brique_knowledge"
FILE_DIR = "./data"

FILE_MARK = os.path.join(CHROMA_DIR, ".file_ingested")

class ChattingService:
    def __init__(self):
        self._client = genai.Client(api_key=GOOGLE_API_KEY)
        self._lock = asyncio.Lock()

        # sessionId -> deque(messages)
        self._memory: Dict[str, Deque[Dict[str, Any]]] = {}

        # Splitter / Embeddings / Vectorstore
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self._vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            embedding_function=self._embeddings,
        )
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 20})

    def _marked(self, path: str) -> bool:
        return os.path.exists(path)

    def _mark(self, path: str) -> None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("ok")

    def _content(self, role: str, text: str) -> Dict[str, Any]:
        return {"role": role, "parts": [{"text": text}]}

    def _trim_history(self, history: Deque[Dict[str, Any]], max_messages: int = 12) -> None:
        while len(history) > max_messages:
            if len(history) <= 1:
                break
            first = history.popleft()
            _second = history.popleft()
            history.appendleft(first)

    def _chroma_has_data(self) -> bool:
        try:
            # langchain_chroma 내부 collection count
            return self._vectorstore._collection.count() > 0
        except Exception:
            # 폴더 기준 fallback
            return os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR)

    def _make_ids(self, docs: List[Document]) -> List[str]:
        # 중복 방지를 위한 deterministic id
        ids: List[str] = []
        for d in docs:
            src = str(d.metadata.get("source", ""))
            st = str(d.metadata.get("source_type", ""))
            pg = str(d.metadata.get("page", ""))
            base = f"{st}|{src}|{pg}|{d.page_content}"
            hid = hashlib.sha1(base.encode("utf-8")).hexdigest()
            ids.append(hid)
        return ids

    async def ingest(self, file_dir: str, force_rebuild: bool = False) -> None:
        async with self._lock:
            # 0) 강제 리빌드면 DB 폴더 삭제 + 마커 삭제
            if force_rebuild:
                if os.path.exists(CHROMA_DIR):
                    shutil.rmtree(CHROMA_DIR, ignore_errors=True)

                # DB 다시 열기
                self._vectorstore = Chroma(
                    collection_name=COLLECTION_NAME,
                    persist_directory=CHROMA_DIR,
                    embedding_function=self._embeddings,
                )
                self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 20})

            # 1) 무엇을 넣어야 하는지 판단(마커 기반)
            need_file = force_rebuild or (not self._marked(FILE_MARK))

            # 이미 있으면 종료
            if not need_file: return

            # 2) file ingest
            if need_file:
                txt_loader = DirectoryLoader(
                    file_dir,  # 데이터가 있는 디렉토리 경로
                    glob='**/*.txt',  # .txt 파일만 선택
                    loader_cls=TextLoader,
                    loader_kwargs={'encoding': 'utf-8'}
                )
                txt_docs = txt_loader.load()
                for d in txt_docs:
                    d.metadata = d.metadata or {}
                    d.metadata["source_type"] = "txt"

                pdf_loader = DirectoryLoader(
                    file_dir,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader
                )
                pdf_docs = pdf_loader.load()
                for d in pdf_docs:
                    d.metadata = d.metadata or {}
                    d.metadata["source_type"] = "pdf"

                all_documents = txt_docs + pdf_docs
                data_splits = self._splitter.split_documents(all_documents)
                for d in data_splits:
                    d.metadata = d.metadata or {}
                    if "source_type" not in d.metadata:
                        src = (d.metadata.get("source") or "").lower()
                        if src.endswith(".pdf"):
                            d.metadata["source_type"] = "pdf"
                        elif src.endswith(".txt"):
                            d.metadata["source_type"] = "txt"
                        else:
                            d.metadata["source_type"] = "file"

                if data_splits:
                    await asyncio.to_thread(self._vectorstore.add_documents, data_splits)
                    self._mark(FILE_MARK)
                    print("[INGEST] file added")
                else:
                    print("[INGEST] file empty (경로/파일 확인 필요)")

    def _format_context(self, docs: List[Document], max_chars: int = 3000) -> str:
        # 너무 길어지면 모델이 컨텍스트를 못 쓰므로 제한
        parts: List[str] = []
        total = 0
        for d in docs:
            src = d.metadata.get("source", "")
            st = d.metadata.get("source_type", "")
            page = d.metadata.get("page", None)
            head = f"[{st} | source={src}"
            if page is not None:
                head += f", page={page + 1}"
            head += "] "

            chunk = head + d.page_content
            if total + len(chunk) > max_chars:
                break
            parts.append(chunk)
            total += len(chunk)

        return "\n\n".join(parts)

    async def ask(self, session_id: str, user_input: str) -> str:
        await self.ingest(FILE_DIR, force_rebuild=False)

        # 1) 세션 히스토리 준비
        history = self._memory.get(session_id)
        if history is None:
            history = deque()
            history.append(self._content("user", SYSTEM_PROMPT))
            self._memory[session_id] = history

        # 2) 검색
        docs = await asyncio.to_thread(self._retriever.invoke, user_input)
        context = self._format_context(docs)

        # 3) 프롬프트 구성
        user_message = (
            "[근거자료]\n"
            f"{context if context else '(검색 결과 없음)'}\n\n"
            f"사용자 질문: {user_input}"
        )
        history.append(self._content("user", user_message))
        self._trim_history(history, 12)

        # 4) Gemini 호출
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model="gemini-2.5-flash-lite",
            contents=list(history),
            config=types.GenerateContentConfig(
                temperature=0.8,
                top_p=0.95,
                max_output_tokens=800
            ),
        )

        model_text = (response.text or "").strip()
        if model_text:
            history.append(self._content("model", model_text))
            self._trim_history(history, 12)

        return model_text

    async def ask_with_sources(self, session_id: str, user_input: str) -> Dict[str, Any]:
        await self.ingest(FILE_DIR, force_rebuild=False)

        history = self._memory.get(session_id)
        if history is None:
            history = deque()
            history.append(self._content("user", SYSTEM_PROMPT))
            self._memory[session_id] = history

        docs = await asyncio.to_thread(self._retriever.invoke, user_input)
        context = self._format_context(docs)

        user_message = (
            "[근거자료]\n"
            f"{context if context else '(검색 결과 없음)'}\n\n"
            f"사용자 질문: {user_input}"
        )
        history.append(self._content("user", user_message))
        self._trim_history(history, 12)

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model="gemini-2.5-flash-lite",
            contents=list(history),
            config=types.GenerateContentConfig(
                temperature=0.8,
                top_p=0.95,
                max_output_tokens=800
            ),
        )

        answer = (response.text or "").strip()
        if answer:
            history.append(self._content("model", answer))
            self._trim_history(history, 12)

        # 근거자료 표용 데이터 만들기
        sources: List[Dict[str, Any]] = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            src = meta.get("source", "UNKNOWN_SOURCE")
            st = meta.get("source_type", "")
            page = meta.get("page", meta.get("page_number", None))

            # pdf page는 0-based일 때가 많아서 +1
            if isinstance(page, int):
                page_disp = page + 1
            else:
                page_disp = page

            base = os.path.basename(src) if isinstance(src, str) else str(src)

            sources.append({
                "rank": i,
                "type": st,
                "source": base,
                "page": page_disp if st == "pdf" else "",
            })

        return {"answer": answer, "sources": sources}

    def list_sources(self, file_dir: str = FILE_DIR) -> Dict[str, Any]:
        if not os.path.exists(file_dir):
            return {"dir": file_dir, "files": []}

        files: List[Dict[str, Any]] = []
        for root, _, names in os.walk(file_dir):
            for name in names:
                lower = name.lower()
                if lower.endswith(".pdf") or lower.endswith(".txt"):
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, file_dir)
                    files.append({
                        "name": name,
                        "relpath": rel.replace("\\", "/"),
                        "type": "pdf" if lower.endswith(".pdf") else "txt",
                        "size": os.path.getsize(full),
                    })

        files.sort(key=lambda x: (x["type"], x["relpath"]))
        return {"dir": file_dir, "files": files}