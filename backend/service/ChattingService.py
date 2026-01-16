from __future__ import annotations
from typing import Deque, Dict, Any, List, Optional, Set
from collections import deque
import asyncio
import os
import shutil
import hashlib
import json

from google import genai
from google.genai import types

from backend.core.config import GOOGLE_API_KEY
from backend.service.CardService import CardService, CardRecord

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
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

# ====== Paths / Collections ======
DATA_DIR = "./data"
FILE_DIR = DATA_DIR

CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
COLL_CARDS = "brique_cards"
COLL_CONTENT = "brique_content"

# 마커(ingest skip용)
CARD_MARK = os.path.join(CHROMA_DIR, ".card_ingested")
CONTENT_MARK = os.path.join(CHROMA_DIR, ".content_ingested")

# Parent 저장소(문맥 복원용)
PARENT_STORE_PATH = os.path.join(CHROMA_DIR, "parent_store.json")


class ChattingService:
    def __init__(self):
        self._client = genai.Client(api_key=GOOGLE_API_KEY)
        self._lock = asyncio.Lock()

        # sessionId -> deque(messages)
        self._memory: Dict[str, Deque[Dict[str, Any]]] = {}

        # Splitter / Embeddings
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
        )
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
        )

        self._embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vectorstores (카드 / 내용 분리)
        self._vs_cards = Chroma(
            collection_name=COLL_CARDS,
            persist_directory=CHROMA_DIR,
            embedding_function=self._embeddings,
        )
        self._vs_content = Chroma(
            collection_name=COLL_CONTENT,
            persist_directory=CHROMA_DIR,
            embedding_function=self._embeddings,
        )

        # 카드 서비스
        self._card_service = CardService(DATA_DIR)

        # Parent store (doc_id -> parent_id -> record)
        self._parent_store: Dict[str, Dict[str, Any]] = self._load_parent_store()

    # ---------- Utils ----------
    def _content(self, role: str, text: str) -> Dict[str, Any]:
        return {"role": role, "parts": [{"text": text}]}

    def _trim_history(self, history: Deque[Dict[str, Any]], max_messages: int = 12) -> None:
        while len(history) > max_messages:
            if len(history) <= 1:
                break
            first = history.popleft()
            _second = history.popleft()
            history.appendleft(first)

    def _marked(self, path: str) -> bool:
        return os.path.exists(path)

    def _mark(self, path: str) -> None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("ok")

    def _load_parent_store(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(PARENT_STORE_PATH):
            return {}
        try:
            with open(PARENT_STORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_parent_store(self) -> None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        with open(PARENT_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(self._parent_store, f, ensure_ascii=False, indent=2)

    def _hash(self, s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def _make_parent_id(self, doc_id: str, content_source: str, parent_text: str) -> str:
        base = f"{doc_id}|{content_source}|{parent_text}"
        return self._hash(base)

    def _format_context_from_parents(self, parent_records: List[Dict[str, Any]], max_chars: int = 3200) -> str:
        parts: List[str] = []
        total = 0
        for r in parent_records:
            head = f"[pdf | source={r.get('source','')}] "
            chunk = head + (r.get("text") or "")
            if total + len(chunk) > max_chars:
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts)

    # ---------- Ingest ----------
    async def ingest(self, file_dir: str, force_rebuild: bool = False) -> None:
        async with self._lock:
            if force_rebuild:
                if os.path.exists(CHROMA_DIR):
                    shutil.rmtree(CHROMA_DIR, ignore_errors=True)

                # vectorstore 재오픈
                self._vs_cards = Chroma(
                    collection_name=COLL_CARDS,
                    persist_directory=CHROMA_DIR,
                    embedding_function=self._embeddings,
                )
                self._vs_content = Chroma(
                    collection_name=COLL_CONTENT,
                    persist_directory=CHROMA_DIR,
                    embedding_function=self._embeddings,
                )
                self._parent_store = {}

            need_cards = force_rebuild or (not self._marked(CARD_MARK))
            need_content = force_rebuild or (not self._marked(CONTENT_MARK))

            if not need_cards and not need_content:
                return

            # 1) 카드 ingest
            if need_cards:
                cards: List[CardRecord] = self._card_service.load_cards()
                card_docs = self._card_service.to_documents(cards)
                if card_docs:
                    await asyncio.to_thread(self._vs_cards.add_documents, card_docs)
                    self._mark(CARD_MARK)
                    print(f"[INGEST] cards added: {len(card_docs)}")
                else:
                    print("[INGEST] cards empty (data/metadata 확인 필요)")

            # 2) 내용 ingest (Parent-Child)
            if need_content:
                # PDF 로드
                pdf_loader = DirectoryLoader(
                    file_dir,
                    glob="content/**/*.pdf",
                    loader_cls=PyPDFLoader,
                )
                pdf_docs = pdf_loader.load()
                for d in pdf_docs:
                    d.metadata = d.metadata or {}
                    d.metadata["source_type"] = "pdf"

                # doc_id 매칭을 위해: 카드의 content_source(상대경로) -> doc_id 매핑 만들기
                cards = self._card_service.load_cards()
                rel_to_docid: Dict[str, str] = {}
                for c in cards:
                    if c.content_source:
                        rel_to_docid[c.content_source.replace("\\", "/")] = c.doc_id

                # Parent 저장 + Child 임베딩
                child_docs: List[Document] = []
                parent_store_changed = False

                for d in pdf_docs:
                    src = (d.metadata.get("source") or "").replace("\\", "/")
                    # langchain loader는 보통 "./data/content/..." 절대/상대가 섞일 수 있어 relpath로 정규화
                    try:
                        rel = os.path.relpath(src, DATA_DIR).replace("\\", "/")
                    except Exception:
                        rel = src

                    doc_id = rel_to_docid.get(rel)
                    if not doc_id:
                        # 카드 매칭이 안 되면 pdf 파일명 기반으로 doc_id 생성(최소 동작 보장)
                        base = os.path.splitext(os.path.basename(rel))[0]
                        doc_id = base.strip().lower()

                    # Page별 텍스트(d.page_content)는 "한 페이지" 단위임.
                    # 먼저 Parent(큰 덩어리)로 묶고, 그 Parent를 Child로 다시 쪼개 저장
                    parents = self._parent_splitter.split_text(d.page_content)

                    for pi, ptxt in enumerate(parents):
                        parent_id = self._make_parent_id(doc_id, rel, ptxt)

                        # parent store 저장(문맥 복원용)
                        self._parent_store.setdefault(doc_id, {})
                        if parent_id not in self._parent_store[doc_id]:
                            self._parent_store[doc_id][parent_id] = {
                                "doc_id": doc_id,
                                "source": rel,
                                # 원본 페이지 번호(0-based)
                                "page": d.metadata.get("page", None),
                                "text": ptxt,
                            }
                            parent_store_changed = True

                        # child 생성
                        childs = self._child_splitter.split_text(ptxt)
                        for ci, ctxt in enumerate(childs):
                            child_docs.append(
                                Document(
                                    page_content=ctxt,
                                    metadata={
                                        "source_type": "pdf",
                                        "doc_id": doc_id,
                                        "source": rel,
                                        "page": d.metadata.get("page", None),
                                        "parent_id": parent_id,
                                        "parent_index": pi,
                                        "child_index": ci,
                                    },
                                )
                            )

                if child_docs:
                    await asyncio.to_thread(self._vs_content.add_documents, child_docs)
                    self._mark(CONTENT_MARK)
                    if parent_store_changed:
                        self._save_parent_store()
                    print(f"[INGEST] content child docs added: {len(child_docs)}")
                else:
                    print("[INGEST] content empty (data/content 확인 필요)")

    # ---------- Retrieval ----------
    async def _route_docs_by_cards(self, query: str, k_cards: int = 4) -> List[str]:
        """
        1차: 카드 벡터스토어에서 doc_id 후보를 뽑는다.
        """
        try:
            card_hits: List[Document] = await asyncio.to_thread(self._vs_cards.similarity_search, query, k_cards)
        except Exception:
            return []

        doc_ids: List[str] = []
        seen: Set[str] = set()
        for d in card_hits:
            did = (d.metadata or {}).get("doc_id")
            did = (did or "").strip()
            if did and did not in seen:
                seen.add(did)
                doc_ids.append(did)
        return doc_ids

    async def _search_content(self, query: str, doc_ids: Optional[List[str]] = None, k: int = 12) -> List[Document]:
        """
        2차: 내용 벡터스토어 검색
        - doc_ids가 있으면 각 doc_id별로 검색해서 합친 뒤 상위 k개를 사용
        (Chroma $in 필터가 환경에 따라 내부 에러를 내는 경우가 있어 안전하게 구현)
        """
        if doc_ids:
            merged: List[Document] = []
            # doc별로 조금씩만 뽑아서 합침 (전체 k보다 약간 크게)
            per_k = max(3, (k // max(1, len(doc_ids))) + 2)

            for did in doc_ids:
                try:
                    hits = await asyncio.to_thread(
                        self._vs_content.similarity_search,
                        query,
                        per_k,
                        filter={"doc_id": did},  # 단일 값 필터는 안정적으로 동작
                    )
                    merged.extend(hits)
                except Exception:
                    # 특정 doc_id만 필터가 깨져도 전체 실패하지 않게
                    continue

            # 중복 제거(동일 parent_id 기준)
            seen = set()
            uniq: List[Document] = []
            for d in merged:
                pid = (d.metadata or {}).get("parent_id")
                key = pid or (d.page_content[:80] + str(d.metadata))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(d)

            return uniq[:k]

        # 필터 없이 전체 검색
        return await asyncio.to_thread(self._vs_content.similarity_search, query, k)

    def _collect_parent_records(self, child_docs: List[Document], max_parents: int = 6) -> List[Dict[str, Any]]:
        """
        검색된 child들의 parent_id를 모아서 parent_store에서 parent 텍스트를 복원한다.
        """
        parent_ids: List[str] = []
        seen: Set[str] = set()

        for d in child_docs:
            pid = (d.metadata or {}).get("parent_id")
            did = (d.metadata or {}).get("doc_id")
            if not pid or not did:
                continue
            key = f"{did}:{pid}"
            if key not in seen:
                seen.add(key)
                parent_ids.append(key)

        parents: List[Dict[str, Any]] = []
        for key in parent_ids:
            did, pid = key.split(":", 1)
            rec = (self._parent_store.get(did) or {}).get(pid)
            if rec:
                parents.append(rec)
            if len(parents) >= max_parents:
                break

        return parents

    # ---------- Ask ----------
    async def ask_with_sources(self, session_id: str, user_input: str) -> Dict[str, Any]:
        await self.ingest(FILE_DIR, force_rebuild=False)

        # 1) 세션 히스토리
        history = self._memory.get(session_id)
        if history is None:
            history = deque()
            history.append(self._content("user", SYSTEM_PROMPT))
            self._memory[session_id] = history

        # 2) 1차 라우팅(카드)
        routed_doc_ids = await self._route_docs_by_cards(user_input, k_cards=4)

        # 3) 2차 검색(내용) - doc_id 필터
        child_hits = await self._search_content(user_input, doc_ids=routed_doc_ids, k=60)

        # doc_id 필터 결과가 너무 약하면(0개) 전체에서 fallback
        if not child_hits:
            child_hits = await self._search_content(user_input, doc_ids=None, k=60)

        # 4) Parent 복원(문맥)
        parent_records = self._collect_parent_records(child_hits, max_parents=20)
        context = self._format_context_from_parents(parent_records, max_chars=6000)

        # 5) 프롬프트 구성
        user_message = (
            "[참고자료]\n"
            f"{context if context else '(검색 결과 없음)'}\n\n"
            f"사용자 질문: {user_input}"
        )
        history.append(self._content("user", user_message))
        self._trim_history(history, 12)

        # 6) Gemini 호출
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model="gemini-2.5-flash-lite",
            contents=list(history),
            config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.95,
                max_output_tokens=800,
            ),
        )

        answer = (response.text or "").strip()
        if answer:
            history.append(self._content("model", answer))
            self._trim_history(history, 12)

        # 7) 참고자료 표(Parent 기반으로 보여주는 게 사람이 보기 좋음)
        sources: List[Dict[str, Any]] = []
        for i, r in enumerate(parent_records, start=1):
            src = r.get("source", "UNKNOWN_SOURCE")
            page = r.get("page", None)
            base = os.path.basename(src) if isinstance(src, str) else str(src)

            sources.append(
                {
                    "rank": i,
                    "type": "pdf",
                    "source": base,
                    "page": (page + 1) if isinstance(page, int) else (page or ""),
                }
            )

        return {"answer": answer, "sources": sources}

    async def ask(self, session_id: str, user_input: str) -> str:
        r = await self.ask_with_sources(session_id, user_input)
        return r["answer"]

    # ---------- Sources List ----------
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
                    files.append(
                        {
                            "name": name,
                            "relpath": rel.replace("\\", "/"),
                            "type": "pdf" if lower.endswith(".pdf") else "txt",
                            "size": os.path.getsize(full),
                        }
                    )

        files.sort(key=lambda x: (x["type"], x["relpath"]))
        return {"dir": file_dir, "files": files}
