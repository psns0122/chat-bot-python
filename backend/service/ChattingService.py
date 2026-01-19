from __future__ import annotations
from typing import Deque, Dict, Any, List, Optional, Set, Tuple
from collections import deque
import asyncio
import os
import shutil
import hashlib
import json

from google import genai
from google.genai import types

from backend.core.config import GOOGLE_API_KEY
from backend.service.CardService import CardService, CardRecord, _normalize

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


SYSTEM_PROMPT = (
    "너는 주식회사 '브릭(BRIQUE)'의 공식 챗봇이다.\n"
    "- 처음 1회만 짧게 인사하고, 이후에는 자기소개를 반복하지 않는다. 저희 브릭(X) 브릭(O)\n"
    "- 사용자의 질문에 필요하면 이전 대화를 이어서 답한다.\n"
    "- 제공된 [참고자료]의 제목을 우선적으로 참고하며, 그 다음 내용과 질문의 유사성을 토대로 답변한다.\n"
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
    async def _route_docs_by_cards(self, query: str, k_cards: int = 8) -> List[str]:
        hits = await asyncio.to_thread(self._vs_cards.similarity_search_with_score, query, k_cards)

        q = _normalize(query)
        q_tokens = set(q.split())

        scored: List[Tuple[str, float]] = []
        for doc, score in hits:
            md = doc.metadata or {}
            did = (md.get("doc_id") or "").strip()
            if not did:
                continue

            title_norm = md.get("title_norm") or _normalize(md.get("title") or "")
            kw_norm = set((md.get("keywords_norm") or "").split())

            boost = 0.0
            # 타이틀 직접 매칭
            if any(t in title_norm for t in q_tokens):
                boost += 0.25
            # 키워드 매칭
            for t in q_tokens:
                if t and any(t == k for k in kw_norm):
                    boost += 0.05
            boost = min(boost, 0.35)

            # score가 거리면 낮을수록 좋으니 "score - boost"로 개선
            scored.append((did, score - boost))

        scored.sort(key=lambda x: x[1])
        out, seen = [], set()
        for did, _ in scored:
            if did not in seen:
                seen.add(did)
                out.append(did)
        return out[:4]

    async def _search_content(self, query: str, doc_ids: Optional[List[str]] = None, k: int = 12) -> List[Document]:
        if doc_ids:
            merged: List[Tuple[Document, float]] = []
            per_k = max(3, (k // max(1, len(doc_ids))) + 6)

            for did in doc_ids:
                try:
                    hits = await asyncio.to_thread(
                        self._vs_content.similarity_search_with_score,
                        query,
                        per_k,
                        filter={"doc_id": did},
                    )
                    merged.extend(hits)
                except Exception:
                    continue

            # score 정렬: Chroma는 보통 "거리(distance)"라서 낮을수록 좋을 수 있음.
            # 일단 오름차순 정렬로 시작하고, 결과가 이상하면 내림차순으로 바꿔.
            merged.sort(key=lambda x: x[1])

            # parent_id 기준 중복 제거(더 좋은 점수 유지)
            best: Dict[str, Tuple[Document, float]] = {}
            for d, s in merged:
                pid = (d.metadata or {}).get("parent_id") or d.page_content[:80]
                if pid not in best or s < best[pid][1]:
                    best[pid] = (d, s)

            return [ds[0] for ds in list(best.values())[:k]]

        hits = await asyncio.to_thread(self._vs_content.similarity_search_with_score, query, k)
        hits.sort(key=lambda x: x[1])
        return [d for d, _ in hits]

    def _collect_parent_records(self, child_docs: List[Document], max_parents: int = 6, per_doc: int = 3) -> List[
        Dict[str, Any]]:
        picked: Dict[str, int] = {}
        seen: Set[str] = set()
        parents: List[Dict[str, Any]] = []

        for d in child_docs:
            md = d.metadata or {}
            did, pid = md.get("doc_id"), md.get("parent_id")
            if not did or not pid:
                continue
            if picked.get(did, 0) >= per_doc:
                continue

            key = f"{did}:{pid}"
            if key in seen:
                continue
            seen.add(key)

            rec = (self._parent_store.get(did) or {}).get(pid)
            if rec:
                parents.append(rec)
                picked[did] = picked.get(did, 0) + 1
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
        child_hits = await self._search_content(user_input, doc_ids=routed_doc_ids, k=36)

        # doc_id 필터 결과가 너무 약하면(0개) 전체에서 fallback
        if not child_hits:
            child_hits = await self._search_content(user_input, doc_ids=None, k=36)

        # 4) Parent 복원(문맥)
        parent_records = self._collect_parent_records(child_hits, max_parents=20)
        context = self._format_context_from_parents(parent_records, max_chars=6000)

        # 디버깅용 출력
        # print("[ROUTE]", routed_doc_ids)
        # for i, d in enumerate(child_hits[:10]):
        #     md = d.metadata or {}
        #     print(i, md.get("doc_id"), md.get("source"), md.get("page"), md.get("parent_id"))

        # 5) 프롬프트 구성
        user_message = (
            "[참고자료]\n"
            f"{context if context else '(검색 결과 없음)'}\n\n"
            f"사용자 질문: {user_input}"
        )
        history.append(self._content("user", user_message))
        self._trim_history(history, 12)

        # print(user_message)
        # 6) Gemini 호출
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model="gemini-2.5-flash-lite",
            contents=list(history),
            config=types.GenerateContentConfig(
                temperature=0.3,
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
                if lower.endswith(".pdf"):
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
