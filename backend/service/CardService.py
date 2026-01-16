from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.documents import Document


def _safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = stem.strip()
    return stem


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _try_json_load(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        # 가끔 txt 앞뒤에 잡문이 붙는 경우 방어
        # JSON 시작/끝을 대충 찾는 보정
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None


@dataclass
class CardRecord:
    doc_id: str
    title: str
    description: str
    expected_q: List[str]
    keywords: List[str]
    meta_source: str              # metadata txt 경로(상대/표시용)
    content_source: Optional[str] # 연결된 pdf 경로(상대/표시용)


class CardService:
    """
    data/metadata 의 txt(JSON)들을 읽어서 '문서 카드' Document를 만든다.
    - doc_id: 원문(pdf)과 연결되는 키
    - content_source: 해당 카드가 가리키는 원문(pdf) 파일(가능하면 자동 매칭)
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.content_dir = os.path.join(data_dir, "content")
        self.meta_dir = os.path.join(data_dir, "metadata")

    def _list_files(self, root: str, exts: Tuple[str, ...]) -> List[str]:
        out: List[str] = []
        if not os.path.exists(root):
            return out
        for r, _, names in os.walk(root):
            for n in names:
                if n.lower().endswith(exts):
                    out.append(os.path.join(r, n))
        return out

    def _build_pdf_index(self) -> List[Tuple[str, str]]:
        """
        return: [(normalized_filename, relpath_from_data), ...]
        """
        pdfs = self._list_files(self.content_dir, (".pdf",))
        idx: List[Tuple[str, str]] = []
        for p in pdfs:
            rel = os.path.relpath(p, self.data_dir).replace("\\", "/")
            idx.append((_normalize(os.path.basename(p)), rel))
        return idx

    def _match_pdf_by_meta_filename(self, meta_path: str, pdf_index: List[Tuple[str, str]]) -> Optional[str]:
        """
        '교육규정 메타데이터.txt' -> '교육규정' 같은 키로 pdf 파일명에 포함된 것을 찾아 매칭.
        """
        meta_stem = _safe_stem(meta_path)
        meta_stem = meta_stem.replace("메타데이터", "").strip()
        key = _normalize(meta_stem)
        if not key:
            return None

        # 1) 포함 매칭 (가장 단순/강력)
        candidates = []
        for norm_name, rel in pdf_index:
            if key in norm_name:
                candidates.append((len(norm_name), rel))

        if candidates:
            # 가장 짧은 파일명(보통 더 직접 매칭) 우선
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        return None

    def load_cards(self) -> List[CardRecord]:
        pdf_index = self._build_pdf_index()
        meta_files = self._list_files(self.meta_dir, (".txt",))

        cards: List[CardRecord] = []
        for mp in meta_files:
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    raw = f.read()
            except Exception:
                continue

            obj = _try_json_load(raw)
            if not obj:
                print(f"[CARD][SKIP] JSON parse failed: {mp}")
                continue
            print(f"[CARD][OK] {mp}")

            title = str(obj.get("title") or "").strip()
            description = str(obj.get("description") or "").strip()
            expected_q = obj.get("expected_q") or []
            keywords = obj.get("keywords") or []

            if not isinstance(expected_q, list):
                expected_q = []
            if not isinstance(keywords, list):
                keywords = []

            # doc_id는 "metadata 파일 stem" 기반으로 안정적으로 생성
            # (원문 파일명은 바뀔 수 있어도, 카드 파일명이 기준이 되면 운영이 편함)
            meta_stem = _safe_stem(mp).replace("메타데이터", "").strip()
            doc_id = _normalize(meta_stem) or _normalize(title) or _normalize(os.path.basename(mp))

            meta_rel = os.path.relpath(mp, self.data_dir).replace("\\", "/")
            content_rel = self._match_pdf_by_meta_filename(mp, pdf_index)

            cards.append(
                CardRecord(
                    doc_id=doc_id,
                    title=title or meta_stem,
                    description=description,
                    expected_q=[str(x).strip() for x in expected_q if str(x).strip()],
                    keywords=[str(x).strip() for x in keywords if str(x).strip()],
                    meta_source=meta_rel,
                    content_source=content_rel,
                )
            )

        return cards

    def to_documents(self, cards: List[CardRecord]) -> List[Document]:
        docs: List[Document] = []
        for c in cards:
            # 카드 텍스트(라우팅용)
            card_text = (
                f"TITLE: {c.title}\n"
                f"DESCRIPTION: {c.description}\n"
                f"EXPECTED_QUESTIONS:\n- " + "\n- ".join(c.expected_q[:30]) + "\n"
                f"KEYWORDS:\n- " + "\n- ".join(c.keywords[:80])
            ).strip()

            docs.append(
                Document(
                    page_content=card_text,
                    metadata={
                        "source_type": "card",
                        "doc_id": c.doc_id,
                        "meta_source": c.meta_source,
                        "content_source": c.content_source or "",
                        "title": c.title,
                    },
                )
            )
        return docs
