import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# .env 파일 로드
load_dotenv()

# 1. 문서 로드
# data 폴더 바로 아래 pdf만 로드 (하위 폴더는 무시)
PDF_DIR = "../data"  # 여기에 pdf 폴더 경로
loader = DirectoryLoader(
    PDF_DIR,
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)

docs = loader.load()

# 2. 텍스트 분할 (Chunking)
# 긴 문서를 작은 조각(청크)으로 나누어 검색 효율을 높입니다.
# 각 청크 사이즈는 500자, 청크 간에는 200자씩 겹치는 상황 가정
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len)
all_splits = text_splitter.split_documents(docs)

# 3. 임베딩 및 벡터 저장소 생성
# OpenAI 임베딩 모델을 사용하여 텍스트 청크를 벡터로 변환하고 ChromaDB에 저장합니다.
# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_DIR = "./chroma_db"
if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    # 이미 벡터 DB가 존재하면 재사용 (임베딩 다시 안 함)
    print("기존 Chroma DB 발견, 기존 DB를 사용해 탐색을 진행합니다.")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
else:
    # 벡터 DB가 없으면 새로 생성
    print("새로운 Chroma DB를 생성합니다. (임베딩 수행 중)")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# 4. LLM 및 프롬프트 설정
# OpenAI의 강력한 LLM 모델을 사용합니다.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    top_p=0.95,
    max_output_tokens=800
)



# RAG 프롬프트 템플릿 정의
SYSTEM_PROMPT = (
    "너는 주식회사 '브릭(BRIQUE)'의 공식 챗봇이다.\n"
    "- 처음 1회만 짧게 인사하고, 이후에는 자기소개를 반복하지 않는다. 저희 브릭(X) 브릭(O)\n"
    "- 사용자의 질문에 바로 답하고, 필요하면 이전 대화를 이어서 답한다.\n"
    "- 제공된 [브릭 회사 정보] 범위 내에서 답변한다."
    "- 단어의 의미를 물어봤을 경우, 브릭에 특화된 단어가 아니라면 상식 범위에서 답변한다."
    "- 답변을 찾아내는데 실패했을 경우에만, 홈페이지 (http://www.brique.co.kr/) 참고를 안내한다. 링크 괄호 앞뒤로 띄어쓰기를 넣는다.\n"
    "- 답변은 한국어로, 공손하게. 글자수는 대략 200~300 글자수가 되도록. \n"
    "- {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)

# 5. RAG 체인 구축
# 검색된 문서를 프롬프트에 채워넣는 체인을 만듭니다.
def _format_docs(docs):
    # retriever가 반환한 Document 리스트를 프롬프트의 {context}에 넣을 문자열로 변환
    return "\n\n".join(
        f"[source={d.metadata.get('source')}, page={d.metadata.get('page')}] {d.page_content}"
        for d in docs
    )

# 입력 {"input": "..."} -> 질문 문자열만 뽑아 retriever에 전달
_input_to_question = RunnableLambda(lambda x: x["input"])

# retriever는 질문 문자열을 받아 Document 리스트를 반환
_context_chain = _input_to_question | retriever

# 프롬프트에 넣을 {context} 문자열로 변환
_context_to_text = _context_chain | RunnableLambda(_format_docs)

# 최종적으로 LLM 호출 후 answer 텍스트만 뽑기
_answer_chain = (
    {"context": _context_to_text, "input": _input_to_question}
    | prompt
    | llm
    | StrOutputParser()
)

# create_retrieval_chain처럼 결과 딕셔너리 형태를 유지: {"answer": ..., "context": ...}
def rag_invoke(x):
    docs = _context_chain.invoke(x),
    answer = _answer_chain.invoke({
        "input": x["input"],
        "context": docs,
    })
    return {"answer": answer, "context": docs}
rag_chain = RunnableLambda(rag_invoke)

# 6. 질문 및 답변/출처 생성
question = ("연구노트를 작성하려고 하는데 어떤 노트를 살까?")
response = rag_chain.invoke({"input": question})
print(f"\n질문: {question}\n")
print(f"\n답변: {response['answer']}\n")

docs = retriever.invoke(question)
print(f"[근거 자료]\n검색된 문서 수: {len(docs)}")
for i, d in enumerate(docs, start=1):
    meta = d.metadata or {}
    source = meta.get("source", "UNKNOWN_SOURCE")
    page = meta.get("page", meta.get("page_number", "UNKNOWN_PAGE"))
    print(f"[{i}] 출처: {source} | page: {page+1}")