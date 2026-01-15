import streamlit as st
import requests
import uuid

BACKEND = "http://localhost:8080"
st.title("BRIQUE ChatBot")

with st.sidebar:
    st.header("참고자료 목록")

    try:
        s = requests.get(f"{BACKEND}/api/sources", timeout=10).json()
        files = s.get("files", [])
        if files:
            for f in files:
                st.write("•", f['name'])
        else:
            st.write("(파일 없음)")
    except Exception as e:
        st.write("목록 로딩 실패:", e)

    if st.button("새로고침"):
        r = requests.post(f"{BACKEND}/api/ingest/rebuild", timeout=300)
        st.success(r.text if r.status_code == 200 else f"실패: {r.status_code}\n{r.text}")

# 1. 채팅 메시지 저장소 초기화
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 기존 대화 내용 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 사용자 입력창
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. 백엔드 API 호출 (REST API)
    with st.chat_message("assistant"):
        try:
            payload = {
                "sessionId": st.session_state.session_id,
                "message": prompt
            }
            response = requests.post(
                f"{BACKEND}/api/get-gemini",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                # 1. response.text 대신 .json()으로 받아야 딕셔너리가 됩니다.
                result_json = response.json()

                # 2. 텍스트 데이터만 추출
                bot_text = result_json["candidates"][0]["content"]["parts"][0]["text"]

                # 3. 화면에는 추출한 텍스트만 출력
                st.markdown(bot_text)

                # 3-1. 응답의 근거자료도 표로 만들어서 출력
                st.subheader("근거 자료")
                sources = result_json.get("sources", [])
                if sources:
                    st.dataframe(sources, use_container_width=True)
                else:
                    st.write("(근거 자료 탐색 실패)")

                # 4. 세션 저장소에도 텍스트만 저장
                st.session_state.messages.append({"role": "assistant", "content": bot_text})
            else:
                st.error(f"백엔드 에러: {response.status_code}\n{response.text}")
        except Exception as e:
            st.error(f"연결 실패: {e}")