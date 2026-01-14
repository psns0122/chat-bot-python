import streamlit as st
import requests
import uuid

st.title("BRIQUE ChatBot")

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

    # 4. Spring Boot 백엔드 API 호출 (REST API)
    with st.chat_message("assistant"):
        try:
            payload = {
                "sessionId": st.session_state.session_id,
                "message": prompt
            }
            response = requests.post(
                "http://localhost:8080/api/get-gemini",
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

                # 4. 세션 저장소에도 텍스트만 저장
                st.session_state.messages.append({"role": "assistant", "content": bot_text})
            else:
                st.error(f"백엔드 에러: {response.status_code}\n{response.text}")
        except Exception as e:
            st.error(f"연결 실패: {e}")