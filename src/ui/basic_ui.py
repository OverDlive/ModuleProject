import streamlit as st

# 제목 추가
st.title("안녕하세요, Streamlit 웹페이지입니다!")

# 텍스트 추가
st.write("이것은 Streamlit으로 만든 간단한 웹 애플리케이션입니다.")

# 입력 받기
name = st.text_input("이름을 입력하세요:")

# 버튼 클릭 이벤트
if st.button("제출"):
    st.write(f"환영합니다, {name}님!")
