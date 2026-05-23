
import streamlit as st

# 페이지 초기화
if "page" not in st.session_state:
    st.session_state.page = "main"
if "query" not in st.session_state:
    st.session_state.query = ""
if "region" not in st.session_state:
    st.session_state.region = ""
if "animal_type" not in st.session_state:
    st.session_state.animal_type = ""

# ─────────────────────────────
# 메인 페이지
# ─────────────────────────────
def main_page():
    st.title("🐾 SafeSight")
    st.caption("놓치지 않는 시선, 연결되는 안전")
    st.divider()

    animal_type = st.radio(
        "동물 종류",
        ["🐶 강아지", "🐱 고양이"],
        horizontal=True
    )

    region = st.selectbox(
        "잃어버린 지역",
        ["지역 선택", "서울", "경기", "인천",
         "부산", "대구", "대전", "광주", "제주", "기타"]
    )

    st.divider()
    st.subheader("🔍 인상착의 입력")
    query = st.text_input(
        "찾는 반려동물 특징을 입력하세요",
        placeholder="예: 흰색 말티즈 빨간 목줄 착용"
    )

    st.subheader("📸 사진 업로드")
    image = st.file_uploader(
        "또는 사진으로 검색",
        type=["jpg", "jpeg", "png"]
    )
    if image:
        st.image(image, caption="업로드된 사진", width=200)

    st.divider()

    if st.button("🔍 탐색 시작", use_container_width=True):
        if region == "지역 선택":
            st.warning("⚠️ 지역을 선택해주세요!")
        elif not query and not image:
            st.warning("⚠️ 특징을 입력하거나 사진을 업로드해주세요!")
        else:
            # 페이지 전환
            st.session_state.page = "result"
            st.session_state.query = query
            st.session_state.region = region
            st.session_state.animal_type = animal_type
            st.rerun()

# ─────────────────────────────
# 결과 페이지
# ─────────────────────────────
def result_page():
    # 뒤로가기 버튼
    if st.button("← 뒤로가기"):
        st.session_state.page = "main"
        st.rerun()

    st.title("검색 결과")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(f'🔍 "{st.session_state.query}"')
        st.caption(f'📍 {st.session_state.region} · {st.session_state.animal_type}')
    with col2:
        st.metric("검색 건수", "3건")

    st.divider()

    # 더미 결과 (나중에 CLIP 결과로 교체)
    results = [
        {
            "id": "2024-03871",
            "location": f"{st.session_state.region} 강남구",
            "found_date": "2일 전 발견",
            "similarity": 92,
            "tags": ["말티즈", "흰색", "목줄"],
            "tag_colors": ["#e6f1fb", "#eaf3de", "#faeeda"],
        },
        {
            "id": "2024-03645",
            "location": f"{st.session_state.region} 송파구",
            "found_date": "4일 전 발견",
            "similarity": 84,
            "tags": ["믹스견", "크림색"],
            "tag_colors": ["#e6f1fb", "#eaf3de"],
        },
        {
            "id": "2024-03512",
            "location": "경기 하남시",
            "found_date": "1주일 전 발견",
            "similarity": 77,
            "tags": ["비숑", "흰색"],
            "tag_colors": ["#e6f1fb", "#eaf3de"],
        },
    ]

    for result in results:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**공고 #{result['id']}**")
                st.caption(f"📍 {result['location']} · {result['found_date']}")

                tag_html = " ".join([
                    f'<span style="background:{color};padding:2px 8px;border-radius:10px;font-size:12px;margin-right:4px">{tag}</span>'
                    for tag, color in zip(result["tags"], result["tag_colors"])
                ])
                st.markdown(tag_html, unsafe_allow_html=True)

            with col2:
                score = result["similarity"]
                color = "#185fa5" if score >= 80 else "#854f0b"
                st.markdown(
                    f'<div style="text-align:right;font-size:24px;font-weight:bold;color:{color}">{score}%</div>',
                    unsafe_allow_html=True
                )
                st.progress(score / 100)

            if st.button(f"상세보기 →", key=result["id"]):
                st.session_state.page = "detail"
                st.session_state.selected = result
                st.rerun()

# ─────────────────────────────
# 페이지 라우팅
# ─────────────────────────────
if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "result":
    result_page()
