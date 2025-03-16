# main.py
import streamlit as st
import pandas as pd
import calendar
from datetime import datetime
import matplotlib.pyplot as plt
from crawling import naver_news_crawler  # 크롤링 모듈 import
from preprocessing import preprocess_dataframe  # 전처리 모듈 import
from word_cloud import generate_wordcloud  # 워드클라우드 모듈 추가
from lda import train_lda_and_visualize  # LDA 모듈 추가
import streamlit.components.v1 as components  # HTML 삽입을 위한 components 사용
from sentiment_analysis import perform_sentiment_analysis, calculate_sentiment_ratio  # 감성 분석 모듈 추가
import predict  # 예측 모듈 추가가

# 스트림릿 페이지 설정
st.set_page_config(page_title="전기차 업계 분석", layout="wide")

# 메인 제목
st.title("🚗 전기차 업계 분석")
st.write("특정 기간과 브랜드, 차종을 선택하면 네이버 뉴스 기사 100개를 크롤링 후 감정분석 결과와 전 판매량의 추세를 바탕으로 판매량을 예측합니다.")

## --------------- 사용자 입력 (년, 월, 브랜드, 차종 선택) --------------- ##
st.header("📰 관련 뉴스 크롤링 후 감정 분석")
# 사용자가 선택할 수 있는 옵션 설정
year_options = ["선택"] + list(range(2023, datetime.today().year + 1))  # 2023년부터 현재까지
month_options = ["선택"] + list(range(1, 13))  # 1월 ~ 12월
brand_options = ["선택", "현대", "기아"]
model_options = {"현대": ["아이오닉5", "아이오닉6"], "기아": ["EV6", "EV9"]}

# 스트림릿 UI
col1, col2, col3 = st.columns(3)

with col1:
    selected_year = st.selectbox("📅 년도 선택", year_options)
with col2:
    selected_month = st.selectbox("📅 월 선택", month_options)
with col3:
    selected_brand = st.selectbox("🚗 브랜드 선택", brand_options)

# 차종 선택 (브랜드를 먼저 선택해야 보임)
selected_model = None
if selected_brand != "선택":
    selected_model = st.selectbox("🚘 차종 선택", ["선택"] + model_options[selected_brand])

# 모든 옵션이 선택되었는지 확인
if selected_year != "선택" and selected_month != "선택" and selected_brand != "선택" and selected_model != "선택":

    # 해당 월의 마지막 날짜 자동 설정
    last_day = calendar.monthrange(int(selected_year), int(selected_month))[1]
    s_date = f"{selected_year}.{int(selected_month):02d}.01"
    e_date = f"{selected_year}.{int(selected_month):02d}.{last_day}"

    # 크롤링 키워드 생성
    search_query = f"{selected_brand} {selected_model}"

    # 크롤링 버튼
    if st.button("📰 뉴스 기사 크롤링 시작"):
        st.write(f"🔍 검색 키워드: {search_query}")
        st.write(f"📅 크롤링 기간: {s_date} ~ {e_date}")

        # 크롤링 실행 (crawling.py의 함수 호출)
        news_df = naver_news_crawler(search_query, s_date, e_date)
        
        # 크롤링 직후 데이터 표시
        st.write(f"🔹 총 {len(news_df)}개 뉴스 기사 크롤링 완료!")
        st.dataframe(news_df)

        # 전처리 
        processed_df = preprocess_dataframe(news_df, "제목")
        st.write(f"🔹 총 {len(news_df)}개 뉴스 기사 전처리 완료!")
        # st.dataframe(news_df)

        # CSV 다운로드 버튼 추가
        csv = news_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 CSV 다운로드", data=csv, file_name=f"{search_query}_{s_date}.csv", mime="text/csv")

        ## --------------- 워드클라우드 --------------- ##
        st.write("")
        st.header("📊 워드클라우드")
        generate_wordcloud(processed_df)

        ## --------------- LDA 토픽 모델링 --------------- ##
        st.write("")
        st.header("📊 LDA 토픽 모델링")
        with st.spinner("LDA 모델 학습 중... ⏳"):
            lda_html = train_lda_and_visualize(processed_df)
        components.html(lda_html, height=800, scrolling=True)

        ## --------------- 감정 분석 --------------- ##
        st.write("")
        st.header("📊 감정 분석")
        processed_df = perform_sentiment_analysis(processed_df)

        # 감정분석 결과 데이터프레임 출력
        st.subheader("📝 감정 분석 결과")
        # st.dataframe(processed_df[["제목", "sentiment_label_sentiword", "sentiment_textblob", "sentiment_vader", "sentiment_flair", "majority_sentiment"]])
        st.dataframe(processed_df[["제목", "majority_sentiment"]])

        # 감정 비율 계산 
        sentiment_result = calculate_sentiment_ratio(processed_df)

        st.write(f"😊 **Positive 비율:** {sentiment_result['positive_ratio']}%")
        st.write(f"😢 **Negative 비율:** {sentiment_result['negative_ratio']}%")
        st.write(f"⚖️ **Positive/Negative:** {sentiment_result['pnr']}")

        # 감정 분석 결과 시각화 (파이 차트)
        # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # font_path = os.path.join(BASE_DIR, "fonts", "NanumGothic.ttf")
        # plt.rcParams["font.family"] = font_path
        
        # fig, ax = plt.subplots()
        # ax.pie([sentiment_result["positive_ratio"], sentiment_result["negative_ratio"]], labels=["Positive", "Negative"], autopct="%1.1f%%", colors=["#A7C7E7", "#FFB6B9"])
        # ax.set_title("감정분석 결과 분포", fontproperties=plt.rcParams["font.family"])
        # st.pyplot(fig)
else:
        st.warning("⚠️ 모든 옵션을 선택해주세요.")

## --------------- 판매량 예측 --------------- ##
st.write("")
st.header("📈 판매량 예측")
        
uploaded_file = st.file_uploader("📤 판매량 + 감정 분석 CSV 업로드 (예측할 부분은 빈칸으로 남겨주세요)", type=["csv"])

st.write("##### 📌 필요한 컬럼")
st.write("- **year_month**: 연-월 (예: 'Jan-23')")
st.write("- **sales**: 판매량")
st.write("- **pnr_naver**: 네이버 뉴스 감정분석 결과 (Positive/Negative 값)")
st.write("- **previous_month_sales**: 전월 판매량")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 업로드된 CSV 표시
    st.write("📝 업로드된 데이터 샘플")
    st.dataframe(df.head())

    # 예측 실행 버튼 추가 (사용자가 클릭하면 실행)
    if st.button("📈 판매량 예측 시작"):
        predict.predict_sales(df)
