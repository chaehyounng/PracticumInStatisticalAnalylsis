import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections
import streamlit as st
import os

def generate_wordcloud(data):
    """ 뉴스 데이터에서 토큰을 추출하여 워드클라우드 생성 """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리
    font_path = os.path.join(BASE_DIR, "fonts", "NanumGothic.ttf")

    # 'token_lst' 컬럼 활용 (리스트 형태로 저장된 토큰들)
    all_tokens = []
    for tokens in data['token_lst']:
        all_tokens.extend(tokens)  # 리스트 합치기

    # 단어 빈도수 계산
    word_counts = collections.Counter(all_tokens)

    # 워드클라우드 생성
    wordcloud = WordCloud(
        font_path = font_path,
        background_color='white',
        max_words=100,
        max_font_size=50,
        scale=3,
        random_state=1
    ).generate_from_frequencies(word_counts)

    # Streamlit에서 출력
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    # Streamlit에 출력
    st.pyplot(fig)
