import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
from koelectra import sentiment_analysis_koelectra

# OpenAI API Key 설정
OPENAI_API_KEY = "api_key"
client = OpenAI(api_key=OPENAI_API_KEY)

# 감성어 사전 파일 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTIWORD_PATH = os.path.join(BASE_DIR, "SentiWord_info.json")

# 감성어 사전 로드 함수
def load_sentiword_dict():
    try:
        with open(SENTIWORD_PATH, encoding="utf-8-sig", mode="r") as f:
            SentiWord_info = json.load(f)
        
        sentiword_dic = pd.DataFrame(SentiWord_info)
        sentiword_dic["polarity"] = pd.to_numeric(sentiword_dic["polarity"], errors="coerce")
        sentiment_dict = dict(zip(sentiword_dic["word"], sentiword_dic["polarity"]))
        return sentiment_dict
    
    except FileNotFoundError:
        print(f"❌ 감성어 사전 파일을 찾을 수 없습니다: {SENTIWORD_PATH}")
        return None

# 1. 감성어 사전 기반 감정 분석
def calculate_sentiment(tokens, sentiment_dict):
    sentiment_score = sum(sentiment_dict.get(word, 0) for word in tokens)
    return sentiment_score

def classify_sentiment(score):
    if score > 0:
        return 1  # 긍정
    elif score < 0:
        return -1  # 부정
    else:
        return 0  # 중립

# 2. 감정 분석 라이브러리 활용: gpt-4o 활용 번역 자동화화
def gpt_translate_to_english(text):
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a translator."},
                {"role": "user", "content": f"Translate the following text to English. {text}"}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

# 2.1 TextBlob
def sentiment_analysis_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 1
    elif polarity < 0:
        return -1
    else:
        return 0

# 2.2 Vader
analyzer = SentimentIntensityAnalyzer()

def sentiment_analysis_vader(text):
    sentiment = analyzer.polarity_scores(text)
    if sentiment["compound"] > 0.05:
        return 1
    elif sentiment["compound"] < -0.05:
        return -1
    else:
        return 0

# 2.3 Flair
classifier = TextClassifier.load("sentiment")

def sentiment_analysis_flair(text):
    sentence = Sentence(text)
    classifier.predict(sentence)
    sentiment = sentence.labels[0].value
    if sentiment == "POSITIVE":
        return 1
    elif sentiment == "NEGATIVE":
        return -1
    else:
        return 0

# 총 감정 분석 수행 함수 (4가지 방식 + Voting)
def perform_sentiment_analysis(news_df):
    sentiment_dict = load_sentiword_dict()

    # 1. 감성어 사전 분석
    st.markdown("#### 1. 감성어 사전 기반 감정 분석")
    with st.spinner("감성어 사전 기반 감정 분석 수행 중... ⏳"):
        news_df["sentiment_score_sentiword"] = news_df["token_lst"].apply(lambda tokens: calculate_sentiment(tokens, sentiment_dict))
        news_df["sentiment_label_sentiword"] = news_df["sentiment_score_sentiword"].apply(classify_sentiment)
    st.write("✅ 감성어 사전 기반 감정 분석 완료!")

    # 2.0 GPT 번역
    st.markdown("#### 2. 감정 분석 라이브러리")
    st.markdown("##### 2.0 감정 분석 라이브러리를 위한 GPT로 영어 번역")
    with st.spinner("기사 제목을 영어로 번역 중... ⏳"):
        news_df["title_en"] = news_df["제목"].apply(gpt_translate_to_english)
    st.write("✅ GPT 번역 완료!") 

    # 2.1 감정 분석 라이브러리 적용
    st.markdown("##### 2.1 감정 분석 라이브러리 적용")

    with st.spinner("TextBlob 감정 분석 중... ⏳"):
        news_df["sentiment_textblob"] = news_df["title_en"].apply(sentiment_analysis_textblob)
    st.write("  ✅ TextBlob 감정 분석 완료!")

    with st.spinner("Vader 감정 분석 중... ⏳"):
        news_df["sentiment_vader"] = news_df["title_en"].apply(sentiment_analysis_vader)
    st.write("  ✅ Vader 감정 분석 완료!")

    with st.spinner("Flair 감정 분석 중... ⏳"):
        news_df["sentiment_flair"] = news_df["title_en"].apply(sentiment_analysis_flair)
    st.write("  ✅ Flair 감정 분석 완료!")
    
    # 3. Fine-tuned KoELECTRA
    st.markdown("#### 3. Fine-Tuned KoELECTRA 감성 분석 적용")
    with st.spinner("KoELECTRA 감성 분석 중... ⏳"):
        news_df["sentiment_koelectra"] = news_df["제목"].apply(sentiment_analysis_koelectra)
    st.write("  ✅ KoELECTRA 감성 분석 완료!")


    # 4. 최종 Voting (다수결)
    st.markdown("#### 4. 감정 분석 결과 투표")
    with st.spinner("투표 중... ⏳"):
        sentiment_columns = ["sentiment_label_sentiword", "sentiment_textblob", "sentiment_vader", "sentiment_flair", "sentiment_koelectra"]

        def majority_vote(row):
            counts = row.value_counts()
            max_count = counts.max()
            max_sentiments = counts[counts == max_count].index.tolist()
            return 0 if len(max_sentiments) > 1 else max_sentiments[0]

        news_df["majority_sentiment"] = news_df[sentiment_columns].apply(majority_vote, axis=1)
    st.write("✅ 감정 분석 다수결 결과 도출 완료!")

    return news_df

# 감정 분석 비율 계산 함수
def calculate_sentiment_ratio(news_df):
    total = len(news_df)
    positive_ratio = (news_df["majority_sentiment"] == 1).sum() / total if total > 0 else 0
    negative_ratio = (news_df["majority_sentiment"] == -1).sum() / total if total > 0 else 0
    pnr = (news_df["majority_sentiment"] == 1).sum() / ((news_df["majority_sentiment"] == -1).sum() + 0.1)

    return {
        "positive_ratio": round(positive_ratio * 100, 2),
        "negative_ratio": round(negative_ratio * 100, 2),
        "pnr": round(pnr, 2),
    }
